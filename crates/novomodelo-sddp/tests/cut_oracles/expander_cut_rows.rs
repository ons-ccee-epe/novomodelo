use novomodelo_core::System;
use novomodelo_sddp::indexer::CutSlot;
use novomodelo_sddp::test_support::decks::committed_decks;
use novomodelo_sddp::test_support::empty_row_batch;
use novomodelo_sddp::test_support::expander::{expand_stage_templates, expanded_cut_rows_into};
use novomodelo_sddp::test_support::template_structure::state_lag_dims;
use novomodelo_sddp::{StudySetup, build_cut_row_batch_into};
use novomodelo_solver::RowBatch;

use crate::common::in_code_studies::fpha_lag_study;
use crate::common::stored_cut_oracles::apply_oracle_config;
use crate::common::{
    Emission, build_setup_in_code_with_models_as, fresh_setup_with, fresh_system_and_setup_as,
    train_storing_trial_states,
};

pub(super) use crate::common::in_code_studies::FPHA_LAG_STUDY_KEY as LAG_STUDY;

struct CutRowPower {
    successor_pools_without_lag_cut: Vec<usize>,
    lag_cuts: usize,
    shifted_lag_cuts: usize,
}

pub(super) fn assert_forms_share_cut_geometry(
    key: &str,
    logical: &StudySetup,
    compact: &StudySetup,
) {
    assert_eq!(
        logical.stage_state().nonzero_state_indices,
        compact.stage_state().nonzero_state_indices,
        "{key}: the forms mask different state dimensions"
    );
    let (logical_layouts, compact_layouts) = (
        logical.training_ctx().cut_state_layouts,
        compact.training_ctx().cut_state_layouts,
    );
    assert_eq!(
        logical_layouts.len(),
        compact_layouts.len(),
        "{key}: the forms have different pool counts"
    );
    for (pool, (l, c)) in logical_layouts.iter().zip(compact_layouts).enumerate() {
        assert_eq!(
            l.n_slots(),
            c.n_slots(),
            "{key}: pool {pool}: the forms project different slot counts"
        );
        for j in 0..l.n_slots() {
            let slot = CutSlot::new(j);
            assert_eq!(
                l.global_state_index(slot),
                c.global_state_index(slot),
                "{key}: pool {pool} slot {j}: the forms project different state dimensions"
            );
        }
    }
}

fn sorted_entries(batch: &RowBatch, row: usize) -> Vec<(i32, u64)> {
    let offset = |r: usize| usize::try_from(batch.row_starts[r]).expect("CSR offsets are >= 0");
    let span = offset(row)..offset(row + 1);
    let mut entries: Vec<(i32, u64)> = batch.col_indices[span.clone()]
        .iter()
        .zip(&batch.values[span])
        .map(|(&col, value)| (col, value.to_bits()))
        .collect();
    entries.sort_unstable();
    entries
}

fn assert_rows_match(key: &str, pool: usize, got: &RowBatch, want: &RowBatch) {
    assert_eq!(
        got.num_rows, want.num_rows,
        "{key}: pool {pool}: expanded row count vs today's"
    );
    for row in 0..want.num_rows {
        for (what, g, w) in [
            ("row_lower", got.row_lower[row], want.row_lower[row]),
            ("row_upper", got.row_upper[row], want.row_upper[row]),
        ] {
            assert!(
                g.to_bits() == w.to_bits(),
                "{key}: pool {pool} row {row}: {what} expanded {g:?}, today's {w:?}"
            );
        }
        let (g, w) = (sorted_entries(got, row), sorted_entries(want, row));
        assert_eq!(
            g.len(),
            w.len(),
            "{key}: pool {pool} row {row}: expanded entry count vs today's"
        );
        if let Some(k) = (0..w.len()).find(|&k| g[k] != w[k]) {
            panic!(
                "{key}: pool {pool} row {row}: sorted entry {k}: expanded (col {}, value {:?}), \
                 today's (col {}, value {:?})",
                g[k].0,
                f64::from_bits(g[k].1),
                w[k].0,
                f64::from_bits(w[k].1)
            );
        }
    }
}

fn assert_expanded_cut_rows_match(
    key: &str,
    system: &System,
    logical: &StudySetup,
    compact: &StudySetup,
) -> CutRowPower {
    assert_forms_share_cut_geometry(key, logical, compact);
    let expanded = expand_stage_templates(system, compact);
    let state = expanded.layout.state();
    let lags = state_lag_dims(state);
    let logical_templates = &logical.inputs.stage_data.stage_templates.templates;
    let n_pools = logical.fcf.pools.len();
    assert_eq!(
        n_pools,
        logical_templates.len(),
        "{key}: a pool must render into the template of its own stage"
    );
    let (logical_layouts, compact_layouts) = (
        logical.training_ctx().cut_state_layouts,
        compact.training_ctx().cut_state_layouts,
    );

    let (mut want, mut got) = (empty_row_batch(), empty_row_batch());
    let mut power = CutRowPower {
        successor_pools_without_lag_cut: Vec::new(),
        lag_cuts: 0,
        shifted_lag_cuts: 0,
    };
    for pool in 0..n_pools {
        build_cut_row_batch_into(
            &mut want,
            &logical.fcf,
            pool,
            logical.stage_state(),
            &logical_layouts[pool],
            &logical_templates[pool].col_scale,
        );
        let active_slots: Vec<usize> = logical
            .fcf
            .active_cuts(pool)
            .map(|(slot, _, _)| slot)
            .collect();
        expanded_cut_rows_into(
            &mut got,
            &logical.fcf.pools[pool],
            &active_slots,
            &compact_layouts[pool],
            &expanded.layout,
            &expanded.scaled[pool].col_scale,
        );
        assert_rows_match(key, pool, &got, &want);

        let mut pool_lag_cuts = 0;
        for (_, _, coefficients) in logical.fcf.active_cuts(pool) {
            let deepest_lag = coefficients
                .iter()
                .enumerate()
                .filter_map(|(j, &c)| {
                    let d = logical_layouts[pool].global_state_index(CutSlot::new(j));
                    let rendered = state.nonzero_state_indices.binary_search(&d).is_ok();
                    (c != 0.0 && rendered && lags.contains(&d.get()))
                        .then(|| (d.get() - lags.start) / state.hydro_count)
                })
                .max();
            pool_lag_cuts += usize::from(deepest_lag.is_some());
            power.shifted_lag_cuts += usize::from(deepest_lag >= Some(1));
        }
        power.lag_cuts += pool_lag_cuts;
        if pool + 1 < n_pools && pool_lag_cuts == 0 {
            power.successor_pools_without_lag_cut.push(pool);
        }
    }
    power
}

#[test]
fn expanded_cut_rows_match_todays_cut_rows_on_the_lag_study() {
    let build = |emission| {
        let (system, mut config, hydro_models) = fpha_lag_study();
        apply_oracle_config(&mut config);
        build_setup_in_code_with_models_as(system, &config, hydro_models, emission)
    };
    let mut logical = build(Emission::Logical);
    let compact = build(Emission::Compact);
    train_storing_trial_states(LAG_STUDY, &mut logical);
    let (system, _, _) = fpha_lag_study();

    let power = assert_expanded_cut_rows_match(LAG_STUDY, &system, &logical, &compact);
    assert!(
        power.successor_pools_without_lag_cut.is_empty(),
        "{LAG_STUDY}: pools {:?} hold no active cut with a nonzero lag coefficient",
        power.successor_pools_without_lag_cut
    );
    assert!(
        power.shifted_lag_cuts > 0,
        "{LAG_STUDY}: no active cut has a nonzero coefficient at lag depth 1 or more"
    );
}

#[test]
fn expanded_cut_rows_match_todays_cut_rows_on_every_lag_deck() {
    let mut selected: Vec<String> = Vec::new();
    let (mut lag_cuts, mut shifted_lag_cuts) = (0, 0);
    for deck in committed_decks() {
        let mut logical = fresh_setup_with(&deck.dir, apply_oracle_config);
        if logical.stage_state().max_par_order == 0 {
            continue;
        }
        let (system, compact) = fresh_system_and_setup_as(&deck.dir, Emission::Compact);
        train_storing_trial_states(&deck.key, &mut logical);
        let power = assert_expanded_cut_rows_match(&deck.key, &system, &logical, &compact);
        lag_cuts += power.lag_cuts;
        shifted_lag_cuts += power.shifted_lag_cuts;
        selected.push(deck.key.clone());
    }
    eprintln!(
        "selected {} decks: {selected:?}; lag_cuts={lag_cuts} shifted_lag_cuts={shifted_lag_cuts}",
        selected.len()
    );
    assert!(
        !selected.is_empty(),
        "no committed deck has inflow-lag state"
    );
    assert!(
        lag_cuts > 0,
        "no active cut on any selected deck has a nonzero lag coefficient"
    );
    assert!(
        shifted_lag_cuts > 0,
        "no active cut on any selected deck has a nonzero coefficient at a lag of at least 1"
    );
}
