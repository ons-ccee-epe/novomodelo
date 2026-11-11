use std::collections::BTreeSet;

use novomodelo_core::System;
use novomodelo_io::config::StoppingRuleConfig;
use novomodelo_sddp::setup::StageIdx;
use novomodelo_sddp::test_support::decks::{SLOW_DECKS, committed_decks};
use novomodelo_sddp::test_support::{
    ProbeCut, ProbeCuts, ProbeOpening, cut_rows_for_slots_into, empty_row_batch,
    opening_cut_for_probe,
};
use novomodelo_sddp::{RiskMeasure, StudySetup};
use novomodelo_solver::{ActiveSolver, SolverInterface};

use crate::common::expanded_solve::{
    CutRowSet, stages_whose_producer_drops_a_state_dimension, stored_incoming_states,
};
use crate::common::in_code_studies::{FphaLagOptions, fpha_lag_study_with};
use crate::common::stored_cut_oracles::apply_oracle_config;
use crate::common::{
    StubComm, build_setup_in_code_with_models, fresh_system_and_setup_with, is_chain,
    train_storing_trial_states,
};

const LAG_STUDY_LABEL: &str = "fpha lag study";
const LAG_STUDY_ITERATIONS: u32 = 5;

struct StageCounts {
    stage: usize,
    bound: usize,
    resident_rows: usize,
    nnz: Vec<usize>,
    touched: Vec<usize>,
    union: usize,
}

// Rationale (too_many_lines): the stage loop's skip and drop bookkeeping feeds the closing
// assertions, so splitting it would thread those locals through one-use helpers.
#[allow(
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::cast_precision_loss
)]
fn record_cut_dual_counts(
    label: &str,
    system: &System,
    setup: &StudySetup,
    states_per_stage: usize,
) -> Vec<StageCounts> {
    let risk_measures: Vec<RiskMeasure> = system
        .stages()
        .iter()
        .filter(|stage| stage.id >= 0)
        .map(|stage| RiskMeasure::from(stage.risk_config))
        .collect();
    let ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let tree = training_ctx.stochastic.tree_view();
    let n_pools = setup.fcf.pools.len();
    let mut workspaces = setup
        .create_workspace_pool(&StubComm, 1, ActiveSolver::new)
        .expect("the workspace pool must build");
    let ws = &mut workspaces.workspaces[0];
    let mut cut = ProbeCut::new(&training_ctx);
    let quantiles = |values: &[usize]| -> [usize; 4] {
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        let at = |percent: usize| sorted[(sorted.len() * percent).div_ceil(100) - 1];
        [sorted[0], at(50), at(99), at(100)]
    };

    let mut dropped = Vec::new();
    let mut recorded = Vec::new();
    for t in 0..n_pools {
        let projection = &training_ctx.cut_state_layouts[t];
        let rows = CutRowSet::active(&setup.fcf, t, t.saturating_sub(1));
        let skipped = if t + 1 == n_pools {
            Some("terminal stage")
        } else if !projection.has_lag_slots() {
            Some("its pool has no lag slots")
        } else if rows.slots.is_empty() {
            Some("its pool holds no active cut")
        } else {
            None
        };
        if let Some(reason) = skipped {
            println!("{label}: stage {t} skipped: {reason}");
            continue;
        }

        let states = if t == 0 {
            vec![training_ctx.initial_state.to_vec()]
        } else {
            let Some(stored) = stored_incoming_states(setup, t) else {
                println!("{label}: stage {t} dropped: its producer pool drops a state dimension");
                dropped.push(t);
                continue;
            };
            if stored.is_empty() {
                println!("{label}: stage {t} skipped: its producer pool stored no state");
                continue;
            }
            let picked = states_per_stage.min(stored.len());
            let wanted: Vec<usize> = (0..picked).map(|i| i * stored.len() / picked).collect();
            stored
                .into_iter()
                .enumerate()
                .filter_map(|(k, x)| wanted.contains(&k).then_some(x))
                .collect()
        };

        if let RiskMeasure::CVaR { alpha, lambda } = risk_measures[t] {
            assert!(
                lambda < 1.0 || alpha >= 1.0,
                "{label}: stage {t}: pure CVaR (lambda {lambda}, alpha {alpha}) can give an \
                 opening zero weight, so the openings with positive weight are not every opening"
            );
        }

        let template = &setup.inputs.stage_data.stage_templates.templates[t];
        let mut batch = empty_row_batch();
        cut_rows_for_slots_into(
            &mut batch,
            &setup.fcf.pools[t],
            &rows.slots,
            setup.stage_state(),
            projection,
            &template.col_scale,
        );
        let mut columns: BTreeSet<usize> = batch
            .col_indices
            .iter()
            .map(|&column| usize::try_from(column).expect("column index is non-negative"))
            .collect();
        columns.remove(&setup.stage_state().theta);
        let bound = 1 + columns.len();
        let resident_rows = rows.slots.len();
        ws.solver.load_model(template);
        ws.solver.add_rows(&batch);

        let openings = tree.n_openings(t);
        let touched_bound = (openings * bound).min(resident_rows);
        let (mut nnz, mut touched, mut union) = (Vec::new(), Vec::new(), BTreeSet::new());
        for (i, x) in states.iter().enumerate() {
            let mut aggregate = BTreeSet::new();
            for omega in 0..openings {
                let context = format!("{label}: stage {t} state {i} opening {omega}");
                opening_cut_for_probe(
                    ws,
                    &ctx,
                    &training_ctx,
                    &ProbeOpening {
                        stage: StageIdx(t),
                        pinned_state: x,
                        raw_noise: tree.opening(t, omega),
                        cuts: ProbeCuts {
                            pool: &setup.fcf.pools[t],
                            pool_id: t,
                            slots: &rows.slots,
                        },
                        extracted_pool_id: rows.extracted_pool,
                        inflow_slack_bounds: None,
                    },
                    &mut cut,
                )
                .unwrap_or_else(|error| {
                    panic!("{context}: the stored-state solve failed: {error:?}")
                });
                let count = cut.cut_dual_slots.len();
                assert!(
                    count <= bound,
                    "{context}: {count} non-zero cut duals exceed B_t {bound} (resident rows \
                     {resident_rows}, largest |y| {})",
                    cut.cut_dual_slots
                        .iter()
                        .map(|&(_, y)| y.abs())
                        .fold(0.0, f64::max)
                );
                nnz.push(count);
                aggregate.extend(cut.cut_dual_slots.iter().map(|&(slot, _)| slot));
            }
            assert!(
                aggregate.len() <= touched_bound,
                "{label}: stage {t} state {i}: {} touched slots exceed min({openings} * B_t \
                 {bound}, resident rows {resident_rows})",
                aggregate.len()
            );
            touched.push(aggregate.len());
            union.extend(aggregate);
        }

        println!(
            "{label}: t={t} R_t={resident_rows} B_t={bound} openings={openings} \
             nnz[min/median/p99/max]={:?} touched[min/median/p99/max]={:?} stage_union={}",
            quantiles(&nnz),
            quantiles(&touched),
            union.len()
        );
        recorded.push(StageCounts {
            stage: t,
            bound,
            resident_rows,
            nnz,
            touched,
            union: union.len(),
        });
    }

    let listed = stages_whose_producer_drops_a_state_dimension(setup);
    for &stage in &dropped {
        assert!(
            listed.contains(&stage),
            "{label}: stage {stage} was dropped but is not among the stages whose producer \
             drops a state dimension {listed:?}: pool state_dimension {}, projection n_slots {}",
            setup.fcf.pools[stage - 1].state_dimension,
            training_ctx.cut_state_layouts[stage - 1].n_slots()
        );
    }
    if !recorded.is_empty() {
        let pooled: Vec<usize> = recorded
            .iter()
            .flat_map(|counts| counts.touched.iter().copied())
            .collect();
        let (worst_stage, worst_ratio) = recorded
            .iter()
            .map(|counts| {
                let peak = counts.nnz.iter().copied().max().unwrap_or(0);
                (counts.stage, peak as f64 / counts.bound as f64)
            })
            .fold(
                (0, 0.0),
                |worst, next| if next.1 > worst.1 { next } else { worst },
            );
        println!(
            "{label}: {} stages recorded, dropped {dropped:?}; max nnz/B_t {worst_ratio:.4} \
             (stage {worst_stage}); touched p99 {}; max stage union {}",
            recorded.len(),
            quantiles(&pooled)[2],
            recorded
                .iter()
                .map(|counts| counts.union)
                .max()
                .unwrap_or(0)
        );
    }
    recorded
}

#[test]
fn cut_dual_counts_stay_within_the_structural_bound() {
    let options = FphaLagOptions {
        openings: 3,
        ..FphaLagOptions::default()
    };
    let (system, mut config, hydro_models) = fpha_lag_study_with(options);
    apply_oracle_config(&mut config);
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
        limit: LAG_STUDY_ITERATIONS,
    }]);
    let mut setup = build_setup_in_code_with_models(system, &config, hydro_models);
    train_storing_trial_states(LAG_STUDY_LABEL, &mut setup);
    let (system, _, _) = fpha_lag_study_with(options);

    let counts = record_cut_dual_counts(LAG_STUDY_LABEL, &system, &setup, 4);
    assert!(
        counts.iter().any(|stage| stage.stage >= 1),
        "{LAG_STUDY_LABEL}: no stage past the first was recorded"
    );
    assert!(
        counts.iter().flat_map(|stage| &stage.nnz).any(|&n| n > 0),
        "{LAG_STUDY_LABEL}: no recorded opening has a non-zero cut dual"
    );
    assert!(
        counts.iter().any(|stage| stage.resident_rows > stage.bound),
        "{LAG_STUDY_LABEL}: no recorded stage has more resident rows than its bound, so a \
         dense dual vector would pass; (stage, R_t, B_t) = {:?}",
        counts
            .iter()
            .map(|stage| (stage.stage, stage.resident_rows, stage.bound))
            .collect::<Vec<_>>()
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "trains every committed deck with lagged inflow"
)]
fn cut_dual_counts_stay_within_the_structural_bound_on_every_lagged_deck() {
    let (mut visited, mut not_chain, mut recorded) = (Vec::new(), Vec::new(), Vec::new());
    let mut some_non_zero = false;
    for deck in committed_decks() {
        if SLOW_DECKS.contains(&deck.key.as_str()) {
            continue;
        }
        let (system, mut setup) = fresh_system_and_setup_with(&deck.dir, apply_oracle_config);
        if setup.stage_state().max_par_order == 0 {
            continue;
        }
        if !is_chain(&setup) {
            println!("{}: skipped, not a chain", deck.key);
            not_chain.push(deck.key);
            continue;
        }
        visited.push(deck.key.clone());
        train_storing_trial_states(&deck.key, &mut setup);
        let counts = record_cut_dual_counts(&deck.key, &system, &setup, 8);
        some_non_zero |= counts.iter().flat_map(|stage| &stage.nnz).any(|&n| n > 0);
        if !counts.is_empty() {
            recorded.push(deck.key);
        }
    }
    println!(
        "visited {} lagged decks: {visited:?}; not chains: {not_chain:?}; recorded: {recorded:?}",
        visited.len()
    );
    assert!(
        !visited.is_empty(),
        "no committed lagged chain deck was visited"
    );
    assert!(
        some_non_zero,
        "no recorded opening of {recorded:?} has a non-zero cut dual"
    );
}
