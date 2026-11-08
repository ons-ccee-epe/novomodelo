//! `water_balance_coupling` section tests.
//!
//! Every water-balance row reads the realized inflow through
//! `push_local_inflow_coupling` in `lp/builder/entries.rs`, which records it in the
//! inflow-coupling sink, rather than through an AR-lag column or a PAR base baked
//! into the row's own RHS.

use std::collections::{BTreeSet, HashSet};

use super::*;

use novomodelo_core::System;
use novomodelo_sddp::StudySetup;
use novomodelo_sddp::indexer::{BlockIdx, HydroSys};
use novomodelo_sddp::test_support::template_structure::{
    RowOwner, UnscaledMatrix, hours_to_hm3, row_owners, z_inflow_column,
};

use super::common::{self, Emission};

#[derive(Debug, Default)]
struct SinkPower {
    unequal_chronological_stages: usize,
    prefilling_routes: usize,
    generic_entries: usize,
}

fn check_sink_coupling(
    key: &str,
    emission: Emission,
    system: &System,
    setup: &StudySetup,
    power: &mut SinkPower,
) {
    let state = setup.stage_state();
    let n_hydros = state.hydro_count;
    if n_hydros == 0 {
        return;
    }
    let templates = &setup.inputs.stage_data.stage_templates;
    let coupling_per_stage = setup.stage_ctx().inflow_coupling;

    for (s, t) in templates.templates.iter().enumerate() {
        let geom = &templates.geometry_per_stage[s];
        let coupling = &coupling_per_stage[s];
        let owners = row_owners(system, geom, state);
        let matrix = UnscaledMatrix::of(t);
        let rows_per_entity = geom.water_balance.rows_per_entity(geom.n_blks);
        let block_hours = &templates.block_hours_per_stage[s];
        let zeta = hours_to_hm3(block_hours.iter().sum());
        let generic_row_start = t.num_rows - templates.generic_constraint_row_entries[s].len();
        let water_hydro_of = |r: usize| -> Option<HydroSys> {
            match owners.get(&r) {
                Some(&RowOwner::Water { hydro, .. }) => Some(hydro),
                _ => None,
            }
        };
        let water_row = |d: usize, k: usize| {
            matrix.row(geom.water_balance_row(HydroSys::new(d), BlockIdx::new(k)))
        };
        let is_frozen = |d: usize| -> bool {
            (0..rows_per_entity).all(|k| {
                water_row(d, k)
                    .iter()
                    .filter(|&&(c, _)| !state.z_inflow.contains(&c))
                    .count()
                    == 2
            })
        };

        if emission == Emission::Logical {
            for d in 0..n_hydros {
                let whole_row_frozen = (0..rows_per_entity).all(|k| water_row(d, k).len() == 2);
                assert_eq!(
                    is_frozen(d),
                    whole_row_frozen,
                    "{key} {emission:?}: stage {s} hydro {d}: the frozen predicate must \
                     equal the whole-row entry count"
                );
            }
        }

        for h in 0..n_hydros {
            for (r, v) in coupling.hydro_entries(HydroSys::new(h)) {
                assert!(
                    r >= generic_row_start || water_hydro_of(r).is_some(),
                    "{key} {emission:?}: stage {s} hydro {h}: sink entry at row {r} is on \
                     neither a water row nor a generic row"
                );
                if r >= generic_row_start && v != 0.0 {
                    power.generic_entries += 1;
                }
            }
        }

        let mut checked_water_entries = false;
        for h in 0..n_hydros {
            let water_entries: Vec<(usize, f64)> = coupling
                .hydro_entries(HydroSys::new(h))
                .filter(|&(r, _)| water_hydro_of(r).is_some())
                .collect();

            if water_entries.is_empty() {
                assert!(
                    is_frozen(h),
                    "{key} {emission:?}: stage {s} hydro {h}: no water-row entry, but \
                     hydro {h} is not frozen"
                );
                continue;
            }
            checked_water_entries = true;

            let hydros_hit: BTreeSet<usize> = water_entries
                .iter()
                .map(|&(r, _)| water_hydro_of(r).expect("filtered above").get())
                .collect();
            assert_eq!(
                hydros_hit.len(),
                1,
                "{key} {emission:?}: stage {s} hydro {h}: water-row entries on more than \
                 one hydro's rows: {hydros_hit:?}"
            );
            let d = *hydros_hit
                .iter()
                .next()
                .expect("hydros_hit has exactly one element");

            assert_eq!(
                water_entries.len(),
                rows_per_entity,
                "{key} {emission:?}: stage {s} hydro {h}: must have exactly one entry per \
                 row of hydro {d} ({rows_per_entity} rows), got {}",
                water_entries.len()
            );
            let distinct_rows: BTreeSet<usize> = water_entries.iter().map(|&(r, _)| r).collect();
            assert_eq!(
                distinct_rows.len(),
                water_entries.len(),
                "{key} {emission:?}: stage {s} hydro {h}: water-row entries repeat a row: \
                 {water_entries:?}"
            );

            for &(r, v) in &water_entries {
                let RowOwner::Water { blk, .. } = owners[&r] else {
                    unreachable!("filtered to Water rows above")
                };
                let expected = if rows_per_entity == 1 {
                    -zeta
                } else {
                    -hours_to_hm3(block_hours[blk.get()])
                };
                assert!(
                    (v - expected).abs() < 1e-12 * zeta.abs(),
                    "{key} {emission:?}: stage {s} hydro {h}: coefficient at row {r} = \
                     {v}, expected {expected}"
                );
            }

            let sum: f64 = water_entries.iter().map(|&(_, v)| v).sum();
            let expected_sum = -zeta;
            assert!(
                (sum - expected_sum).abs() <= 1e-12 * zeta.abs(),
                "{key} {emission:?}: stage {s} hydro {h}: sum of water coefficients = \
                 {sum}, expected {expected_sum}"
            );

            if is_frozen(h) {
                assert!(
                    d == h || !is_frozen(d),
                    "{key} {emission:?}: stage {s} hydro {h}: routes to hydro {d}'s water \
                     rows, but hydro {d} is also frozen"
                );
                if d != h {
                    power.prefilling_routes += 1;
                }
            } else {
                assert_eq!(
                    d, h,
                    "{key} {emission:?}: stage {s} hydro {h}: hydro {h} is not frozen, but \
                     its inflow routes to hydro {d}'s water rows"
                );
            }
        }

        if rows_per_entity > 1
            && checked_water_entries
            && block_hours
                .iter()
                .any(|hours| hours.to_bits() != block_hours[0].to_bits())
        {
            power.unequal_chronological_stages += 1;
        }

        for c in state.inflow_lags.clone() {
            for &(r, _) in matrix.col(c) {
                assert!(
                    water_hydro_of(r).is_none(),
                    "{key} {emission:?}: stage {s}: inflow-lag column {c} has an entry on \
                     water row {r}"
                );
            }
        }

        // (d) no (row, col) pair occurs twice on any water row, for any column.
        for (&r, owner) in &owners {
            if matches!(owner, RowOwner::Water { .. }) {
                let mut seen = HashSet::new();
                for &(c, _) in matrix.row(r) {
                    assert!(
                        seen.insert(c),
                        "{key} {emission:?}: stage {s}: water row {r} has a duplicate entry \
                         for column {c}"
                    );
                }
            }
        }
    }
}

#[test]
fn every_study_reads_z_inflow_on_its_water_rows() {
    let mut logical_power = SinkPower::default();
    let mut compact_power = SinkPower::default();
    let count = common::for_each_study_in_both_forms(|key, system, logical, compact| {
        check_sink_coupling(key, Emission::Logical, system, logical, &mut logical_power);
        check_sink_coupling(key, Emission::Compact, system, compact, &mut compact_power);
    });

    assert_eq!(
        count,
        common::expected_study_count(),
        "every committed deck (minus SLOW_DECKS skips) plus every in-code and structural \
         study must be swept"
    );
    for (emission, power) in [
        (Emission::Logical, &logical_power),
        (Emission::Compact, &compact_power),
    ] {
        assert!(
            power.unequal_chronological_stages >= 1
                && power.prefilling_routes >= 1
                && power.generic_entries >= 1,
            "{emission:?}: the sweep must visit a chronological stage with unequal block \
             hours, a PreFilling route and a generic HydroInflow entry, got {power:?}"
        );
    }
}

/// Builder-level structural check on the PAR fixture of
/// `max_par_order_z_inflow_row_has_twelve_lag_entries`: the water rows carry
/// no PAR base (row bounds both 0.0), and the z-inflow row for each hydro
/// carries exactly `par_lp.deterministic_base`.
#[test]
fn water_rows_carry_no_par_base() {
    use novomodelo_core::scenario::AnnualComponent;

    let ar_coeffs: Vec<f64> = vec![0.3, 0.2];
    let ann = AnnualComponent {
        coefficient: 0.5,
        mean_m3s: 80.0,
        std_m3s: 20.0,
    };
    let inflow_models = vec![
        InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: Some(ann),
        },
        InflowModel {
            hydro_id: EntityId(3),
            stage_id: 0,
            mean_m3s: 60.0,
            std_m3s: 15.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: None,
        },
    ];

    let system = two_hydro_par_system(2, inflow_models.clone());
    let stages = system.stages().to_vec();
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let par_lp =
        PrecomputedPar::build(&inflow_models, &stages, &hydro_ids, None).expect("par build ok");

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &par_lp,
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build_stage_templates_resolving_layout ok");

    let t = &result.templates[0];
    let water = result.geometry_per_stage[0].water_balance.range();
    for r in water {
        assert_eq!(
            t.row_lower[r], 0.0,
            "water row {r} row_lower must be 0.0, got {}",
            t.row_lower[r]
        );
        assert_eq!(
            t.row_upper[r], 0.0,
            "water row {r} row_upper must be 0.0, got {}",
            t.row_upper[r]
        );
    }

    let n_h = 2_usize;
    for h in 0..n_h {
        let expected = par_lp.deterministic_base(0, h);
        assert_eq!(
            t.row_lower[h], expected,
            "z row {h} row_lower must equal par_lp.deterministic_base(0, {h}) = {expected}, got {}",
            t.row_lower[h]
        );
    }
}

#[test]
fn stored_inflow_coupling_rescales_to_every_studys_z_columns_and_row_bounds() {
    let count = common::for_each_study(|key, _system, setup| {
        let state = setup.stage_state();
        let ctx = setup.stage_ctx();
        assert_eq!(
            ctx.inflow_coupling.len(),
            ctx.templates.len(),
            "{key}: one inflow coupling per stage"
        );

        for (s, (t, coupling)) in ctx.templates.iter().zip(ctx.inflow_coupling).enumerate() {
            assert_eq!(coupling.n_hydros(), state.hydro_count, "{key}: stage {s}");

            for h in 0..state.hydro_count {
                let h_sys = HydroSys::new(h);
                let col_z = z_inflow_column(state, h_sys);
                let row_z = state.z_inflow_row(h_sys);
                let span = usize::try_from(t.col_starts[col_z]).unwrap()
                    ..usize::try_from(t.col_starts[col_z + 1]).unwrap();
                let mut in_template: Vec<(usize, u64)> = span
                    .map(|k| (usize::try_from(t.row_indices[k]).unwrap(), t.values[k]))
                    .filter(|&(row, _)| row != row_z)
                    .map(|(row, value)| (row, value.to_bits()))
                    .collect();
                let mut rescaled: Vec<(usize, u64)> = coupling
                    .hydro_entries(h_sys)
                    .map(|(row, c)| (row, ((c * t.col_scale[col_z]) * t.row_scale[row]).to_bits()))
                    .collect();
                in_template.sort_unstable();
                rescaled.sort_unstable();
                assert_eq!(
                    in_template, rescaled,
                    "{key}: stage {s} hydro {h}: z column entries off the z row"
                );
            }

            let touched = coupling.touched_rows();
            assert!(
                touched.windows(2).all(|pair| pair[0] < pair[1]),
                "{key}: stage {s}: touched rows must be distinct and ascending"
            );
            assert_eq!(touched.len(), coupling.touched_row_lower().len());
            assert_eq!(touched.len(), coupling.touched_row_upper().len());
            for (i, &row) in touched.iter().enumerate() {
                let row = row as usize;
                assert_eq!(
                    (coupling.touched_row_lower()[i] * t.row_scale[row]).to_bits(),
                    t.row_lower[row].to_bits(),
                    "{key}: stage {s} row {row}: lower bound"
                );
                assert_eq!(
                    (coupling.touched_row_upper()[i] * t.row_scale[row]).to_bits(),
                    t.row_upper[row].to_bits(),
                    "{key}: stage {s} row {row}: upper bound"
                );
            }
        }
    });

    assert_eq!(
        count,
        common::expected_study_count(),
        "every committed deck (minus SLOW_DECKS skips) plus every in-code and structural \
         study must be swept"
    );
}
