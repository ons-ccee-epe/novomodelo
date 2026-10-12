//! Gates every stored Benders cut with a validity, tightness, and mask-soundness oracle.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

mod common;

use novomodelo_core::System;
use novomodelo_io::Config;
use novomodelo_io::config::{
    InflowNonNegativityMethod, SelectionMethod, StoppingRuleConfig, TrainingSelection,
};
use novomodelo_sddp::setup::{NodePos, StageIdx};
use novomodelo_sddp::test_support::decks::committed_decks;
use novomodelo_sddp::test_support::{
    inflow_slack_bounds_for_probe, stage_state_box_bounds,
    write_backward_opening_outcome_with_inflow_slack_bounds_for_probe,
};
use novomodelo_sddp::{DEFAULT_COST_SCALE_FACTOR, StudySetup};
use novomodelo_solver::ActiveSolver;

use common::cut_oracles::{
    apply_oracle_config, apply_oracle_config_with_selection, run_cut_oracles,
};
use common::in_code_studies::{
    ChronologicalNoiseSpec, chronological_noise_hydro_models, chronological_noise_study,
    discounted_anticipated_study, mixed_lead_anticipated_study, negative_draw_par1_spec,
    negative_draw_water_value_spec, parallel_multiblock_evaporation_study,
    stochastic_parallel_study,
};
use common::{StubComm, build_setup_in_code, build_setup_in_code_with_models, fresh_setup_with};

const D19: &str = "examples/deterministic/d19-multi-hydro-par";
const ONE_D_TOY: &str = "examples/1dtoy";
const LEVEL1_FORWARD_PASSES: u32 = 2;
const LEVEL1_ITERATIONS: u32 = 10;
/// Magnitude, in $ per hm³ or per m³/s, that a stored coefficient times
/// `DEFAULT_COST_SCALE_FACTOR` must exceed to count as sloped.
const SLOPED_CUT_MARGIN: f64 = 1_000.0;
/// Two storages and two inflow lags.
const WATER_VALUE_STATE_DIMENSION: usize = 4;

fn deck_setup(key: &str, mutate: impl FnOnce(&mut Config)) -> StudySetup {
    let decks = committed_decks();
    let deck = decks
        .iter()
        .find(|deck| deck.key == key)
        .unwrap_or_else(|| panic!("committed deck {key} is absent"));
    fresh_setup_with(&deck.dir, mutate)
}

fn d19_setup(mutate: impl FnOnce(&mut Config)) -> StudySetup {
    deck_setup(D19, mutate)
}

fn oracle_setup(system: System, mut config: Config) -> StudySetup {
    apply_oracle_config(&mut config);
    build_setup_in_code(system, &config)
}

#[test]
fn cut_oracles_hold_on_the_discounted_anticipated_study() {
    let (system, config) = discounted_anticipated_study();
    let setup = oracle_setup(system, config);
    run_cut_oracles("discounted_anticipated_study", setup)
        .assert_sound("discounted_anticipated_study");
}

#[test]
fn cut_oracles_hold_on_the_parallel_multiblock_evaporation_study() {
    let (system, mut config, hydro_models) = parallel_multiblock_evaporation_study();
    apply_oracle_config(&mut config);
    let setup = build_setup_in_code_with_models(system, &config, hydro_models);
    run_cut_oracles("parallel_multiblock_evaporation_study", setup)
        .assert_sound("parallel_multiblock_evaporation_study");
}

#[test]
fn cut_oracles_hold_on_the_stochastic_parallel_study() {
    let (system, config) = stochastic_parallel_study();
    let setup = oracle_setup(system, config);
    run_cut_oracles("stochastic_parallel_study", setup).assert_sound("stochastic_parallel_study");
}

#[test]
fn cut_oracles_hold_on_deactivated_cuts_under_level1() {
    let setup = deck_setup(ONE_D_TOY, |config| {
        apply_oracle_config_with_selection(
            config,
            SelectionMethod::Level1 {
                tie_tolerance: 1e-10,
                check_frequency: 1,
            },
        );
        config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
            limit: LEVEL1_ITERATIONS,
        }]);
        config.training.selection = Some(TrainingSelection::Sampled {
            forward_passes: LEVEL1_FORWARD_PASSES,
        });
    });
    let report = run_cut_oracles(ONE_D_TOY, setup);
    assert!(
        report.inactive_cuts_checked > 0,
        "{ONE_D_TOY}: level1 left no inactive cut, so the oracle checked none"
    );
    assert!(
        report.pools_with_inactive_successor_cuts > 0,
        "{ONE_D_TOY}: no checked pool's successor holds an inactive cut, so validity ran \
         against the active templates only"
    );
    report.assert_sound(ONE_D_TOY);
}

#[test]
fn cut_oracles_hold_on_the_chronological_noise_study() {
    let (system, config) = chronological_noise_study(&ChronologicalNoiseSpec::default());
    let setup = oracle_setup(system, config);
    run_cut_oracles("chronological_noise_study", setup).assert_sound("chronological_noise_study");
}

#[test]
fn cut_oracles_hold_on_the_mixed_lead_anticipated_study() {
    let (system, config) = mixed_lead_anticipated_study(false);
    let setup = oracle_setup(system, config);
    run_cut_oracles("mixed_lead_anticipated_study", setup)
        .assert_sound("mixed_lead_anticipated_study");
}

#[test]
fn cut_oracles_hold_on_the_chronological_pumping_study() {
    let spec = ChronologicalNoiseSpec {
        pumping_station: true,
        ..ChronologicalNoiseSpec::default()
    };
    let (system, config) = chronological_noise_study(&spec);
    let setup = oracle_setup(system, config);
    run_cut_oracles("chronological_pumping_study", setup)
        .assert_sound("chronological_pumping_study");
}

#[test]
fn cut_oracles_resolve_frozen_corrections_on_the_negative_draw_par1_study() {
    for (label, method) in [
        (
            "negative_draw_par1/truncation",
            InflowNonNegativityMethod::Truncation,
        ),
        (
            "negative_draw_par1/truncation_with_penalty",
            InflowNonNegativityMethod::TruncationWithPenalty,
        ),
    ] {
        let spec = ChronologicalNoiseSpec {
            branching_factor: 4,
            ..negative_draw_par1_spec(method)
        };
        let (system, config) = chronological_noise_study(&spec);
        let setup = oracle_setup(system, config);
        let report = run_cut_oracles(label, setup);
        assert!(
            report.frozen_resolves > 0,
            "{label}: the frozen-correction re-solve branch never ran"
        );
        assert!(
            report.frozen_resolves > report.frozen_infeasible,
            "{label}: every frozen-correction re-solve was infeasible, so no cut was compared"
        );
        report.assert_sound(label);
    }
}

fn water_value_oracle_setup(spec: &ChronologicalNoiseSpec) -> StudySetup {
    let (system, mut config) = chronological_noise_study(spec);
    apply_oracle_config(&mut config);
    let hydro_models = chronological_noise_hydro_models(spec, &system);
    build_setup_in_code_with_models(system, &config, hydro_models)
}

/// `run_cut_oracles` consumes its setup, so the stored cuts are read from a twin trained the
/// same way; returns their count after asserting every coefficient is below `-SLOPED_CUT_MARGIN`.
fn count_sloped_stored_cuts(label: &str, spec: &ChronologicalNoiseSpec) -> usize {
    let mut twin = water_value_oracle_setup(spec);
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = twin
        .train(&mut solver, &StubComm, 1, ActiveSolver::new, None, None)
        .expect("training must return Ok");
    assert!(
        outcome.error.is_none(),
        "{label}: training must not error: {:?}",
        outcome.error
    );
    let mut cuts = 0;
    for (slot, _, coefficients) in twin.fcf.pools[0].active_cuts() {
        assert_eq!(
            coefficients.len(),
            WATER_VALUE_STATE_DIMENSION,
            "{label}: slot {slot} coefficient count"
        );
        for (dimension, coefficient) in coefficients.iter().enumerate() {
            let slope = coefficient * DEFAULT_COST_SCALE_FACTOR;
            assert!(
                slope < -SLOPED_CUT_MARGIN,
                "{label}: slot {slot} dimension {dimension} has slope {slope}, a flat cut"
            );
        }
        cuts += 1;
    }
    cuts
}

#[test]
fn cut_oracles_hold_with_sloped_cuts_on_the_negative_draw_water_value_study() {
    for (label, method) in [
        (
            "negative_draw_water_value/truncation",
            InflowNonNegativityMethod::Truncation,
        ),
        (
            "negative_draw_water_value/truncation_with_penalty",
            InflowNonNegativityMethod::TruncationWithPenalty,
        ),
    ] {
        let spec = ChronologicalNoiseSpec {
            branching_factor: 4,
            ..negative_draw_water_value_spec(method)
        };
        let report = run_cut_oracles(label, water_value_oracle_setup(&spec));
        assert!(
            report.frozen_resolves > 0,
            "{label}: the frozen-correction re-solve branch never ran"
        );
        assert!(
            report.frozen_resolves > report.frozen_infeasible,
            "{label}: every frozen-correction re-solve was infeasible, so no cut was compared"
        );
        report.assert_sound(label);
        assert_eq!(
            count_sloped_stored_cuts(label, &spec),
            report.cuts_checked,
            "{label}: the second training run stored a different number of cuts than the oracle checked"
        );
    }
}

#[test]
fn cut_oracles_hold_on_the_chronological_and_parallel_storage_decks() {
    for key in [
        "crates/novomodelo-sddp/tests/fixtures/chronological_storage",
        "crates/novomodelo-sddp/tests/fixtures/parallel_storage",
    ] {
        let setup = deck_setup(key, apply_oracle_config);
        run_cut_oracles(key, setup).assert_sound(key);
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "trains every committed deck that carries inflow-lag or anticipated state"
)]
fn cut_oracles_hold_on_every_committed_deck_with_lag_or_anticipated_state() {
    let mut selected: Vec<String> = Vec::new();
    let mut any_max_par_order = false;

    for deck in committed_decks() {
        let setup = fresh_setup_with(&deck.dir, apply_oracle_config);
        let (has_anticipated, has_lag) = {
            let state = setup.stage_state();
            (state.n_anticipated > 0, state.max_par_order > 0)
        };
        if !has_anticipated && !has_lag {
            continue;
        }
        if has_lag {
            any_max_par_order = true;
        }

        let report = run_cut_oracles(&deck.key, setup);
        eprintln!(
            "{}: mask drops at least one dimension = {}",
            deck.key,
            report.off_mask_slots_checked > 0
        );
        report.assert_sound(&deck.key);
        selected.push(deck.key.clone());
    }

    eprintln!("selected decks: {selected:?}");
    for required in [
        "examples/deterministic/d34-anticipated-varying-blocks",
        "examples/deterministic/d37-anticipated-commissioning",
        "examples/deterministic/d55-post-study-anticipated-lanes",
    ] {
        assert!(
            selected.iter().any(|k| k == required),
            "deck {required} must be selected by the anticipated/lag criterion"
        );
    }
    assert!(
        any_max_par_order,
        "at least one selected deck must have max_par_order > 0"
    );
}

#[test]
fn inflow_slack_bounds_for_probe_evaluates_the_given_state() {
    let setup = d19_setup(|config| {
        apply_oracle_config(config);
        config.modeling.inflow_non_negativity.method =
            InflowNonNegativityMethod::TruncationWithPenalty;
    });
    let training_ctx = setup.training_ctx();
    let state = setup.stage_state();
    let par = training_ctx.stochastic.par();
    let tree_view = training_ctx.stochastic.tree_view();
    let raw_noise = tree_view.opening(1, 0);

    let psi_sum: Vec<f64> = (0..state.hydro_count)
        .map(|h| par.psi_slice(1, h).iter().sum())
        .collect();
    let guarded: Vec<usize> = (0..state.hydro_count)
        .filter(|&h| psi_sum[h].abs() > 1e-6)
        .collect();
    assert!(
        !guarded.is_empty(),
        "vacuity guard: no hydro has a lag-sensitive PAR(1) model at stage 1"
    );

    let state_with_lag_term = |term: f64| {
        let mut x = training_ctx.initial_state.to_vec();
        for &h in &guarded {
            for lag in 0..state.max_par_order {
                x[state.inflow_lags.start + lag * state.hydro_count + h] = term / psi_sum[h];
            }
        }
        x
    };
    let x_lo = state_with_lag_term(-1.0e6);
    let x_hi = state_with_lag_term(1.0e6);

    let comm = StubComm;
    let mut workspace_pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("create_workspace_pool must succeed");
    let ws = &mut workspace_pool.workspaces[0];

    let (lo_lower, lo_upper) =
        inflow_slack_bounds_for_probe(ws, &training_ctx, StageIdx(1), &x_lo, raw_noise);
    let (_, hi_upper) =
        inflow_slack_bounds_for_probe(ws, &training_ctx, StageIdx(1), &x_hi, raw_noise);
    for &h in &guarded {
        assert!(
            lo_upper[h] > 0.0,
            "hydro {h}: x_lo upper bound {}",
            lo_upper[h]
        );
        assert_eq!(lo_lower[h], 0.0, "hydro {h}: x_lo lower bound");
        assert_eq!(
            hi_upper[h].to_bits(),
            0.0_f64.to_bits(),
            "hydro {h}: x_hi upper bound {}",
            hi_upper[h]
        );
    }
}

#[test]
fn frozen_inflow_slack_bounds_reach_the_successor_lp() {
    let setup = d19_setup(|_| {});
    let training_ctx = setup.training_ctx();
    let stage_ctx = setup.stage_ctx();
    let initial = training_ctx.initial_state;
    let (lo, hi) = stage_state_box_bounds(&setup, 1);
    let canonical_x: Vec<f64> = (0..initial.len())
        .map(|j| {
            if lo[j].is_finite() && hi[j].is_finite() {
                f64::midpoint(lo[j], hi[j])
            } else {
                initial[j]
            }
        })
        .collect();
    let tree_view = training_ctx.stochastic.tree_view();
    let raw_noise = tree_view.opening(2, 0);
    let node_id = setup.inputs.node_graph.node_ids[NodePos(2)];

    let comm = StubComm;
    let mut workspace_pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("create_workspace_pool must succeed");
    let ws = &mut workspace_pool.workspaces[0];

    for bounds in [[1.0, 2.0], [0.0, 0.0]] {
        let (_, slack_primal) = write_backward_opening_outcome_with_inflow_slack_bounds_for_probe(
            ws,
            &stage_ctx,
            &training_ctx,
            &setup.fcf.pools[2],
            &training_ctx.cut_state_layouts[1],
            StageIdx(2),
            node_id,
            &canonical_x,
            raw_noise,
            &bounds,
            &bounds,
        )
        .expect("the successor LP must be feasible with the slack pinned");
        assert_eq!(slack_primal.len(), bounds.len());
        for (v, expected) in slack_primal.iter().zip(bounds) {
            assert!(
                (v - expected).abs() <= 1e-9 * 1.0_f64.max(v.abs()),
                "slack primal {v} != pinned {expected}"
            );
        }
    }
}
