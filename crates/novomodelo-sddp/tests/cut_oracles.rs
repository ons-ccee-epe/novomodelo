//! Gates every stored Benders cut with a validity, tightness, and mask-soundness oracle.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

mod common;

use std::path::Path;

use novomodelo_core::System;
use novomodelo_io::Config;
use novomodelo_io::config::{
    InflowNonNegativityMethod as CfgInflowMethod, SelectionMethod, StoppingRuleConfig,
    TrainingSelection,
};
use novomodelo_sddp::StudySetup;
use novomodelo_sddp::setup::{NodePos, StageIdx};
use novomodelo_sddp::test_support::decks::committed_decks;
use novomodelo_sddp::test_support::{
    inflow_slack_bounds_for_probe, stage_state_box_bounds,
    write_backward_opening_outcome_with_inflow_slack_bounds_for_probe,
};
use novomodelo_solver::ActiveSolver;

use common::cut_oracles::{
    apply_oracle_config, apply_oracle_config_with_selection, run_cut_oracles,
};
use common::in_code_studies::{
    ChronologicalNoiseSpec, chronological_noise_study, discounted_anticipated_study,
    mixed_lead_anticipated_study, negative_draw_par1_spec, parallel_multiblock_evaporation_study,
    stochastic_parallel_study,
};
use common::{
    StubComm, build_setup_in_code, build_setup_in_code_with_models, fresh_setup_with, positive,
};

const D19_DIR: &str = "../../examples/deterministic/d19-multi-hydro-par";
const ONE_D_TOY: &str = "examples/1dtoy";
const LEVEL1_FORWARD_PASSES: u32 = 2;
const LEVEL1_ITERATIONS: u32 = 10;

fn deck_setup(key: &str, mutate: impl FnOnce(&mut Config)) -> StudySetup {
    let decks = committed_decks();
    let deck = decks
        .iter()
        .find(|deck| deck.key == key)
        .unwrap_or_else(|| panic!("committed deck {key} is absent"));
    fresh_setup_with(&deck.dir, mutate)
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
                check_frequency: positive(1),
                lookback_iterations: None,
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

/// Under `none` the negative-draw PAR(1) study stays feasible (the initial
/// storage covers every draw's loss), so it can serve as the untruncated twin.
#[test]
fn negative_draw_par1_study_trains_without_an_infeasible_solve_under_none() {
    let (system, config) =
        chronological_noise_study(&negative_draw_par1_spec(CfgInflowMethod::None));
    let setup = oracle_setup(system, config);
    run_cut_oracles("negative_draw_par1_study/none", setup)
        .assert_sound("negative_draw_par1_study/none");
}

/// With PAR(1) lags the correction `max(−r, 0)` depends on the trial lag, so
/// sample points re-solve the successor with another cut's frozen slack bounds,
/// and each cut must stay a minorant of its own frozen-correction value.
#[test]
fn frozen_correction_resolves_cuts_at_their_own_trial_lags() {
    for method in [
        CfgInflowMethod::Truncation,
        CfgInflowMethod::TruncationWithPenalty,
    ] {
        let label = format!("negative_draw_par1_study/{method:?}");
        let (system, config) = chronological_noise_study(&negative_draw_par1_spec(method));
        let setup = oracle_setup(system, config);
        let report = run_cut_oracles(&label, setup);
        eprintln!(
            "{label}: frozen_resolves={} frozen_infeasible={}",
            report.frozen_resolves, report.frozen_infeasible
        );
        assert!(
            report.frozen_resolves > 0,
            "{label}: no point re-solved with another cut's frozen correction \
             (frozen_infeasible={})",
            report.frozen_infeasible
        );
        assert!(
            report.frozen_resolves > report.frozen_infeasible,
            "{label}: every frozen-correction re-solve was infeasible, so no cut was compared"
        );
        report.assert_sound(&label);
    }
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

/// The out-of-LP slack-bound evaluator reads the state it is given: lags driving
/// `Σψ·â` to `-1e6` give a positive `TruncationWithPenalty` cap with a zero lower bound, and `+1e6`
/// gives a `+0.0` cap.
#[test]
fn inflow_slack_bounds_for_probe_evaluates_the_given_state() {
    let setup = fresh_setup_with(Path::new(D19_DIR), |config| {
        apply_oracle_config(config);
        config.modeling.inflow_non_negativity.method = CfgInflowMethod::TruncationWithPenalty;
    });
    let mut workspace_pool = setup
        .create_workspace_pool(&StubComm, 1, ActiveSolver::new)
        .expect("create_workspace_pool must succeed");
    let ws = &mut workspace_pool.workspaces[0];
    let training_ctx = setup.training_ctx();
    let state = setup.stage_state();
    let par = training_ctx.stochastic.par();
    let raw_noise = training_ctx.stochastic.tree_view().opening(1, 0).to_vec();

    let lag_driven: Vec<(usize, f64)> = (0..state.hydro_count)
        .map(|h| (h, par.psi_slice(1, h).iter().sum::<f64>()))
        .filter(|&(_, psi_sum)| psi_sum.abs() > 1e-6)
        .collect();
    assert!(
        !lag_driven.is_empty(),
        "vacuity guard: no D19 hydro has a nonzero stage-1 lag coefficient"
    );

    let state_at = |target: f64| {
        let mut x = training_ctx.initial_state.to_vec();
        for &(h, psi_sum) in &lag_driven {
            for lag in 0..par.psi_slice(1, h).len() {
                x[state.inflow_lags.start + lag * state.hydro_count + h] = target / psi_sum;
            }
        }
        x
    };
    let x_lo = state_at(-1.0e6);
    let x_hi = state_at(1.0e6);

    let (lower_lo, upper_lo) =
        inflow_slack_bounds_for_probe(ws, &training_ctx, StageIdx(1), &x_lo, &raw_noise);
    let (_, upper_hi) =
        inflow_slack_bounds_for_probe(ws, &training_ctx, StageIdx(1), &x_hi, &raw_noise);
    for &(h, _) in &lag_driven {
        assert!(
            upper_lo[h] > 0.0,
            "hydro {h}: x_lo must give a positive cap"
        );
        assert_eq!(lower_lo[h], 0.0, "hydro {h}: the TwP lower bound is 0");
        assert_eq!(
            upper_hi[h].to_bits(),
            0.0_f64.to_bits(),
            "hydro {h}: x_hi must give a +0.0 cap"
        );
    }
}

/// The bound-override probe reaches the successor LP: pinning the slack at given
/// bounds returns that slack as the solved primal.
#[test]
fn frozen_inflow_slack_bounds_reach_the_successor_lp() {
    let setup = fresh_setup_with(Path::new(D19_DIR), |_| {});
    let mut workspace_pool = setup
        .create_workspace_pool(&StubComm, 1, ActiveSolver::new)
        .expect("create_workspace_pool must succeed");
    let ws = &mut workspace_pool.workspaces[0];
    let training_ctx = setup.training_ctx();
    let stage_ctx = setup.stage_ctx();
    let (lo, hi) = stage_state_box_bounds(&setup, 1);
    let x: Vec<f64> = (0..lo.len())
        .map(|j| {
            if lo[j].is_finite() && hi[j].is_finite() {
                f64::midpoint(lo[j], hi[j])
            } else {
                training_ctx.initial_state[j]
            }
        })
        .collect();
    let raw_noise = training_ctx.stochastic.tree_view().opening(2, 0).to_vec();

    for pinned in [[1.0_f64, 2.0], [0.0, 0.0]] {
        let (_, slack) = write_backward_opening_outcome_with_inflow_slack_bounds_for_probe(
            ws,
            &stage_ctx,
            &training_ctx,
            &setup.fcf.pools[2],
            &training_ctx.cut_state_layouts[1],
            StageIdx(2),
            setup.inputs.node_graph.node_ids[NodePos(2)],
            &x,
            &raw_noise,
            &pinned,
            &pinned,
        )
        .expect("the override probe must solve");
        assert_eq!(slack.len(), pinned.len());
        for (h, (&got, &want)) in slack.iter().zip(&pinned).enumerate() {
            assert!(
                (got - want).abs() <= 1e-9 * 1.0_f64.max(want.abs()),
                "hydro {h}: slack primal {got}, pinned at {want}"
            );
        }
    }
}
