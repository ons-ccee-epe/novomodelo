//! Gates stored Benders cuts and the stage LP that folds the inflow lags: each stored cut's
//! validity, tightness and mask soundness, and each folded solve's agreement with the expanded LP
//! the test-support expander rebuilds from it.

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
use novomodelo_sddp::setup::{NodePos, StageIdx};
use novomodelo_sddp::test_support::decks::committed_decks;
use novomodelo_sddp::test_support::template_structure::state_lag_dims;
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
use common::{
    StubComm, build_setup_in_code, build_setup_in_code_with_models, fresh_setup_with, positive,
};

const D19_DIR: &str = "../../examples/deterministic/d19-multi-hydro-par";
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

/// The negative-draw study's cuts are flat because its water has no value; with a stage-1 load
/// they are sloped, so the frozen-correction comparator can tell a right reference from a wrong one.
#[test]
fn cut_oracles_hold_with_sloped_cuts_on_the_negative_draw_water_value_study() {
    for method in [
        CfgInflowMethod::Truncation,
        CfgInflowMethod::TruncationWithPenalty,
    ] {
        let label = format!("negative_draw_water_value/{method:?}");
        let spec = ChronologicalNoiseSpec {
            branching_factor: 4,
            ..negative_draw_water_value_spec(method)
        };
        let report = run_cut_oracles(&label, water_value_oracle_setup(&spec));
        assert!(
            report.frozen_resolves > 0,
            "{label}: the frozen-correction re-solve branch never ran"
        );
        assert!(
            report.frozen_resolves > report.frozen_infeasible,
            "{label}: every frozen-correction re-solve was infeasible, so no cut was compared"
        );
        report.assert_sound(&label);
        assert_eq!(
            count_sloped_stored_cuts(&label, &spec),
            report.cuts_checked,
            "{label}: the second training run stored a different number of cuts than the oracle checked"
        );
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
                x[state_lag_dims(state).start + lag * state.hydro_count + h] = target / psi_sum;
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

mod power_guard_checks {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use crate::common::power_guards::{DualFamily, Guard, PowerEvidence};

    const ALL_GUARDS: [Guard; 10] = [
        Guard::BindingLagCut,
        Guard::BindingShiftedLagCut,
        Guard::NonUnitZScale,
        Guard::NonUnitLagScale,
        Guard::RescaledCoupledRowWithDual,
        Guard::NonUnitCompactCoupledRowScaleWithDual,
        Guard::FamilyDual(DualFamily::Fpha),
        Guard::FamilyDual(DualFamily::Evaporation),
        Guard::FamilyDual(DualFamily::Generic),
        Guard::CouplingDualEveryHydro,
    ];

    fn failure_message(evidence: &PowerEvidence, guards: &[Guard]) -> Option<String> {
        let payload = catch_unwind(AssertUnwindSafe(|| {
            evidence.assert_holds("demo gate", guards);
        }))
        .err()?;
        Some(
            *payload
                .downcast::<String>()
                .expect("an assert message is a String"),
        )
    }

    #[test]
    fn complete_evidence_meets_every_guard() {
        let mut evidence = PowerEvidence::new(2, 2);
        evidence.record_cut(1.5, &[3.0, 0.0, 0.0, 0.0]);
        evidence.record_cut(0.5, &[0.0, 0.0, 2.0, 0.0]);
        evidence.record_expanded_scales(&[0.5, 1.0], &[1.0, 2.0]);
        evidence.record_coupled_row(0.7, 2.0, 4.0);
        for family in [
            DualFamily::Fpha,
            DualFamily::Evaporation,
            DualFamily::Generic,
        ] {
            evidence.record_family_dual(family, 1.0);
        }
        evidence.record_coupling_dual(0, 1.0);
        evidence.record_coupling_dual(1, -2.0);

        evidence.assert_holds("demo gate", &ALL_GUARDS);
    }

    #[test]
    fn a_negative_zero_dual_is_not_binding() {
        let coefficients = [1.0, 0.0, 0.0, 0.0];
        let mut evidence = PowerEvidence::new(2, 2);
        evidence.record_cut(-0.0, &coefficients);
        evidence.record_cut(1.0, &[-0.0; 4]);
        let message = failure_message(&evidence, &[Guard::BindingLagCut])
            .expect("neither a -0.0 dual nor a -0.0 coefficient is binding");
        assert!(message.contains("[BindingLagCut]"), "{message}");

        evidence.record_cut(1.0, &coefficients);
        evidence.assert_holds("demo gate", &[Guard::BindingLagCut]);
    }

    #[test]
    fn a_lag_zero_only_cut_fails_the_shifted_guard() {
        let mut evidence = PowerEvidence::new(2, 2);
        evidence.record_cut(1.0, &[2.0, 0.0, 0.0, 0.0]);
        evidence.record_cut(1.0, &[0.0, 2.0, 0.0, 0.0]);
        evidence.assert_holds("demo gate", &[Guard::BindingLagCut]);
        let message = failure_message(&evidence, &[Guard::BindingShiftedLagCut])
            .expect("a lag-0 coefficient is not shifted");
        assert!(message.contains("[BindingShiftedLagCut]"), "{message}");

        evidence.record_cut(1.0, &[0.0, 0.0, 2.0, 0.0]);
        evidence.assert_holds("demo gate", &[Guard::BindingShiftedLagCut]);
    }

    #[test]
    fn every_unmet_guard_is_listed_in_one_panic() {
        let evidence = PowerEvidence::new(2, 2);
        evidence.assert_holds("demo gate", &[]);
        let message = failure_message(
            &evidence,
            &[
                Guard::NonUnitZScale,
                Guard::FamilyDual(DualFamily::Evaporation),
                Guard::CouplingDualEveryHydro,
            ],
        )
        .expect("empty evidence meets no guard");
        assert_eq!(
            message,
            "demo gate: the fixture lacks power: unmet guards \
             [NonUnitZScale, FamilyDual(Evaporation), CouplingDualEveryHydro]"
        );
    }

    #[test]
    fn a_zero_coupling_dual_on_one_hydro_fails() {
        let mut evidence = PowerEvidence::new(3, 1);
        evidence.record_coupling_dual(0, 1.0);
        evidence.record_coupling_dual(1, -0.0);
        evidence.record_coupling_dual(2, 3.0);
        let message = failure_message(&evidence, &[Guard::CouplingDualEveryHydro])
            .expect("hydro 1 has no nonzero coupling dual");
        assert!(message.contains("[CouplingDualEveryHydro]"), "{message}");

        evidence.record_coupling_dual(1, 0.5);
        evidence.assert_holds("demo gate", &[Guard::CouplingDualEveryHydro]);
    }

    #[test]
    fn unit_scales_fail_the_scale_guards() {
        let mut evidence = PowerEvidence::new(2, 2);
        evidence.record_expanded_scales(&[1.0, 1.0], &[1.0, 1.0]);
        evidence.record_coupled_row(2.0, 1.0, 1.0);
        evidence.record_coupled_row(0.0, 2.0, 4.0);
        let message = failure_message(
            &evidence,
            &[
                Guard::NonUnitZScale,
                Guard::NonUnitLagScale,
                Guard::RescaledCoupledRowWithDual,
                Guard::NonUnitCompactCoupledRowScaleWithDual,
            ],
        )
        .expect("unit scales and a zero-dual row are not evidence");
        assert!(
            message.ends_with(
                "[NonUnitZScale, NonUnitLagScale, RescaledCoupledRowWithDual, \
                 NonUnitCompactCoupledRowScaleWithDual]"
            ),
            "{message}"
        );
    }

    #[test]
    fn a_unit_compact_scale_leaves_the_compact_scale_guard_unmet() {
        let mut evidence = PowerEvidence::new(2, 2);
        evidence.record_coupled_row(1.0, 1.0, 4.0);
        assert!(evidence.holds(Guard::RescaledCoupledRowWithDual));
        assert!(!evidence.holds(Guard::NonUnitCompactCoupledRowScaleWithDual));

        let mut evidence = PowerEvidence::new(2, 2);
        evidence.record_coupled_row(1.0, 4.0, 4.0);
        assert!(evidence.holds(Guard::NonUnitCompactCoupledRowScaleWithDual));
        assert!(!evidence.holds(Guard::RescaledCoupledRowWithDual));
    }
}

mod fpha_lag_study_checks {
    use novomodelo_core::temporal::BlockMode;
    use novomodelo_sddp::StudySetup;
    use novomodelo_sddp::lp::builder::RowFamily;
    use novomodelo_sddp::setup::{NodePos, StageIdx};
    use novomodelo_sddp::test_support::{capture_patched_node_template_at, oracle_initial_state};
    use novomodelo_solver::{ActiveSolver, SolverInterface};

    use super::shared_dual_identity::dual_family;
    use crate::common::compact_map::CompactMap;
    use crate::common::in_code_studies::fpha_lag_study;
    use crate::common::power_guards::{DualFamily, Guard, PowerEvidence};
    use crate::common::row_families::{RowClass, classify_stage_rows};
    use crate::common::{Emission, build_setup_in_code_with_models_as};

    const CHRONOLOGICAL_STAGE: usize = 1;
    const UNIT_SCALE_BITS: u64 = 1.0_f64.to_bits();

    fn build(emission: Emission) -> StudySetup {
        let (system, config, hydro_models) = fpha_lag_study();
        build_setup_in_code_with_models_as(system, &config, hydro_models, emission)
    }

    #[test]
    fn fpha_lag_study_has_every_required_feature() {
        let (system, _, _) = fpha_lag_study();
        let logical = build(Emission::Logical);
        let compact = build(Emission::Compact);
        let logical_state = logical.stage_state();
        let compact_state = compact.stage_state();

        assert_eq!(
            logical_state.max_par_order, 2,
            "the logical form's lag state must be PAR(2)"
        );
        let logical_ctx = logical.training_ctx();
        let par = logical_ctx.stochastic.par();
        for stage in 0..system.stages().len() {
            for hydro in 0..logical_state.hydro_count {
                for (lag, &psi) in par.psi_slice(stage, hydro).iter().enumerate() {
                    assert!(
                        psi != 0.0 && psi.abs() != 1.0,
                        "stage {stage} hydro {hydro} lag {lag}: psi {psi} must be nonzero with magnitude not 1"
                    );
                }
            }
        }

        let stages = system.stages();
        let modes: Vec<BlockMode> = stages.iter().map(|stage| stage.block_mode).collect();
        assert_eq!(
            modes,
            [
                BlockMode::Parallel,
                BlockMode::Chronological,
                BlockMode::Parallel
            ],
            "only the middle stage is chronological"
        );
        let hours: Vec<f64> = stages[CHRONOLOGICAL_STAGE]
            .blocks
            .iter()
            .map(|block| block.duration_hours)
            .collect();
        assert!(
            hours.len() == 2 && hours[0] != hours[1],
            "the chronological stage needs two blocks of different hours, got {hours:?}"
        );

        let compact_templates = &compact.inputs.stage_data.stage_templates;
        let compact_classes = classify_stage_rows(
            &compact,
            CHRONOLOGICAL_STAGE,
            compact_templates.templates[CHRONOLOGICAL_STAGE].num_rows,
        );
        let coupling = &compact.stage_ctx().inflow_coupling[CHRONOLOGICAL_STAGE];
        assert!(
            coupling.touched_rows().iter().any(|&row| {
                compact_classes[row as usize] == RowClass::Base(RowFamily::GenericConstraint)
            }),
            "stage {CHRONOLOGICAL_STAGE}: no touched inflow-coupling row is a generic-constraint \
             row; touched rows {:?}",
            coupling.touched_rows()
        );

        for (form, state) in [("logical", logical_state), ("compact", compact_state)] {
            assert!(
                state.n_anticipated > 0 && !state.commit_out.is_empty(),
                "{form}: anticipated state absent (n_anticipated {}, commit_out {:?})",
                state.n_anticipated,
                state.commit_out
            );
        }

        let logical_templates = &logical.inputs.stage_data.stage_templates;
        let scales_by_stage = |columns: &std::ops::Range<usize>| -> Vec<Vec<f64>> {
            (0..system.stages().len())
                .map(|stage| logical_templates.templates[stage].col_scale[columns.clone()].to_vec())
                .collect()
        };
        let any_off_unit = |scales: &[Vec<f64>]| {
            scales
                .iter()
                .flatten()
                .any(|scale| scale.to_bits() != UNIT_SCALE_BITS)
        };
        let z_scales = scales_by_stage(&logical_state.z_inflow);
        let lag_scales = scales_by_stage(&logical_state.inflow_lags);
        assert!(
            any_off_unit(&z_scales),
            "no z column scale differs from 1.0 on the logical templates; z scales by stage {z_scales:?}"
        );
        assert!(
            any_off_unit(&lag_scales),
            "no inflow-lag column scale differs from 1.0 on the logical templates; \
             lag scales by stage {lag_scales:?}"
        );

        let coupled_row_scales: Vec<(usize, u32, f64, f64)> = (0..system.stages().len())
            .flat_map(|stage| {
                let logical_template = &logical_templates.templates[stage];
                let compact_template = &compact_templates.templates[stage];
                let map = CompactMap::of_stage(
                    logical_state,
                    compact_state,
                    logical_template,
                    compact_template,
                );
                compact_templates.inflow_coupling_per_stage[stage]
                    .touched_rows()
                    .iter()
                    .map(|&row| {
                        let logical_row = (0..logical_template.num_rows)
                            .find(|&r| map.row(r) == Some(row as usize))
                            .expect("a coupled compact row is a kept logical row");
                        (
                            stage,
                            row,
                            logical_template.row_scale[logical_row],
                            compact_template.row_scale[row as usize],
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(
            coupled_row_scales
                .iter()
                .any(|&(_, _, logical, compact)| logical.to_bits() != compact.to_bits()),
            "no compact coupled row's scale differs from its logical row's; \
             (stage, compact row, logical scale, compact scale): {coupled_row_scales:?}"
        );
    }

    #[test]
    fn fpha_lag_study_duals_bind_on_a_no_cut_stage_solve() {
        let setup = build(Emission::Logical);
        assert_eq!(
            setup.inputs.node_graph.nodes[NodePos(CHRONOLOGICAL_STAGE)].stage,
            StageIdx(CHRONOLOGICAL_STAGE),
            "node position and stage coincide on the chain"
        );
        let zero_noise = vec![0.0; setup.inputs.stochastic.dim()];
        let template = capture_patched_node_template_at(
            &setup,
            NodePos(CHRONOLOGICAL_STAGE),
            &zero_noise,
            &oracle_initial_state(&setup),
        );
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        solver.load_model(&template);
        let view = solver.solve(None).expect("the no-cut stage LP must solve");

        let state = setup.stage_state();
        let mut evidence = PowerEvidence::new(state.hydro_count, state.max_par_order);
        let classes = classify_stage_rows(&setup, CHRONOLOGICAL_STAGE, template.num_rows);
        for (row, &class) in classes.iter().enumerate() {
            if let Some(family) = dual_family(class) {
                evidence.record_family_dual(family, view.dual[row]);
            }
        }
        evidence.assert_holds(
            "no-cut chronological stage",
            &[
                Guard::FamilyDual(DualFamily::Fpha),
                Guard::FamilyDual(DualFamily::Evaporation),
                Guard::FamilyDual(DualFamily::Generic),
            ],
        );
    }
}

mod expander_cut_rows {
    use novomodelo_core::System;
    use novomodelo_sddp::indexer::CutSlot;
    use novomodelo_sddp::test_support::decks::committed_decks;
    use novomodelo_sddp::test_support::expander::{expand_stage_templates, expanded_cut_rows_into};
    use novomodelo_sddp::test_support::template_structure::state_lag_dims;
    use novomodelo_sddp::{StudySetup, build_cut_row_batch_into};
    use novomodelo_solver::RowBatch;

    use crate::common::cut_oracles::apply_oracle_config;
    use crate::common::expanded_solve::empty_batch;
    use crate::common::in_code_studies::fpha_lag_study;
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

        let (mut want, mut got) = (empty_batch(), empty_batch());
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
            expanded_cut_rows_into(
                &mut got,
                &logical.fcf,
                pool,
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
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "trains every committed deck with inflow-lag state"
    )]
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
}

mod mapped_basis_warm_start {
    use novomodelo_core::System;
    use novomodelo_sddp::StudySetup;
    use novomodelo_sddp::setup::StageIdx;
    use novomodelo_sddp::test_support::decks::committed_decks;
    use novomodelo_sddp::test_support::expander::expand_stage_templates;

    use super::expander_cut_rows::LAG_STUDY;
    use crate::common::cut_oracles::apply_oracle_config;
    use crate::common::expanded_solve::{
        assemble_expanded_solve, record_expanded_evidence, solve_from_mapped_basis,
        stored_incoming_states,
    };
    use crate::common::in_code_studies::fpha_lag_study;
    use crate::common::power_guards::{Guard, PowerEvidence};
    use crate::common::{
        Emission, build_setup_in_code_with_models_as, fresh_setup_with, fresh_system_and_setup_as,
        train_storing_trial_states,
    };

    struct WarmStartCounts {
        solves: usize,
        skipped_stages: usize,
    }

    fn is_chain(setup: &StudySetup) -> bool {
        setup
            .inputs
            .node_graph
            .nodes
            .iter_indexed()
            .all(|(pos, node)| node.stage == StageIdx(pos.0) && node.pool_id == pos.0)
    }

    fn warm_start_every_stage(
        label: &str,
        system: &System,
        logical: &mut StudySetup,
        compact: &StudySetup,
        evidence: &mut PowerEvidence,
    ) -> WarmStartCounts {
        train_storing_trial_states(label, logical);
        let expanded = expand_stage_templates(system, compact);
        let mut counts = WarmStartCounts {
            solves: 0,
            skipped_stages: 0,
        };
        for stage in 1..logical.fcf.pools.len() {
            if logical.fcf.active_cuts(stage).next().is_none() {
                continue;
            }
            let Some(states) = stored_incoming_states(logical, stage) else {
                counts.skipped_stages += 1;
                continue;
            };
            let raw_noise = compact
                .training_ctx()
                .stochastic
                .tree_view()
                .opening(stage, 0)
                .to_vec();
            for x_hat in &states {
                let solve = assemble_expanded_solve(
                    compact,
                    &expanded,
                    &logical.fcf,
                    stage,
                    stage,
                    x_hat,
                    &raw_noise,
                );
                let warm = solve_from_mapped_basis(label, &expanded, &solve);
                record_expanded_evidence(evidence, compact, &expanded, stage, &solve, &warm.dual);
                counts.solves += 1;
            }
        }
        counts
    }

    #[test]
    fn the_mapped_compact_basis_is_optimal_on_the_expanded_lp() {
        let build = |emission| {
            let (system, mut config, hydro_models) = fpha_lag_study();
            apply_oracle_config(&mut config);
            build_setup_in_code_with_models_as(system, &config, hydro_models, emission)
        };
        let mut logical = build(Emission::Logical);
        let compact = build(Emission::Compact);
        let (system, _, _) = fpha_lag_study();
        let state = compact.stage_state();
        let mut evidence = PowerEvidence::new(state.hydro_count, state.max_par_order);

        let counts =
            warm_start_every_stage(LAG_STUDY, &system, &mut logical, &compact, &mut evidence);

        assert_eq!(
            counts.skipped_stages, 0,
            "{LAG_STUDY}: a stage's incoming pool drops a state dimension"
        );
        assert!(counts.solves >= 1, "{LAG_STUDY}: no warm solve ran");
        evidence.assert_holds(
            LAG_STUDY,
            &[
                Guard::BindingLagCut,
                Guard::BindingShiftedLagCut,
                Guard::NonUnitZScale,
                Guard::NonUnitLagScale,
                Guard::RescaledCoupledRowWithDual,
            ],
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "trains every committed chain deck with inflow-lag state"
    )]
    fn every_lag_deck_warm_starts_from_the_mapped_basis() {
        let mut selected: Vec<String> = Vec::new();
        let (mut solves, mut skipped_stages, mut non_chain_decks, mut binding_lag_cut_decks) =
            (0, 0, 0, 0);
        for deck in committed_decks() {
            let mut logical = fresh_setup_with(&deck.dir, apply_oracle_config);
            if logical.stage_state().max_par_order == 0 {
                continue;
            }
            if !is_chain(&logical) {
                non_chain_decks += 1;
                continue;
            }
            let (system, compact) = fresh_system_and_setup_as(&deck.dir, Emission::Compact);
            let state = compact.stage_state();
            let mut evidence = PowerEvidence::new(state.hydro_count, state.max_par_order);
            let counts =
                warm_start_every_stage(&deck.key, &system, &mut logical, &compact, &mut evidence);
            solves += counts.solves;
            skipped_stages += counts.skipped_stages;
            binding_lag_cut_decks += usize::from(evidence.holds(Guard::BindingLagCut));
            selected.push(deck.key.clone());
        }
        eprintln!(
            "selected {} decks: {selected:?}; solves={solves}; skipped_stages={skipped_stages}; \
             non_chain_decks={non_chain_decks}; decks_with_binding_lag_cut={binding_lag_cut_decks}",
            selected.len()
        );
        assert!(
            !selected.is_empty(),
            "no committed chain deck has inflow-lag state"
        );
        assert!(solves >= 1, "no warm solve ran on any selected deck");
        assert!(
            binding_lag_cut_decks >= 1,
            "no selected deck's warm solves bind a lag-bearing cut"
        );
    }
}

mod compact_lower_bound {
    use novomodelo_sddp::StudySetup;
    use novomodelo_sddp::indexer::CutSlot;
    use novomodelo_sddp::setup::{NodePos, StageIdx};
    use novomodelo_sddp::test_support::no_cut_root_lower_bound;
    use novomodelo_sddp::test_support::template_structure::state_lag_dims;
    use novomodelo_solver::ActiveSolver;

    use super::expander_cut_rows::{LAG_STUDY, assert_forms_share_cut_geometry};
    use crate::common::cut_oracles::apply_oracle_config;
    use crate::common::in_code_studies::fpha_lag_study;
    use crate::common::{Emission, build_setup_in_code_with_models_as, train_storing_trial_states};

    const ABS_TOL: f64 = 1e-6;
    const REL_TOL: f64 = 1e-7;

    fn trained_forms() -> (StudySetup, StudySetup) {
        let build = |emission| {
            let (system, mut config, hydro_models) = fpha_lag_study();
            apply_oracle_config(&mut config);
            build_setup_in_code_with_models_as(system, &config, hydro_models, emission)
        };
        let mut logical = build(Emission::Logical);
        let mut compact = build(Emission::Compact);
        train_storing_trial_states(LAG_STUDY, &mut logical);
        assert_forms_share_cut_geometry(LAG_STUDY, &logical, &compact);
        compact.fcf = logical.fcf.clone();
        (logical, compact)
    }

    fn root_pool(setup: &StudySetup) -> usize {
        let graph = &setup.inputs.node_graph;
        (0..graph.nodes.len())
            .map(|pos| &graph.nodes[NodePos(pos)])
            .find(|node| node.stage == StageIdx(0))
            .expect("study must have a stage-0 node")
            .pool_id
    }

    fn root_lag_cuts(setup: &StudySetup) -> usize {
        let state = setup.stage_state();
        let pool = root_pool(setup);
        let layout = &setup.training_ctx().cut_state_layouts[pool];
        setup
            .fcf
            .active_cuts(pool)
            .filter(|(_, _, coefficients)| {
                coefficients.iter().enumerate().any(|(j, &c)| {
                    let d = layout.global_state_index(CutSlot::new(j));
                    c != 0.0
                        && state.nonzero_state_indices.binary_search(&d).is_ok()
                        && state_lag_dims(state).contains(&d.get())
                })
            })
            .count()
    }

    fn root_lower_bound(setup: &StudySetup) -> f64 {
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        no_cut_root_lower_bound(setup, &mut solver).expect("the root lower bound must solve")
    }

    fn assert_forms_agree(label: &str, logical: &StudySetup, compact: &StudySetup) -> f64 {
        let (want, got) = (root_lower_bound(logical), root_lower_bound(compact));
        let tolerance = ABS_TOL + REL_TOL * want.abs();
        assert!(
            (got - want).abs() <= tolerance,
            "{label}: compact root lower bound {got:?} vs logical {want:?}, tolerance {tolerance:e}"
        );
        got
    }

    #[test]
    fn compact_root_lower_bound_matches_the_logical_form_with_trained_cuts() {
        let (logical, compact) = trained_forms();
        assert!(
            root_lag_cuts(&logical) > 0,
            "{LAG_STUDY}: the root pool holds no active cut with a nonzero lag coefficient"
        );

        let got = assert_forms_agree(LAG_STUDY, &logical, &compact);

        assert_eq!(
            root_lower_bound(&compact).to_bits(),
            got.to_bits(),
            "{LAG_STUDY}: a second compact evaluation on a fresh solver moved the bound"
        );
    }

    #[test]
    fn compact_root_lower_bound_tracks_a_deactivated_cut_like_the_logical_form() {
        let (mut logical, mut compact) = trained_forms();
        let pool = root_pool(&logical);
        let (slot, _, _) = logical
            .fcf
            .active_cuts(pool)
            .next()
            .expect("the root pool holds an active cut");
        let slot = u32::try_from(slot).expect("a cut slot fits u32");
        logical.fcf.pools[pool].set_active(slot, false);
        compact.fcf.pools[pool].set_active(slot, false);

        assert_forms_agree(LAG_STUDY, &logical, &compact);
    }
}

mod expanded_solve_assembly {
    use std::collections::HashSet;

    use novomodelo_io::config::InflowNonNegativityMethod;
    use novomodelo_sddp::StudySetup;
    use novomodelo_sddp::test_support::bound_recording::BoundWrite;
    use novomodelo_sddp::test_support::expander::{
        ExpandedPatch, ExpandedStudy, expand_stage_templates, expanded_solve_patch,
    };
    use novomodelo_solver::{ActiveSolver, SolverInterface};

    use super::expander_cut_rows::LAG_STUDY;
    use crate::common::cut_oracles::apply_oracle_config;
    use crate::common::expanded_solve::{
        ExpandedSolve, assemble_expanded_solve, load_expanded, record_expanded_evidence,
        stored_incoming_states,
    };
    use crate::common::in_code_studies::fpha_lag_study;
    use crate::common::oracle_tolerance::assert_first_try;
    use crate::common::power_guards::{Guard, PowerEvidence};
    use crate::common::{Emission, build_setup_in_code_with_models_as, train_storing_trial_states};

    const STAGE: usize = 1;

    struct Case {
        logical: StudySetup,
        compact: StudySetup,
        expanded: ExpandedStudy,
        incoming_state: Vec<f64>,
        raw_noise: Vec<f64>,
    }

    impl Case {
        fn new() -> Self {
            let build = |emission| {
                let (system, mut config, hydro_models) = fpha_lag_study();
                apply_oracle_config(&mut config);
                config.modeling.inflow_non_negativity.method =
                    InflowNonNegativityMethod::Truncation;
                build_setup_in_code_with_models_as(system, &config, hydro_models, emission)
            };
            let mut logical = build(Emission::Logical);
            let compact = build(Emission::Compact);
            train_storing_trial_states(LAG_STUDY, &mut logical);
            let (system, _, _) = fpha_lag_study();
            let expanded = expand_stage_templates(&system, &compact);
            let incoming_state = stored_incoming_states(&logical, STAGE)
                .expect("the lag study's pools project every state dimension")
                .into_iter()
                .next()
                .expect("pool 0 stores a trial state");
            let raw_noise = compact
                .training_ctx()
                .stochastic
                .tree_view()
                .opening(STAGE, 0)
                .to_vec();
            Self {
                logical,
                compact,
                expanded,
                incoming_state,
                raw_noise,
            }
        }

        fn assemble(&self) -> ExpandedSolve {
            assemble_expanded_solve(
                &self.compact,
                &self.expanded,
                &self.logical.fcf,
                STAGE,
                STAGE,
                &self.incoming_state,
                &self.raw_noise,
            )
        }
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    fn written_bounds(solve: &ExpandedSolve) -> (HashSet<usize>, HashSet<usize>) {
        let mut cols = HashSet::new();
        let mut rows = HashSet::new();
        for write in &solve.compact.writes {
            match write {
                BoundWrite::ColBounds { indices, .. } => cols.extend(indices.iter().copied()),
                BoundWrite::RowBounds { indices, .. } => rows.extend(indices.iter().copied()),
                BoundWrite::AddRows { .. } | BoundWrite::RowLowerRange { .. } => {}
            }
        }
        (cols, rows)
    }

    fn assert_columns_carry_the_compact_bounds(
        case: &Case,
        solve: &ExpandedSolve,
        patch: &ExpandedPatch,
        written: &HashSet<usize>,
    ) {
        let layout = &case.expanded.layout;
        let base = &case.expanded.scaled[STAGE];
        let compact_final = &solve.compact.template;
        let pinned: HashSet<usize> = patch.incoming_cols.iter().map(|&(col, _)| col).collect();
        let mapped: HashSet<usize> = written.iter().map(|&c| layout.col(c)).collect();
        for &c in written {
            let col = layout.col(c);
            if pinned.contains(&col) {
                continue;
            }
            assert_eq!(
                (
                    solve.template.col_lower[col].to_bits(),
                    solve.template.col_upper[col].to_bits()
                ),
                (
                    compact_final.col_lower[c].to_bits(),
                    compact_final.col_upper[c].to_bits()
                ),
                "stage {STAGE}: compact column {c} (expanded column {col}) lost its written bounds"
            );
        }
        for col in 0..base.num_cols {
            if pinned.contains(&col) || mapped.contains(&col) {
                continue;
            }
            assert_eq!(
                (
                    solve.template.col_lower[col].to_bits(),
                    solve.template.col_upper[col].to_bits()
                ),
                (base.col_lower[col].to_bits(), base.col_upper[col].to_bits()),
                "stage {STAGE}: expanded column {col} is neither written nor pinned and moved"
            );
        }
        for &(col, pin) in &patch.incoming_cols {
            assert_eq!(
                (
                    solve.template.col_lower[col].to_bits(),
                    solve.template.col_upper[col].to_bits()
                ),
                (pin.to_bits(), pin.to_bits()),
                "stage {STAGE}: incoming column {col} does not hold its pin"
            );
        }
    }

    fn assert_a_misplaced_column_index_would_move_an_unwritten_column(
        case: &Case,
        solve: &ExpandedSolve,
        patch: &ExpandedPatch,
    ) {
        let layout = &case.expanded.layout;
        let base = &case.expanded.scaled[STAGE];
        let entries: Vec<(usize, u64, u64)> = solve
            .compact
            .writes
            .iter()
            .filter_map(|write| match write {
                BoundWrite::ColBounds {
                    indices,
                    lower,
                    upper,
                } => Some((indices, lower, upper)),
                _ => None,
            })
            .flat_map(|(indices, lower, upper)| {
                indices
                    .iter()
                    .zip(lower)
                    .zip(upper)
                    .map(|((&c, &lo), &up)| (c, lo.to_bits(), up.to_bits()))
            })
            .collect();
        let accounted: HashSet<usize> = patch
            .incoming_cols
            .iter()
            .map(|&(col, _)| col)
            .chain(entries.iter().map(|&(c, _, _)| layout.col(c)))
            .collect();
        let powered = entries.iter().any(|&(c, lo, up)| {
            layout.col(c) != c
                && !accounted.contains(&c)
                && (lo, up) != (base.col_lower[c].to_bits(), base.col_upper[c].to_bits())
        });
        assert!(
            powered,
            "stage {STAGE}: no written column bound has a relocated index whose own expanded column is \
             unwritten, unpinned and has other bounds, so a misplaced column index would not move an \
             unwritten column"
        );
    }

    fn assert_rows_carry_the_compact_bounds(
        case: &Case,
        solve: &ExpandedSolve,
        patch: &ExpandedPatch,
        written: &HashSet<usize>,
    ) {
        let layout = &case.expanded.layout;
        let base = &case.expanded.scaled[STAGE];
        let compact_final = &solve.compact.template;
        let touched: HashSet<usize> = case.compact.stage_ctx().inflow_coupling[STAGE]
            .touched_rows()
            .iter()
            .map(|&r| usize::try_from(r).expect("row fits usize"))
            .collect();
        let mapped: HashSet<usize> = written
            .iter()
            .filter(|&&r| !touched.contains(&r))
            .map(|&r| layout.row(r))
            .collect();
        for &r in written {
            if touched.contains(&r) {
                continue;
            }
            let row = layout.row(r);
            assert_eq!(
                (
                    solve.template.row_lower[row].to_bits(),
                    solve.template.row_upper[row].to_bits()
                ),
                (
                    compact_final.row_lower[r].to_bits(),
                    compact_final.row_upper[r].to_bits()
                ),
                "stage {STAGE}: compact row {r} (expanded row {row}) lost its written bounds"
            );
        }
        let z_rows: HashSet<usize> = patch.z_rows.iter().map(|&(row, _)| row).collect();
        let coupled: HashSet<usize> = touched.iter().map(|&r| layout.row(r)).collect();
        for row in 0..base.num_rows {
            if z_rows.contains(&row) || mapped.contains(&row) || coupled.contains(&row) {
                continue;
            }
            assert_eq!(
                (
                    solve.template.row_lower[row].to_bits(),
                    solve.template.row_upper[row].to_bits()
                ),
                (base.row_lower[row].to_bits(), base.row_upper[row].to_bits()),
                "stage {STAGE}: expanded row {row} is neither written nor a z row and moved"
            );
        }
        for &r in &touched {
            let row = layout.row(r);
            assert_eq!(
                bits(&[solve.template.row_lower[row], solve.template.row_upper[row]]),
                bits(&[base.row_lower[row], base.row_upper[row]]),
                "stage {STAGE}: coupled row {r} (expanded row {row}) lost its own bounds"
            );
        }
        for &(row, rhs) in &patch.z_rows {
            assert_eq!(
                (
                    solve.template.row_lower[row].to_bits(),
                    solve.template.row_upper[row].to_bits()
                ),
                (rhs.to_bits(), rhs.to_bits()),
                "stage {STAGE}: z row {row} does not hold its right-hand side"
            );
        }
    }

    #[test]
    fn the_expanded_solve_carries_every_compact_per_solve_bound() {
        let case = Case::new();
        let solve = case.assemble();
        let hydro_range = case
            .compact
            .inputs
            .stochastic
            .class_dimensions()
            .hydro_range();
        let patch = expanded_solve_patch(
            &case.compact,
            &case.expanded,
            STAGE,
            &case.raw_noise[hydro_range],
            &case.incoming_state,
        );

        let cut_ranges: Vec<(usize, usize)> = solve
            .compact
            .writes
            .iter()
            .filter_map(|write| match write {
                BoundWrite::RowLowerRange { first_row, lower } => Some((*first_row, lower.len())),
                _ => None,
            })
            .collect();
        assert_eq!(
            cut_ranges,
            [(solve.compact.base_rows, solve.slots.len())],
            "stage {STAGE}: the compact solve must push one cut-row range over the appended rows"
        );

        let (written_cols, written_rows) = written_bounds(&solve);
        assert!(
            !written_cols.is_empty() && !written_rows.is_empty(),
            "stage {STAGE}: the compact solve wrote no column or no row bound"
        );
        assert_columns_carry_the_compact_bounds(&case, &solve, &patch, &written_cols);
        assert_a_misplaced_column_index_would_move_an_unwritten_column(&case, &solve, &patch);
        assert_rows_carry_the_compact_bounds(&case, &solve, &patch, &written_rows);

        assert_eq!(
            solve.cut_rows.num_rows,
            solve.slots.len(),
            "stage {STAGE}: expanded cut-row count"
        );
        assert_eq!(
            solve.slots.len(),
            solve.compact.appended.num_rows,
            "stage {STAGE}: compact cut-row count"
        );
    }

    #[test]
    fn the_assembled_expanded_lp_solves_on_the_first_attempt() {
        let case = Case::new();
        let solve = case.assemble();
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        load_expanded(&mut solver, &solve);

        let before = solver.statistics();
        let outcome = solver.solve(None).map(|view| view.dual.to_vec());
        let after = solver.statistics();
        assert_first_try(LAG_STUDY, &before, &after);
        let dual = outcome.expect("the expanded LP must solve");

        let state = case.compact.stage_state();
        let mut evidence = PowerEvidence::new(state.hydro_count, state.max_par_order);
        record_expanded_evidence(
            &mut evidence,
            &case.compact,
            &case.expanded,
            STAGE,
            &solve,
            &dual,
        );
        evidence.assert_holds(LAG_STUDY, &[Guard::NonUnitZScale, Guard::NonUnitLagScale]);
    }
}

mod oracle_tolerance_checks {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use novomodelo_sddp::setup::{NodePos, StageIdx};
    use novomodelo_sddp::test_support::capture_patched_node_template_with_inflow_noise;
    use novomodelo_solver::{ActiveProfile, ActiveSolver, SolverInterface, SolverStatistics};

    use crate::common::build_setup_in_code;
    use crate::common::in_code_studies::{
        CHRONOLOGICAL_NOISE_INFLOW_STD_M3S, ChronologicalNoiseSpec, chronological_noise_study,
    };
    use crate::common::oracle_tolerance::{UnscaledTolerance, assert_first_try};

    const CALLER: &str = "root stage oracle";

    fn assert_close(quantity: &str, got: f64, want: f64) {
        assert!(
            (got - want).abs() <= 1e-15 * want.abs(),
            "{quantity}: expected {want}, got {got}"
        );
    }

    fn failure_message(before: &SolverStatistics, after: &SolverStatistics) -> Option<String> {
        let payload = catch_unwind(AssertUnwindSafe(|| {
            assert_first_try(CALLER, before, after);
        }))
        .err()?;
        Some(
            *payload
                .downcast::<String>()
                .expect("an assert message is a String"),
        )
    }

    #[test]
    fn conversions_scale_by_the_row_or_column_they_bound() {
        let tolerance = UnscaledTolerance::new(1e-9, 1e-7);
        let (row_scale, col_scale) = (4.0, 0.5);
        assert_close("row residual", tolerance.row_residual(row_scale), 2.5e-10);
        assert_close("column bound", tolerance.col_bound(col_scale), 5e-10);
        assert_close("reduced cost", tolerance.reduced_cost(col_scale), 2e-7);
        assert_close("row dual", tolerance.row_dual(row_scale), 4e-7);
    }

    #[test]
    fn from_active_profile_reads_the_backend_defaults() {
        let tolerance = UnscaledTolerance::from_active_profile();
        let profile = ActiveProfile::default();
        assert_eq!(
            tolerance.primal().to_bits(),
            profile.primal_feasibility_tolerance.to_bits()
        );
        assert_eq!(
            tolerance.dual().to_bits(),
            profile.dual_feasibility_tolerance.to_bits()
        );
    }

    #[test]
    fn a_first_try_solve_passes() {
        let (system, config) = chronological_noise_study(&ChronologicalNoiseSpec::default());
        let setup = build_setup_in_code(system, &config);
        let graph = &setup.inputs.node_graph;
        let root = (0..graph.nodes.len())
            .map(NodePos)
            .find(|&pos| graph.nodes[pos].stage == StageIdx(0))
            .expect("study must have a stage-0 node");
        let zero_draw = [0.0; CHRONOLOGICAL_NOISE_INFLOW_STD_M3S.len()];
        let template = capture_patched_node_template_with_inflow_noise(&setup, root, &zero_draw);
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        solver.load_model(&template);

        let before = solver.statistics();
        solver.solve(None).expect("root stage LP must solve");
        let after = solver.statistics();

        assert_first_try(CALLER, &before, &after);
    }

    #[test]
    fn a_retried_solve_fails() {
        let before = SolverStatistics {
            solve_count: 1,
            first_try_successes: 1,
            ..Default::default()
        };
        let after = SolverStatistics {
            solve_count: 2,
            first_try_successes: 1,
            retry_count: 3,
            ..Default::default()
        };
        let message = failure_message(&before, &after).expect("a retried solve must fail");
        assert!(
            message.contains(CALLER) && message.contains("retried"),
            "unexpected message: {message}"
        );
    }

    #[test]
    fn no_new_solve_fails() {
        let snapshot = SolverStatistics::default();
        let message = failure_message(&snapshot, &snapshot).expect("no new solve must fail");
        assert!(
            message.contains(CALLER) && message.contains("no solve ran"),
            "unexpected message: {message}"
        );
    }
}

mod shared_dual_identity {
    use std::path::Path;

    use novomodelo_core::System;
    use novomodelo_sddp::cut::CutPool;
    use novomodelo_sddp::cut::chain_rule::{
        ChainRuleInputs, ChainRuleScratch, OpeningDuals, per_opening_cut_into,
    };
    use novomodelo_sddp::indexer::{ChainRuleSlots, CutSlot, HydroSys};
    use novomodelo_sddp::lp::builder::RowFamily;
    use novomodelo_sddp::setup::StudyParams;
    use novomodelo_sddp::test_support::expander::{ExpandedStudy, expand_stage_templates};
    use novomodelo_sddp::test_support::template_structure::column_entries;
    use novomodelo_sddp::test_support::{fixture_priced_date, write_synthetic_boundary, ymd};
    use novomodelo_sddp::{
        BoundaryLoadRequest, BoundaryStateRequirements, StudySetup, inject_boundary_cuts,
        load_boundary_cuts,
    };
    use novomodelo_solver::{ActiveSolver, LpSolution, RowBatch, SolverInterface};

    use super::expander_cut_rows::LAG_STUDY;
    use crate::D19_DIR;
    use crate::common::cut_oracles::apply_oracle_config;
    use crate::common::expanded_solve::{
        ExpandedSolve, assemble_expanded_solve, coupled_dual_magnitudes, coupled_duals, lag_dims,
        load_expanded, record_expanded_evidence, stored_incoming_states,
    };
    use crate::common::in_code_studies::fpha_lag_study;
    use crate::common::oracle_tolerance::{UnscaledTolerance, assert_first_try};
    use crate::common::power_guards::{DualFamily, Guard, PowerEvidence};
    use crate::common::row_families::{RowClass, classify_expanded_rows};
    use crate::common::{
        Emission, build_setup_in_code_with_models_as, fresh_setup_with, fresh_system_and_setup_as,
        stochastic_in_code, train_storing_trial_states,
    };

    const WIDENED_LAG_DEPTH: u32 = 4;
    const BOUNDARY_INTERCEPT: f64 = 5.0;
    const DEEP_LAG_SLOPE: f64 = 0.5;

    pub(super) fn dual_family(class: RowClass) -> Option<DualFamily> {
        match class {
            RowClass::Base(RowFamily::Fpha) => Some(DualFamily::Fpha),
            RowClass::Base(RowFamily::Evaporation) => Some(DualFamily::Evaporation),
            RowClass::Base(RowFamily::GenericConstraint) => Some(DualFamily::Generic),
            _ => None,
        }
    }

    fn offset(index: i32) -> usize {
        usize::try_from(index).expect("CSC and CSR offsets and indices are >= 0")
    }

    fn csr_column(cut_rows: &RowBatch, col: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        (0..cut_rows.num_rows).flat_map(move |k| {
            (offset(cut_rows.row_starts[k])..offset(cut_rows.row_starts[k + 1]))
                .filter(move |&p| offset(cut_rows.col_indices[p]) == col)
                .map(move |p| (k, cut_rows.values[p]))
        })
    }

    /// `(ρ_j, Σ|terms|)` of column `col` in unscaled units, from the scaled expanded LP's own
    /// entries and the reported duals `dual`: `(c̃ − Σ Ã·π̃) / col_scale`.
    fn reduced_cost_of(solve: &ExpandedSolve, dual: &[f64], col: usize) -> (f64, f64) {
        let template = &solve.template;
        let mut rho = template.objective[col];
        let mut abs_sum = rho.abs();
        let base = column_entries(template, col).map(|(row, a)| a * dual[row]);
        let cuts = csr_column(&solve.cut_rows, col).map(|(k, a)| a * dual[template.num_rows + k]);
        for product in base.chain(cuts) {
            rho -= product;
            abs_sum += product.abs();
        }
        let col_scale = template.col_scale[col];
        (rho / col_scale, abs_sum / col_scale)
    }

    fn solve_cold(label: &str, solve: &ExpandedSolve) -> LpSolution {
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        load_expanded(&mut solver, solve);
        let before = solver.statistics();
        let outcome = solver.solve(None).map(|view| view.to_owned());
        let after = solver.statistics();
        assert!(
            outcome.is_ok(),
            "{label}: the cold expanded solve failed: {:?}",
            outcome.as_ref().err()
        );
        assert_first_try(label, &before, &after);
        outcome.expect("checked above")
    }

    /// The kernel's cut coefficients, in the extracted projection's slot space, and the absolute
    /// magnitudes of the terms it sums, `g_abs[h]` for `g_h` and `w_abs[l * N + h]` for `w` of
    /// lag `l`.
    struct KernelCut {
        coefficients: Vec<f64>,
        g_abs: Vec<f64>,
        w_abs: Vec<f64>,
    }

    /// The kernel's cut, from the duals of the expanded LP itself.
    fn kernel_cut(
        compact: &StudySetup,
        expanded: &ExpandedStudy,
        resident_pool: &CutPool,
        stage: usize,
        x_hat: &[f64],
        solve: &ExpandedSolve,
        solution: &LpSolution,
    ) -> KernelCut {
        let training_ctx = compact.training_ctx();
        let (resident, extracted) = (
            &training_ctx.cut_state_layouts[stage],
            &training_ctx.cut_state_layouts[stage - 1],
        );
        let col_scale = &solve.template.col_scale;
        let state_duals: Vec<f64> = extracted
            .incoming_pairs()
            .map(|(slot, _)| {
                let col = expanded
                    .layout
                    .incoming_state_col(extracted.global_state_index(slot));
                solution.reduced_costs[col] / col_scale[col]
            })
            .collect();
        let coupled = coupled_duals(compact, expanded, stage, &solution.dual);
        let cut_duals: Vec<(usize, f64)> = solve
            .slots
            .iter()
            .enumerate()
            .filter_map(|(k, &slot)| {
                let y = solution.dual[solve.template.num_rows + k];
                (y != 0.0).then_some((slot, y))
            })
            .collect();

        let space = compact.stage_state();
        let slots = ChainRuleSlots::compose(space, resident, extracted);
        let mut g_abs = coupled_dual_magnitudes(compact, expanded, stage, &solution.dual);
        let mut w_abs = vec![0.0; space.hydro_count * space.max_par_order.saturating_sub(1)];
        let (lag0, shifted) = (slots.resident_lag0(), slots.resident_shifted());
        for &(slot, y) in &cut_duals {
            let row = resident_pool.coefficient_row(slot);
            for (acc, &beta) in g_abs.iter_mut().zip(&row[lag0.clone()]) {
                *acc += (y * beta).abs();
            }
            for (acc, &beta) in w_abs.iter_mut().zip(&row[shifted.clone()]) {
                *acc += (y * beta).abs();
            }
        }
        let mut coefficients = vec![0.0; extracted.n_slots()];
        per_opening_cut_into(
            &ChainRuleInputs {
                slots: &slots,
                resident_pool,
                extracted,
                par: training_ctx.stochastic.par(),
                stage,
            },
            &OpeningDuals {
                objective: solution.objective,
                state_duals: &state_duals,
                coupled_duals: &coupled,
                cut_duals: &cut_duals,
            },
            x_hat,
            &mut ChainRuleScratch::new(space.hydro_count, space.max_par_order),
            &mut coefficients,
        );
        KernelCut {
            coefficients,
            g_abs,
            w_abs,
        }
    }

    /// Checks the identity and the reported tie of every `(hydro, lag)` of one solve and returns
    /// how many pairs it checked and whether some lag at or past the PAR order has a nonzero
    /// `β_fold`.
    fn check_lag_pairs(
        label: &str,
        compact: &StudySetup,
        expanded: &ExpandedStudy,
        stage: usize,
        solve: &ExpandedSolve,
        solution: &LpSolution,
        kernel: &KernelCut,
    ) -> (usize, bool) {
        let layout = &expanded.layout;
        let space = compact.stage_state();
        let (n, depth) = (space.hydro_count, space.max_par_order);
        let extracted = &compact.training_ctx().cut_state_layouts[stage - 1];
        let lag_dims = lag_dims(layout);
        let mut lag_slot = vec![None; n * depth];
        for j in 0..extracted.n_slots() {
            let d = extracted.global_state_index(CutSlot::new(j));
            if let Some((h, l)) = lag_dims[d.get()] {
                lag_slot[l * n + h] = Some(j);
            }
        }

        let par = compact.training_ctx().stochastic.par();
        let coupling = &compact.stage_ctx().inflow_coupling[stage];
        let cut_dual_count = solution.dual[solve.template.num_rows..]
            .iter()
            .filter(|&&y| y != 0.0)
            .count();
        let tolerance = UnscaledTolerance::from_active_profile();
        let unit_roundoff = f64::EPSILON / 2.0;
        // The identity is invariant under any shift of the z-row duals (the kernel eliminates z);
        // the shifted leg makes rho_z nonzero, so the sign of psi is observable.
        let mut shifted_dual = solution.dual.clone();
        for h in 0..n {
            shifted_dual[layout.z_row(HydroSys::new(h))] += 1.0;
        }
        let legs = [
            (solution.dual.as_slice(), ""),
            (shifted_dual.as_slice(), ", z-row duals shifted by 1"),
        ];
        let mut checked = 0;
        let mut deep_nonzero = false;
        for h in 0..n {
            let hydro = HydroSys::new(h);
            let z_costs = legs.map(|(duals, _)| reduced_cost_of(solve, duals, layout.z_col(hydro)));
            // Longest rounding chain over both sides. Recomputation: two template scalings, one
            // product, `m_h + K` column-sum adds, one division by the column scale, one `ψ·ρ_z`
            // and the final add. Kernel: the two scalings and `m_h` adds of (Gᵀπ)_h, `K`
            // product-and-add steps into g, the `ψ·g` product and the final add of w, whose own
            // chain is `K` product-and-add steps.
            let order = coupling.hydro_entries(hydro).count() + 2 * cut_dual_count + 6;
            let order_u = f64::from(u32::try_from(order).expect("n fits u32")) * unit_roundoff;
            let gamma = order_u / (1.0 - order_u);
            for l in 0..depth {
                let col = layout.incoming_lag_col(hydro, l);
                let a_costs = legs.map(|(duals, _)| reduced_cost_of(solve, duals, col));
                let beta = kernel.coefficients[lag_slot[l * n + h].expect("every lag has a slot")];
                let psi = par.psi_slice(stage, h).get(l).copied().unwrap_or(0.0);
                let kernel_abs = psi.abs() * kernel.g_abs[h]
                    + kernel.w_abs.get(l * n + h).copied().unwrap_or(0.0);
                deep_nonzero |= l >= par.max_order() && beta != 0.0;
                for (i, (_, leg)) in legs.iter().enumerate() {
                    let ((rho_a, abs_a), (rho_z, abs_z)) = (a_costs[i], z_costs[i]);
                    let gap = (beta - (rho_a + psi * rho_z)).abs();
                    let bound = 2.0 * gamma * (psi.abs() * abs_z + abs_a + kernel_abs);
                    assert!(
                        gap <= bound,
                        "{label} stage {stage} hydro {h} lag {l}{leg}: beta_fold {beta:e}, \
                         rho_a {rho_a:e}, psi {psi:e}, rho_z {rho_z:e}, gap {gap:e} > bound \
                         {bound:e} (n = {order})"
                    );
                }

                let [(rho_a, _), _] = a_costs;
                let col_scale = solve.template.col_scale[col];
                let reported = solution.reduced_costs[col] / col_scale;
                let tie = tolerance.reduced_cost(col_scale);
                assert!(
                    (rho_a - reported).abs() <= tie,
                    "{label} stage {stage} hydro {h} lag {l}: recomputed rho_a {rho_a:e} vs reported \
                     {reported:e}, tolerance {tie:e}"
                );
                checked += 1;
            }
        }
        (checked, deep_nonzero)
    }

    fn check_shared_dual_identity(
        label: &str,
        system: &System,
        logical: &mut StudySetup,
        compact: &StudySetup,
    ) -> PowerEvidence {
        train_storing_trial_states(label, logical);
        let expanded = expand_stage_templates(system, compact);
        let state = compact.stage_state();
        let (n, depth) = (state.hydro_count, state.max_par_order);
        assert!(n >= 1, "{label}: no hydro, so no coupling dual can bind");
        let mut evidence = PowerEvidence::new(n, depth);
        let mut deep_nonzero = false;

        for stage in 1..logical.fcf.pools.len() {
            if logical.fcf.active_cuts(stage).next().is_none() {
                continue;
            }
            let states = stored_incoming_states(logical, stage)
                .unwrap_or_else(|| panic!("{label}: pool {} drops a state dimension", stage - 1));
            let raw_noise = compact
                .training_ctx()
                .stochastic
                .tree_view()
                .opening(stage, 0)
                .to_vec();
            for x_hat in &states {
                let solve = assemble_expanded_solve(
                    compact,
                    &expanded,
                    &logical.fcf,
                    stage,
                    stage,
                    x_hat,
                    &raw_noise,
                );
                let solution = solve_cold(label, &solve);
                let kernel = kernel_cut(
                    compact,
                    &expanded,
                    &logical.fcf.pools[stage],
                    stage,
                    x_hat,
                    &solve,
                    &solution,
                );
                let (checked, stage_deep_nonzero) =
                    check_lag_pairs(label, compact, &expanded, stage, &solve, &solution, &kernel);
                deep_nonzero |= stage_deep_nonzero;
                assert_eq!(
                    checked,
                    n * depth,
                    "{label} stage {stage}: every (hydro, lag) pair must be checked"
                );

                record_expanded_evidence(
                    &mut evidence,
                    compact,
                    &expanded,
                    stage,
                    &solve,
                    &solution.dual,
                );
                let classes = classify_expanded_rows(
                    compact,
                    &expanded.layout,
                    stage,
                    solve.template.num_rows + solve.cut_rows.num_rows,
                );
                for (row, &class) in classes.iter().enumerate() {
                    if let Some(family) = dual_family(class) {
                        evidence.record_family_dual(
                            family,
                            solve.template.row_scale[row] * solution.dual[row],
                        );
                    }
                }
            }
        }
        let par_order = compact.training_ctx().stochastic.par().max_order();
        assert!(
            depth <= par_order || deep_nonzero,
            "{label}: no lag at or past the PAR order {par_order} carries a nonzero beta_fold"
        );
        evidence
    }

    #[test]
    fn the_shared_dual_identity_holds_on_the_fpha_lag_study() {
        let build = |emission| {
            let (system, mut config, hydro_models) = fpha_lag_study();
            apply_oracle_config(&mut config);
            build_setup_in_code_with_models_as(system, &config, hydro_models, emission)
        };
        let mut logical = build(Emission::Logical);
        let compact = build(Emission::Compact);
        let (system, _, _) = fpha_lag_study();

        let evidence = check_shared_dual_identity(LAG_STUDY, &system, &mut logical, &compact);
        evidence.assert_holds(
            LAG_STUDY,
            &[
                Guard::FamilyDual(DualFamily::Fpha),
                Guard::FamilyDual(DualFamily::Evaporation),
                Guard::FamilyDual(DualFamily::Generic),
                Guard::CouplingDualEveryHydro,
                Guard::BindingLagCut,
                Guard::BindingShiftedLagCut,
            ],
        );
    }

    #[test]
    fn the_shared_dual_identity_holds_on_d19() {
        let (system, compact) = fresh_system_and_setup_as(Path::new(D19_DIR), Emission::Compact);
        let mut logical = fresh_setup_with(Path::new(D19_DIR), apply_oracle_config);
        let depth = compact.stage_state().max_par_order;
        println!("d19 lag depth: {depth}");

        let evidence = check_shared_dual_identity("d19", &system, &mut logical, &compact);
        let mut guards = vec![Guard::CouplingDualEveryHydro, Guard::BindingLagCut];
        if depth >= 2 {
            guards.push(Guard::BindingShiftedLagCut);
        }
        evidence.assert_holds("d19", &guards);
    }

    /// The lag study with its lag state widened to `WIDENED_LAG_DEPTH` by a boundary requirement,
    /// and a one-cut boundary whose slope on every lag at or past the PAR order is
    /// `DEEP_LAG_SLOPE` injected into the logical form's terminal pool.
    fn widened_lag_study(boundary_dir: &Path) -> (System, StudySetup, StudySetup) {
        let build = |emission| {
            let (system, mut config, hydro_models) = fpha_lag_study();
            apply_oracle_config(&mut config);
            let sentinel = Path::new("config.json");
            let training_source = config
                .training_scenario_source(sentinel)
                .expect("training_scenario_source must parse");
            let simulation_source = config
                .simulation_scenario_source(sentinel)
                .expect("simulation_scenario_source must parse");
            let mut params = StudyParams::from_config(&config, Vec::new())
                .expect("StudyParams::from_config must succeed");
            params.boundary = BoundaryStateRequirements::present(WIDENED_LAG_DEPTH);
            let construct = match emission {
                Emission::Logical => StudySetup::from_broadcast_params,
                Emission::Compact => StudySetup::from_broadcast_params_compact,
            };
            let setup = construct(
                &system,
                stochastic_in_code(&system),
                params,
                hydro_models,
                &training_source,
                &simulation_source,
            )
            .expect("the widened-lag study must build");
            (system, setup)
        };
        let (_, mut logical) = build(Emission::Logical);
        let (system, compact) = build(Emission::Compact);

        let par_order = compact.training_ctx().stochastic.par().max_order();
        let coefficients: Vec<f64> = lag_dims(&expand_stage_templates(&system, &compact).layout)
            .iter()
            .map(|lag| match lag {
                Some((_, l)) if *l >= par_order => DEEP_LAG_SLOPE,
                _ => 0.0,
            })
            .collect();
        let state_dimension = u32::try_from(logical.fcf.state_dimension).expect("fits u32");
        assert_eq!(
            coefficients.len(),
            logical.fcf.state_dimension,
            "the boundary's slopes must cover the whole state"
        );
        let date = fixture_priced_date(ymd(2030, 1, 1), 0);
        write_synthetic_boundary(
            boundary_dir,
            state_dimension,
            BOUNDARY_INTERCEPT,
            &coefficients,
            date,
        );
        let boundary = load_boundary_cuts(&BoundaryLoadRequest::new(
            boundary_dir,
            date,
            state_dimension,
            &[],
            1.0,
        ))
        .expect("the boundary cut must load");
        inject_boundary_cuts(&mut logical, &boundary).expect("the boundary cut must inject");
        (system, logical, compact)
    }

    #[test]
    fn the_shared_dual_identity_holds_on_a_lag_state_deeper_than_the_par_order() {
        let boundary_dir = tempfile::tempdir().expect("tempdir");
        let (system, mut logical, compact) = widened_lag_study(boundary_dir.path());
        let par_order = compact.training_ctx().stochastic.par().max_order();
        let depth = compact.stage_state().max_par_order;
        assert!(
            depth > par_order,
            "the lag state ({depth}) must be deeper than the PAR order ({par_order})"
        );

        let evidence = check_shared_dual_identity("widened lag", &system, &mut logical, &compact);
        evidence.assert_holds(
            "widened lag",
            &[
                Guard::CouplingDualEveryHydro,
                Guard::BindingLagCut,
                Guard::BindingShiftedLagCut,
            ],
        );
    }
}
