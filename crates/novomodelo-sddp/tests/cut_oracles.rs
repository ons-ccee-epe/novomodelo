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

mod power_guard_checks {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use crate::common::power_guards::{DualFamily, Guard, PowerEvidence};

    const ALL_GUARDS: [Guard; 10] = [
        Guard::BindingLagCut,
        Guard::BindingShiftedLagCut,
        Guard::NonUnitZScale,
        Guard::NonUnitLagScale,
        Guard::RescaledCoupledRowWithDual,
        Guard::NonUnitCoupledRowScaleWithDual,
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
                Guard::NonUnitCoupledRowScaleWithDual,
            ],
        )
        .expect("unit scales and a zero-dual row are not evidence");
        assert!(
            message.ends_with(
                "[NonUnitZScale, NonUnitLagScale, RescaledCoupledRowWithDual, \
                 NonUnitCoupledRowScaleWithDual]"
            ),
            "{message}"
        );
    }
}

mod fpha_lag_study_checks {
    use novomodelo_core::temporal::BlockMode;
    use novomodelo_sddp::StudySetup;
    use novomodelo_sddp::setup::{NodePos, StageIdx};
    use novomodelo_sddp::test_support::{capture_patched_node_template_at, oracle_initial_state};
    use novomodelo_solver::{ActiveSolver, SolverInterface};

    use crate::common::compact_map::CompactMap;
    use crate::common::in_code_studies::fpha_lag_study;
    use crate::common::{Emission, build_setup_in_code_with_models_as};

    const CHRONOLOGICAL_STAGE: usize = 1;
    const UNIT_SCALE_BITS: u64 = 1.0_f64.to_bits();

    fn build(emission: Emission) -> StudySetup {
        let (system, config, hydro_models) = fpha_lag_study();
        build_setup_in_code_with_models_as(system, &config, hydro_models, emission)
    }

    fn generic_row_start(setup: &StudySetup, stage: usize) -> usize {
        let templates = &setup.inputs.stage_data.stage_templates;
        templates.templates[stage].num_rows - templates.generic_constraint_row_entries[stage].len()
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
        let generic_start = generic_row_start(&compact, CHRONOLOGICAL_STAGE);
        let coupling = &compact.stage_ctx().inflow_coupling[CHRONOLOGICAL_STAGE];
        assert!(
            coupling
                .touched_rows()
                .iter()
                .any(|&row| row as usize >= generic_start),
            "stage {CHRONOLOGICAL_STAGE}: no inflow-coupling row at or past the generic-row \
             start {generic_start}; touched rows {:?}",
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
        let templates = &setup.inputs.stage_data.stage_templates;
        let geometry = &templates.geometry_per_stage[CHRONOLOGICAL_STAGE];
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
        let dual = view.dual;

        let fpha_duals: Vec<(usize, f64)> = geometry.fpha.clone().map(|r| (r, dual[r])).collect();
        let evap_duals: Vec<(usize, f64)> = geometry
            .evap_indices
            .iter()
            .map(|e| (e.evap_row, dual[e.evap_row]))
            .collect();
        let generic_start = generic_row_start(&setup, CHRONOLOGICAL_STAGE);
        let generic_duals: Vec<(usize, f64)> = (generic_start..template.num_rows)
            .map(|r| (r, dual[r]))
            .collect();

        assert!(
            fpha_duals.iter().any(|&(_, d)| d != 0.0),
            "stage {CHRONOLOGICAL_STAGE}: every FPHA row dual is zero: {fpha_duals:?}"
        );
        assert!(
            evap_duals.iter().any(|&(_, d)| d != 0.0),
            "stage {CHRONOLOGICAL_STAGE}: every evaporation row dual is zero: {evap_duals:?}"
        );
        assert!(
            generic_duals.iter().any(|&(_, d)| d != 0.0),
            "stage {CHRONOLOGICAL_STAGE}: every generic hydro-inflow row dual is zero: {generic_duals:?}"
        );
    }
}

mod expander_cut_rows {
    use novomodelo_core::System;
    use novomodelo_sddp::indexer::CutSlot;
    use novomodelo_sddp::test_support::decks::committed_decks;
    use novomodelo_sddp::test_support::expander::{expand_stage_templates, expanded_cut_rows_into};
    use novomodelo_sddp::{StudySetup, build_cut_row_batch_into};
    use novomodelo_solver::{ActiveSolver, RowBatch};

    use crate::common::cut_oracles::apply_oracle_config;
    use crate::common::in_code_studies::fpha_lag_study;
    use crate::common::{
        Emission, StubComm, build_setup_in_code_with_models_as, fresh_setup_with,
        fresh_system_and_setup_as,
    };

    const LAG_STUDY: &str = "structural/fpha-lag";

    struct CutRowPower {
        successor_pools_without_lag_cut: Vec<usize>,
        lag_cuts: usize,
        shifted_lag_cuts: usize,
    }

    fn empty_batch() -> RowBatch {
        RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        }
    }

    fn train(key: &str, setup: &mut StudySetup) {
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = setup
            .train(&mut solver, &StubComm, 1, ActiveSolver::new, None, None)
            .expect("training must return Ok");
        assert!(
            outcome.error.is_none(),
            "{key}: training must not error: {:?}",
            outcome.error
        );
    }

    fn assert_forms_share_cut_geometry(key: &str, logical: &StudySetup, compact: &StudySetup) {
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
                        (c != 0.0 && rendered && state.inflow_lags.contains(&d.get()))
                            .then(|| (d.get() - state.inflow_lags.start) / state.hydro_count)
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
        train(LAG_STUDY, &mut logical);
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
            train(&deck.key, &mut logical);
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
