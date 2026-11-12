use std::path::Path;

use novomodelo_core::System;
use novomodelo_sddp::StudySetup;
use novomodelo_sddp::setup::NodePos;
use novomodelo_sddp::test_support::decks::committed_decks;
use novomodelo_sddp::test_support::expander::expand_stage_templates;

use crate::D43_DIR;
use crate::common::expanded_solve::{
    CutRowSet, assemble_expanded_solve, record_expanded_evidence, solve_from_mapped_basis,
    stages_whose_producer_drops_a_state_dimension,
};
use crate::common::in_code_studies::FPHA_LAG_STUDY_KEY as LAG_STUDY;
use crate::common::in_code_studies::fpha_lag_study;
use crate::common::power_guards::{D43_UNMET_SCALE_GUARDS, Guard, PowerEvidence};
use crate::common::stored_cut_oracles::apply_oracle_config;
use crate::common::{
    build_setup_in_code_with_models, fresh_system_and_setup_with, is_chain,
    train_storing_trial_states, try_stored_trial_states,
};

struct WarmStartCounts {
    skipped_stages: Vec<usize>,
    solves_per_stage: Vec<(usize, usize)>,
}

fn warm_start_every_stage(
    label: &str,
    system: &System,
    compact: &mut StudySetup,
    evidence: &mut PowerEvidence,
) -> WarmStartCounts {
    let expanded = expand_stage_templates(system, compact);
    train_storing_trial_states(label, compact);
    let mut counts = WarmStartCounts {
        skipped_stages: Vec::new(),
        solves_per_stage: Vec::new(),
    };
    let training_ctx = compact.training_ctx();
    let tree = training_ctx.stochastic.tree_view();
    for stage in 1..compact.fcf.pools.len() {
        if compact.fcf.active_cuts(stage).next().is_none() {
            continue;
        }
        let Some(states) = try_stored_trial_states(compact, NodePos(stage - 1)) else {
            counts.skipped_stages.push(stage);
            continue;
        };
        let mut solves = 0;
        for opening in 0..tree.n_openings(stage) {
            let raw_noise = tree.opening(stage, opening);
            for x_hat in &states {
                let solve = assemble_expanded_solve(
                    compact,
                    &expanded,
                    &compact.fcf,
                    &CutRowSet::active(&compact.fcf, stage, stage - 1),
                    stage,
                    x_hat,
                    raw_noise,
                );
                let warm = solve_from_mapped_basis(label, &expanded, &solve);
                record_expanded_evidence(evidence, compact, &expanded, stage, &solve, &warm.dual);
                solves += 1;
            }
        }
        counts.solves_per_stage.push((stage, solves));
    }
    counts
}

/// Asserts the skipped stages are the reduced producers' successors and a stage was checked, then
/// `guards` and each scale guard not in `unmet`, and that every guard of `unmet` is still unmet.
fn assert_run(
    label: &str,
    compact: &StudySetup,
    counts: &WarmStartCounts,
    evidence: &PowerEvidence,
    guards: &[Guard],
    unmet: &[(Guard, &str)],
) {
    assert_eq!(
        counts.skipped_stages,
        stages_whose_producer_drops_a_state_dimension(compact),
        "{label}: the skipped stages must be the stages whose producer pool drops a state dimension"
    );
    assert!(
        !counts.solves_per_stage.is_empty(),
        "{label}: no warm solve ran"
    );
    evidence.assert_scale_guards(label, guards, unmet);
}

#[test]
fn the_mapped_compact_basis_is_optimal_on_the_expanded_lp() {
    let (system, mut config, hydro_models) = fpha_lag_study();
    apply_oracle_config(&mut config);
    let mut compact = build_setup_in_code_with_models(system, &config, hydro_models);
    let (system, _, _) = fpha_lag_study();
    let state = compact.stage_state();
    let mut evidence = PowerEvidence::new(state.hydro_count, state.max_par_order);

    let counts = warm_start_every_stage(LAG_STUDY, &system, &mut compact, &mut evidence);

    assert!(
        counts.skipped_stages.is_empty(),
        "{LAG_STUDY}: a stage's incoming pool drops a state dimension"
    );
    assert_run(
        LAG_STUDY,
        &compact,
        &counts,
        &evidence,
        &[Guard::BindingLagCut, Guard::BindingShiftedLagCut],
        &[],
    );
}

#[test]
fn the_mapped_compact_basis_is_optimal_on_d43_outside_its_lag_dropping_stage() {
    let (system, mut compact) =
        fresh_system_and_setup_with(Path::new(D43_DIR), apply_oracle_config);
    let state = compact.stage_state();
    let mut evidence = PowerEvidence::new(state.hydro_count, state.max_par_order);

    let counts = warm_start_every_stage("d43", &system, &mut compact, &mut evidence);

    assert!(
        !counts.skipped_stages.is_empty(),
        "d43: no stage reads the reduced pool"
    );
    assert!(
        counts.solves_per_stage.iter().any(|&(stage, _)| stage == 1),
        "d43: stage 1, whose resident pool is the reduced pool, was not checked"
    );
    assert_run(
        "d43",
        &compact,
        &counts,
        &evidence,
        &[Guard::CouplingDualEveryHydro],
        D43_UNMET_SCALE_GUARDS,
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "trains every committed deck with inflow lags"
)]
fn every_lag_deck_warm_starts_from_the_mapped_basis() {
    let mut selected: Vec<String> = Vec::new();
    let mut skipped_by_deck: Vec<(String, Vec<usize>)> = Vec::new();
    let (mut solves, mut non_chain_decks, mut binding_lag_cut_decks) = (0, 0, 0);
    for deck in committed_decks() {
        let (system, mut compact) = fresh_system_and_setup_with(&deck.dir, apply_oracle_config);
        if compact.stage_state().max_par_order == 0 {
            continue;
        }
        if !is_chain(&compact) {
            non_chain_decks += 1;
            continue;
        }
        let state = compact.stage_state();
        let mut evidence = PowerEvidence::new(state.hydro_count, state.max_par_order);
        let counts = warm_start_every_stage(&deck.key, &system, &mut compact, &mut evidence);
        assert_eq!(
            counts.skipped_stages,
            stages_whose_producer_drops_a_state_dimension(&compact),
            "{}: the skipped stages must be the stages whose producer pool drops a state dimension",
            deck.key
        );
        solves += counts
            .solves_per_stage
            .iter()
            .map(|&(_, stage_solves)| stage_solves)
            .sum::<usize>();
        if !counts.skipped_stages.is_empty() {
            skipped_by_deck.push((deck.key.clone(), counts.skipped_stages));
        }
        binding_lag_cut_decks += usize::from(evidence.holds(Guard::BindingLagCut));
        selected.push(deck.key.clone());
    }
    eprintln!(
        "selected {} decks: {selected:?}; solves={solves}; skipped_stages={skipped_by_deck:?}; \
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
