use novomodelo_sddp::StudySetup;
use novomodelo_sddp::test_support::decks::{SLOW_DECKS, committed_decks};
use novomodelo_sddp::test_support::expander::expand_stage_templates;

use super::expander_cut_rows::LAG_STUDY;
use crate::common::expanded_solve::{
    PinnedStateRun, stages_whose_producer_drops_a_state_dimension, stored_incoming_states,
};
use crate::common::in_code_studies::{FphaLagOptions, fpha_lag_study_with};
use crate::common::stored_cut_oracles::apply_oracle_config;
use crate::common::{
    Emission, build_setup_in_code_with_models_as, fresh_setup_with,
    fresh_system_and_setup_with_emission, is_chain, train_storing_trial_states,
};

const FIXTURE_LABEL: &str = LAG_STUDY;
const OPENINGS: usize = 3;

fn stages_with_an_active_cut(trained: &StudySetup) -> Vec<usize> {
    (1..trained.fcf.pools.len())
        .filter(|&stage| trained.fcf.active_cuts(stage).next().is_some())
        .collect()
}

/// At every stored incoming state and every opening, the compact LP and the expanded LP agree:
/// each cut row's RHS within the rounding of its two evaluations, and the lifted compact solution
/// on every expanded row.
#[test]
fn compact_and_expanded_agree_at_identical_pinned_states() {
    let options = FphaLagOptions {
        openings: OPENINGS,
        ..FphaLagOptions::default()
    };
    let build = |emission| {
        let (system, mut config, hydro_models) = fpha_lag_study_with(options);
        apply_oracle_config(&mut config);
        build_setup_in_code_with_models_as(system, &config, hydro_models, emission)
    };
    let mut logical = build(Emission::Logical);
    let compact = build(Emission::Compact);
    train_storing_trial_states(FIXTURE_LABEL, &mut logical);
    let (system, _, _) = fpha_lag_study_with(options);
    let expanded = expand_stage_templates(&system, &compact);

    let mut run = PinnedStateRun::new(FIXTURE_LABEL, &logical, &compact, &expanded);
    for stage in stages_with_an_active_cut(&logical) {
        let states = stored_incoming_states(&logical, stage).unwrap_or_else(|| {
            panic!("{FIXTURE_LABEL} stage {stage}: the producer pool drops a state dimension")
        });
        run.check_stage(stage, &states);
    }
    eprintln!("{}", run.summary());
    run.assert_powers();
}

/// The same agreement on the first stored incoming state of every stage of every committed deck
/// with inflow-lag state.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "trains every committed deck with inflow lags"
)]
fn compact_and_expanded_agree_at_identical_pinned_states_on_every_lag_deck() {
    let (mut visited, mut with_lag_terms) = (Vec::new(), 0);
    let mut not_chain = Vec::new();
    for deck in committed_decks() {
        if SLOW_DECKS.contains(&deck.key.as_str()) {
            continue;
        }
        let mut logical = fresh_setup_with(&deck.dir, apply_oracle_config);
        if logical.stage_state().max_par_order == 0 {
            continue;
        }
        if !is_chain(&logical) {
            not_chain.push(deck.key);
            continue;
        }
        let (system, compact) =
            fresh_system_and_setup_with_emission(&deck.dir, apply_oracle_config, Emission::Compact);
        train_storing_trial_states(&deck.key, &mut logical);
        let expanded = expand_stage_templates(&system, &compact);

        let mut run = PinnedStateRun::new(&deck.key, &logical, &compact, &expanded);
        let mut skipped = Vec::new();
        for stage in stages_with_an_active_cut(&logical) {
            match stored_incoming_states(&logical, stage) {
                Some(states) => {
                    run.check_stage(stage, &states[..1]);
                }
                None => skipped.push(stage),
            }
        }
        assert_eq!(
            skipped,
            stages_whose_producer_drops_a_state_dimension(&logical),
            "{}: stages skipped vs stages whose producer drops a state dimension",
            deck.key
        );
        eprintln!("{} (skipped stages {skipped:?})", run.summary());
        with_lag_terms += usize::from(run.rhs.rows_with_lag_terms >= 1);
        visited.push(deck.key);
    }
    eprintln!(
        "visited {} decks: {visited:?}; not chains: {not_chain:?}",
        visited.len()
    );
    assert!(
        visited.len() >= 2,
        "fewer than two committed lag decks were visited: {visited:?}"
    );
    assert!(
        with_lag_terms >= 1,
        "no visited deck has a cut row with a nonzero lag coefficient: {visited:?}"
    );
}
