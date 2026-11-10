//! Declaration-order permutation invariance: permuting the input entity
//! declaration order leaves every stage-LP template fact group
//! ([`template_fact_groups`]) of every committed deck and in-code study
//! byte-identical, in the compact form too for the lag study and every deck with
//! PAR lags, and leaves the bits of a compact lag-study training run unchanged.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use novomodelo_io::config::InflowNonNegativityMethod as CfgInflowMethod;
use novomodelo_io::config::{StoppingRuleConfig, TrainingSelection};
use novomodelo_sddp::test_support::decks::{SLOW_DECKS, committed_decks};
use novomodelo_sddp::test_support::template_fact_groups;
use novomodelo_sddp::{StudySetup, TrainingResult};
use novomodelo_solver::ActiveSolver;

use common::in_code_studies::{
    ChronologicalNoiseSpec, FphaLagOptions, chronological_noise_study, fpha_lag_study_with,
    mixed_lead_anticipated_study, negative_draw_par1_spec,
};
use common::permute::permute_case;
use common::selection_fingerprint::SelectionFingerprint;
use common::{
    Emission, StubComm, build_setup_in_code, build_setup_in_code_with_models_as, fresh_setup_with,
    fresh_system_and_setup_with_emission,
};

const SEED: u64 = 0x5EED_C0BE_5EED_C0BE;

#[test]
fn every_deck_template_is_invariant_to_declaration_order() {
    let slow_tests_enabled = cfg!(feature = "slow-tests");
    let mut mismatches: Vec<(String, String)> = Vec::new();

    for deck in committed_decks() {
        if !slow_tests_enabled && SLOW_DECKS.contains(&deck.key.as_str()) {
            continue;
        }

        let base_setup = fresh_setup_with(&deck.dir, |_| {});
        let permuted_dir = permute_case(&deck.dir, SEED);
        let permuted_setup = fresh_setup_with(permuted_dir.path(), |_| {});

        for group in mismatched_groups(&base_setup, &permuted_setup, &deck.key) {
            mismatches.push((deck.key.clone(), group));
        }
    }

    assert!(
        mismatches.is_empty(),
        "template facts differ under declaration-order permutation for (deck, group) pairs:\n{}",
        mismatches
            .iter()
            .map(|(deck, group)| format!("  {deck}\t{group}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn mismatched_groups(
    base_setup: &StudySetup,
    reversed_setup: &StudySetup,
    label: &str,
) -> Vec<String> {
    let base = template_fact_groups(base_setup);
    let reversed = template_fact_groups(reversed_setup);
    assert!(
        base.keys().eq(reversed.keys()),
        "{label}: reversed fact-group key set differs from base (base={:?}, reversed={:?})",
        base.keys().collect::<Vec<_>>(),
        reversed.keys().collect::<Vec<_>>(),
    );
    base.iter()
        .filter(|(group, base_bytes)| reversed.get(*group) != Some(*base_bytes))
        .map(|(group, _)| (*group).to_string())
        .collect()
}

#[test]
fn every_in_code_study_template_is_invariant_to_declaration_order() {
    let mut mismatches: Vec<(String, String)> = Vec::new();

    let mixed_lead_label = "in-code/mixed-lead-anticipated";
    let (system, config) = mixed_lead_anticipated_study(false);
    let base_setup = build_setup_in_code(system, &config);
    let (system, config) = mixed_lead_anticipated_study(true);
    let reversed_setup = build_setup_in_code(system, &config);
    for group in mismatched_groups(&base_setup, &reversed_setup, mixed_lead_label) {
        mismatches.push((mixed_lead_label.to_string(), group));
    }

    let chronological_pumping_label = "in-code/chronological-pumping";
    let (system, config) = chronological_noise_study(&ChronologicalNoiseSpec {
        pumping_station: true,
        ..ChronologicalNoiseSpec::default()
    });
    let base_setup = build_setup_in_code(system, &config);
    let (system, config) = chronological_noise_study(&ChronologicalNoiseSpec {
        pumping_station: true,
        reverse_declaration_order: true,
        ..ChronologicalNoiseSpec::default()
    });
    let reversed_setup = build_setup_in_code(system, &config);
    for group in mismatched_groups(&base_setup, &reversed_setup, chronological_pumping_label) {
        mismatches.push((chronological_pumping_label.to_string(), group));
    }

    let negative_draw_label = "in-code/negative-draw-par1-truncation";
    let (system, config) =
        chronological_noise_study(&negative_draw_par1_spec(CfgInflowMethod::Truncation));
    let base_setup = build_setup_in_code(system, &config);
    let (system, config) = chronological_noise_study(&ChronologicalNoiseSpec {
        reverse_declaration_order: true,
        ..negative_draw_par1_spec(CfgInflowMethod::Truncation)
    });
    let reversed_setup = build_setup_in_code(system, &config);
    for group in mismatched_groups(&base_setup, &reversed_setup, negative_draw_label) {
        mismatches.push((negative_draw_label.to_string(), group));
    }

    assert!(
        mismatches.is_empty(),
        "template facts differ under declaration-order reversal for (study, group) pairs:\n{}",
        mismatches
            .iter()
            .map(|(study, group)| format!("  {study}\t{group}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

const LAG_STUDY_OPENINGS: usize = 3;
const LAG_STUDY_FORWARD_PASSES: u32 = 2;
const LAG_STUDY_ITERATIONS: u32 = 3;
const BUDGET_THAT_EVICTS_NOTHING: u32 = 100_000;

fn lag_study_setup(reverse_declaration_order: bool, emission: Emission) -> StudySetup {
    let (system, config, hydro_models) = fpha_lag_study_with(FphaLagOptions {
        reverse_declaration_order,
        ..FphaLagOptions::default()
    });
    build_setup_in_code_with_models_as(system, &config, hydro_models, emission)
}

#[test]
fn the_lag_study_template_is_invariant_to_declaration_order_in_both_forms() {
    let mut mismatches: Vec<(String, String)> = Vec::new();

    for (form, emission) in [
        ("logical", Emission::Logical),
        ("compact", Emission::Compact),
    ] {
        let label = format!("in-code/fpha-lag/{form}");
        let base_setup = lag_study_setup(false, emission);
        let reversed_setup = lag_study_setup(true, emission);
        for group in mismatched_groups(&base_setup, &reversed_setup, &label) {
            mismatches.push((label.clone(), group));
        }

        if emission == Emission::Compact {
            assert!(
                template_fact_groups(&base_setup).contains_key("lp.inflow_coupling"),
                "{label}: no lp.inflow_coupling fact group"
            );
            assert!(
                base_setup
                    .inputs
                    .stage_data
                    .stage_templates
                    .inflow_coupling_per_stage
                    .iter()
                    .any(|coupling| !coupling.touched_rows().is_empty()),
                "{label}: no stage's inflow coupling has an entry, so the comparison does not reach it"
            );
        }
    }

    assert!(
        mismatches.is_empty(),
        "template facts differ under declaration-order reversal for (study, group) pairs:\n{}",
        mismatches
            .iter()
            .map(|(study, group)| format!("  {study}\t{group}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn every_lag_deck_compact_template_is_invariant_to_declaration_order() {
    let slow_tests_enabled = cfg!(feature = "slow-tests");
    let mut compared: Vec<String> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut mismatches: Vec<(String, String)> = Vec::new();

    for deck in committed_decks() {
        if !slow_tests_enabled && SLOW_DECKS.contains(&deck.key.as_str()) {
            continue;
        }

        let base_setup =
            fresh_system_and_setup_with_emission(&deck.dir, |_| {}, Emission::Compact).1;
        if base_setup.stage_state().max_par_order == 0 {
            skipped.push(deck.key);
            continue;
        }
        let permuted_dir = permute_case(&deck.dir, SEED);
        let permuted_setup =
            fresh_system_and_setup_with_emission(permuted_dir.path(), |_| {}, Emission::Compact).1;

        for group in mismatched_groups(&base_setup, &permuted_setup, &deck.key) {
            mismatches.push((deck.key.clone(), group));
        }
        compared.push(deck.key);
    }

    assert!(
        !compared.is_empty(),
        "no committed deck has PAR lags, so no compact template was compared; decks without: {skipped:?}"
    );
    assert!(
        mismatches.is_empty(),
        "compact template facts differ under declaration-order permutation for (deck, group) pairs:\n{}\ndecks compared: {compared:?}",
        mismatches
            .iter()
            .map(|(deck, group)| format!("  {deck}\t{group}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn train_compact_lag_study(reverse_declaration_order: bool) -> (StudySetup, TrainingResult) {
    let (system, mut config, hydro_models) = fpha_lag_study_with(FphaLagOptions {
        openings: LAG_STUDY_OPENINGS,
        reverse_declaration_order,
        ..FphaLagOptions::default()
    });
    config.training.selection = Some(TrainingSelection::Sampled {
        forward_passes: LAG_STUDY_FORWARD_PASSES,
    });
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
        limit: LAG_STUDY_ITERATIONS,
    }]);
    config.training.cut_selection.max_active_per_stage = Some(BUDGET_THAT_EVICTS_NOTHING);
    let mut setup =
        build_setup_in_code_with_models_as(system, &config, hydro_models, Emission::Compact);
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, &StubComm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "lag study training must not error: {:?}",
        outcome.error
    );
    (setup, outcome.result)
}

fn assert_lag_bearing_binding_cut(setup: &StudySetup, result: &TrainingResult) {
    assert!(
        result.iterations >= 2,
        "lag study trained {} iteration(s), so no cut reached a later iteration's templates",
        result.iterations
    );
    let mut lag_pools = Vec::new();
    let mut binding_cuts = 0;
    let mut lag_coefficient_cuts = 0;
    let mut lag_bearing_binding_cuts = 0;
    for (pool_id, layout) in setup.training_ctx().cut_state_layouts.iter().enumerate() {
        if !layout.has_lag_slots() {
            continue;
        }
        lag_pools.push(pool_id);
        let pool = &setup.fcf.pools[pool_id];
        let lags = layout.lag_point().full();
        for slot in 0..pool.populated() {
            let binding = pool.is_active(slot) && pool.metadata(slot).active_count > 0;
            let lag_bearing = pool.coefficient_row(slot)[lags.clone()]
                .iter()
                .any(|&coefficient| coefficient != 0.0);
            binding_cuts += usize::from(binding);
            lag_coefficient_cuts += usize::from(lag_bearing);
            lag_bearing_binding_cuts += usize::from(binding && lag_bearing);
        }
    }
    assert!(
        lag_bearing_binding_cuts > 0,
        "no binding cut carries a nonzero lag coefficient: over the lag-bearing pools \
         {lag_pools:?}, {binding_cuts} cut(s) bound and {lag_coefficient_cuts} carry a nonzero \
         lag coefficient"
    );
}

#[test]
fn the_compact_lag_study_trains_to_the_same_bits_in_either_declaration_order() {
    let (base_setup, base_result) = train_compact_lag_study(false);
    assert_lag_bearing_binding_cut(&base_setup, &base_result);
    let (reversed_setup, reversed_result) = train_compact_lag_study(true);

    SelectionFingerprint::capture(&reversed_setup, &reversed_result).assert_matches(
        &SelectionFingerprint::capture(&base_setup, &base_result),
        "compact lag study, reversed declaration order",
    );
}
