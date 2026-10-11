//! Declaration-order permutation invariance of every committed deck's and every
//! in-code study's stage-LP template facts: permuting the input entity
//! declaration order leaves every stage-LP template fact group
//! ([`template_fact_groups`]) byte-identical.
//!
//! Under the truncation inflow method the per-solve quantities are held to the
//! same standard: the slack bounds the patch applies and the simulated slack,
//! incremental inflow and raw inflow lags, keyed by hydro id, are bit-identical
//! between two fresh runs and between declaration orders.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::collections::BTreeMap;

use novomodelo_io::config::{InflowNonNegativityMethod as CfgInflowMethod, SimulationSelection};
use novomodelo_sddp::StudySetup;
use novomodelo_sddp::indexer::HydroSys;
use novomodelo_sddp::setup::NodePos;
use novomodelo_sddp::test_support::decks::{SLOW_DECKS, committed_decks};
use novomodelo_sddp::test_support::{
    capture_patched_node_template_at, node_opening_noise, oracle_initial_state, state_space,
    template_fact_groups,
};

use common::build_setup_in_code;
use common::fresh_setup_with;
use common::in_code_studies::{
    CHRONOLOGICAL_NOISE_BLOCK_HOURS, ChronologicalNoiseSpec, NEGATIVE_DRAW_LAG_SEED_M3S,
    chronological_noise_study, mixed_lead_anticipated_study, negative_draw_par1_spec,
};
use common::permute::permute_case;
use common::run_simulation;

const SEED: u64 = 0x5EED_C0BE_5EED_C0BE;
const STAGES: usize = 2;
const OPENINGS: usize = 3;
const SCENARIOS: u32 = 4;

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

    let truncation_label = "in-code/truncation-negative-draw-par1";
    let (system, config) =
        chronological_noise_study(&negative_draw_par1_spec(CfgInflowMethod::Truncation));
    let base_setup = build_setup_in_code(system, &config);
    let (system, config) = chronological_noise_study(&ChronologicalNoiseSpec {
        reverse_declaration_order: true,
        ..negative_draw_par1_spec(CfgInflowMethod::Truncation)
    });
    let reversed_setup = build_setup_in_code(system, &config);
    for group in mismatched_groups(&base_setup, &reversed_setup, truncation_label) {
        mismatches.push((truncation_label.to_string(), group));
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

struct TruncationFingerprint {
    bits: BTreeMap<String, u64>,
    largest_slack_upper: f64,
    smallest_stage_one_lag: f64,
}

fn truncation_fingerprint(reversed: bool) -> TruncationFingerprint {
    let spec = ChronologicalNoiseSpec {
        branching_factor: OPENINGS,
        reverse_declaration_order: reversed,
        ..negative_draw_par1_spec(CfgInflowMethod::Truncation)
    };
    let (system, mut config) = chronological_noise_study(&spec);
    config.simulation.enabled = true;
    config.simulation.selection = Some(SimulationSelection::Sampled {
        num_scenarios: SCENARIOS,
    });
    let hydro_ids: Vec<i32> = system.hydros().iter().map(|hydro| hydro.id.0).collect();
    let mut setup = build_setup_in_code(system, &config);

    let initial = oracle_initial_state(&setup);
    let mut pinned = initial.clone();
    let lag_start = state_space(&setup).inflow_lags.start;
    for (h, &hydro_id) in hydro_ids.iter().enumerate() {
        pinned[lag_start + h] = if hydro_id == 1 { -35.0 } else { -15.0 };
    }

    let mut bits = BTreeMap::new();
    let mut largest_slack_upper = f64::NEG_INFINITY;
    for (stage, incoming) in [&initial, &pinned].into_iter().enumerate() {
        let node = NodePos(stage);
        assert_eq!(setup.inputs.node_graph.nodes[node].openings.len, OPENINGS);
        let geometry = &setup.inputs.stage_data.stage_templates.geometry_per_stage[stage];
        for opening in 0..OPENINGS {
            let noise = node_opening_noise(&setup, node, opening);
            let template = capture_patched_node_template_at(&setup, node, &noise, incoming);
            for (h, hydro_id) in hydro_ids.iter().enumerate() {
                let col = geometry.inflow_slack_col(HydroSys::new(h));
                let (lower, upper) = (template.col_lower[col], template.col_upper[col]);
                bits.insert(
                    format!("bound/{stage}/{opening}/{hydro_id}/lower"),
                    lower.to_bits(),
                );
                bits.insert(
                    format!("bound/{stage}/{opening}/{hydro_id}/upper"),
                    upper.to_bits(),
                );
                largest_slack_upper = largest_slack_upper.max(upper);
            }
        }
    }

    let mut smallest_stage_one_lag = f64::INFINITY;
    for scenario in run_simulation(&mut setup, 1) {
        let scenario_id = scenario.scenario_id;
        for stage in &scenario.stages {
            let stage_id = stage.stage_id;
            for hydro in &stage.hydros {
                let at = format!(
                    "{scenario_id}/{stage_id}/{:?}/{}",
                    hydro.block_id, hydro.hydro_id
                );
                bits.insert(
                    format!("slack/{at}"),
                    hydro.inflow_nonnegativity_slack_m3s.to_bits(),
                );
                bits.insert(
                    format!("inflow/{at}"),
                    hydro.incremental_inflow_m3s.to_bits(),
                );
            }
            for lag in &stage.inflow_lags {
                bits.insert(
                    format!(
                        "lag/{scenario_id}/{stage_id}/{}/{}",
                        lag.hydro_id, lag.lag_index
                    ),
                    lag.inflow_m3s.to_bits(),
                );
                if stage_id == 1 {
                    smallest_stage_one_lag = smallest_stage_one_lag.min(lag.inflow_m3s);
                }
            }
        }
    }

    TruncationFingerprint {
        bits,
        largest_slack_upper,
        smallest_stage_one_lag,
    }
}

fn assert_same_bits(base: &BTreeMap<String, u64>, other: &BTreeMap<String, u64>, context: &str) {
    let only_in_base = base.keys().find(|key| !other.contains_key(*key));
    let only_in_other = other.keys().find(|key| !base.contains_key(*key));
    assert!(
        only_in_base.is_none() && only_in_other.is_none(),
        "{context}: key sets differ (first only in base: {only_in_base:?}, first only in other: {only_in_other:?})"
    );
    let differing = || base.iter().filter(|(key, bits)| other[*key] != **bits);
    if let Some((key, base_bits)) = differing().next() {
        let other_bits = other[key];
        panic!(
            "{context}: {} of {} keys differ, first {key}: base {} ({base_bits:#018x}) vs other {} ({other_bits:#018x})",
            differing().count(),
            base.len(),
            f64::from_bits(*base_bits),
            f64::from_bits(other_bits),
        );
    }
}

#[test]
fn truncation_lag_sequence_is_order_invariant() {
    let base = truncation_fingerprint(false);
    let again = truncation_fingerprint(false);
    let permuted = truncation_fingerprint(true);

    let hydros = NEGATIVE_DRAW_LAG_SEED_M3S.len();
    let blocks = CHRONOLOGICAL_NOISE_BLOCK_HOURS.len();
    let scenarios = usize::try_from(SCENARIOS).unwrap();
    let count = |prefix: &str| {
        base.bits
            .keys()
            .filter(|key| key.starts_with(prefix))
            .count()
    };
    assert_eq!(count("bound/"), STAGES * OPENINGS * hydros * 2);
    assert_eq!(count("slack/"), scenarios * STAGES * blocks * hydros);
    assert_eq!(count("inflow/"), scenarios * STAGES * blocks * hydros);
    assert_eq!(count("lag/"), scenarios * STAGES * hydros);
    assert!(
        base.largest_slack_upper > 0.0,
        "no captured slack upper bound is positive: {}",
        base.largest_slack_upper
    );
    assert!(
        base.smallest_stage_one_lag < 0.0,
        "no stage-1 inflow lag is negative: {}",
        base.smallest_stage_one_lag
    );

    assert_same_bits(&base.bits, &again.bits, "run-to-run");
    assert_same_bits(&base.bits, &permuted.bits, "declaration order changed");
}
