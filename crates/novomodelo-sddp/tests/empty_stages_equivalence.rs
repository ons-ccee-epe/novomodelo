//! An empty `thermal_stages`/`thermal_bounds` pair is equivalent to an absent one for every
//! thermal consumer: discounting, delivery ids, the anticipated ring, and the solved outputs.

#![cfg(feature = "highs")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::fs;
use std::path::Path;

use novomodelo_io::{StateFamily, validate_case};
use novomodelo_sddp::test_support::{state_space, template_fact_groups};
use novomodelo_solver::highs::HighsSolver;
use tempfile::TempDir;

use common::parity_hash::{compute_parity_hash, train_and_simulate_at_dir};
use common::{copy_toy_case, fresh_system_and_setup_with};

const THERMAL_KEYS_ABSENT: &str =
    r#"{ "water_bucket_stages": [{ "start_date": "2024-03-01", "duration_hours": 744.0 }] }"#;
const THERMAL_KEYS_EMPTY: &str = r#"{
  "thermal_stages": [],
  "thermal_bounds": [],
  "water_bucket_stages": [{ "start_date": "2024-03-01", "duration_hours": 744.0 }]
}"#;

fn copy_with_post_study(json: &str) -> TempDir {
    let dir = TempDir::new().expect("tempdir");
    copy_toy_case(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic/d44-travel-time-substage"),
        dir.path(),
    );
    fs::write(dir.path().join("post_study_stages.json"), json)
        .expect("write post_study_stages.json");
    dir
}

#[test]
fn empty_thermal_stages_matches_absent_thermal_keys() {
    let absent = copy_with_post_study(THERMAL_KEYS_ABSENT);
    let empty = copy_with_post_study(THERMAL_KEYS_EMPTY);

    let (_, absent_report) = validate_case(absent.path()).expect("deck A validates");
    assert_eq!(absent_report.error_count, 0, "{:?}", absent_report.errors);
    let (_, empty_report) = validate_case(empty.path()).expect("deck B validates");
    assert_eq!(empty_report.error_count, 0, "{:?}", empty_report.errors);
    let post_study_warnings = empty_report
        .warnings
        .iter()
        .filter(|w| w.file == "post_study_stages.json")
        .count();
    assert_eq!(post_study_warnings, 1, "{:?}", empty_report.warnings);

    let (_, absent_setup) = fresh_system_and_setup_with(absent.path(), |_| {});
    let (_, empty_setup) = fresh_system_and_setup_with(empty.path(), |_| {});
    for setup in [&absent_setup, &empty_setup] {
        let state = state_space(setup);
        assert_eq!(
            state.n_anticipated, 0,
            "no anticipated thermal, so n_delivery is 0"
        );
        assert_eq!(state.k_max, 0);
    }

    let absent_facts = template_fact_groups(&absent_setup);
    let empty_facts = template_fact_groups(&empty_setup);
    assert!(absent_facts.keys().any(|g| g.starts_with("time_value.")));
    assert_eq!(
        absent_facts.keys().collect::<Vec<_>>(),
        empty_facts.keys().collect::<Vec<_>>(),
        "fact group names"
    );
    for (group, bytes) in &absent_facts {
        assert!(bytes == &empty_facts[group], "fact group {group} differs");
    }

    let (absent_trained, absent_results) =
        train_and_simulate_at_dir(absent.path(), HighsSolver::new);
    let (empty_trained, empty_results) = train_and_simulate_at_dir(empty.path(), HighsSolver::new);
    assert_eq!(
        compute_parity_hash(&absent_trained, absent_results),
        compute_parity_hash(&empty_trained, empty_results),
        "parity hash"
    );
}

fn terminal_anticipated_lanes(case_dir: &Path) -> usize {
    let (system, setup) = fresh_system_and_setup_with(case_dir, |_| {});
    setup
        .build_terminal_entity_manifest(&system)
        .iter()
        .filter(|slot| slot.family() == Some(StateFamily::AnticipatedThermalState))
        .count()
}

#[test]
fn empty_thermal_stages_declares_no_anticipated_lane() {
    let anticipated_deck = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/deterministic/d34-anticipated-varying-blocks");
    assert!(terminal_anticipated_lanes(&anticipated_deck) > 0);

    for json in [THERMAL_KEYS_ABSENT, THERMAL_KEYS_EMPTY] {
        let deck = copy_with_post_study(json);
        assert_eq!(terminal_anticipated_lanes(deck.path()), 0, "{json}");
    }
}
