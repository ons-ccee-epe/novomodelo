//! Classification of `post_study_stages.json` into its legacy, current, and
//! mixed formats, driven through the public `validate_case` pipeline.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use novomodelo_core::WaterBucketStages;
use novomodelo_io::validation::{ErrorKind, Severity, rules::RULES};
use novomodelo_io::{LoadError, validate_case};
use tempfile::TempDir;

mod helpers;
use helpers::{make_minimal_case, write_file};

enum Water {
    FromThermalStages,
    Declared(usize),
}

enum Expected {
    Carrier {
        thermal_stages: usize,
        thermal_bounds: usize,
        water: Water,
    },
    Rejected {
        message: &'static str,
    },
}

struct Case {
    name: &'static str,
    json: &'static str,
    expected: Expected,
}

enum Deck {
    NoArc,
    Arc,
}

enum WaterOutcome {
    Accepted,
    Rejected { fragments: &'static [&'static str] },
}

struct WaterCase {
    name: &'static str,
    deck: Deck,
    post_study: Option<&'static str>,
    outcome: WaterOutcome,
}

struct UnusedCase {
    name: &'static str,
    deck: Deck,
    post_study: &'static str,
    unused_keys: &'static [&'static str],
}

enum LegacyOutcome {
    Warnings(usize),
    MixedRejected,
}

struct LegacyCase {
    name: &'static str,
    deck: Deck,
    post_study: Option<&'static str>,
    outcome: LegacyOutcome,
}

const MIGRATION_SCRIPT: &str = "scripts/migrate/post_study_stages.py";
const THERMAL_PAIR_MESSAGE: &str =
    "`thermal_stages` and `thermal_bounds` must be declared together";

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "legacy",
            json: r#"{
              "stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }],
              "thermal_bounds": [
                { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
              ]
            }"#,
            expected: Expected::Carrier {
                thermal_stages: 1,
                thermal_bounds: 1,
                water: Water::FromThermalStages,
            },
        },
        Case {
            name: "legacy missing thermal_bounds",
            json: r#"{ "stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }] }"#,
            expected: Expected::Rejected {
                message: "missing field `thermal_bounds`",
            },
        },
        Case {
            name: "current full",
            json: r#"{
              "thermal_stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }],
              "thermal_bounds": [
                { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
              ],
              "water_bucket_stages": [
                { "start_date": "2024-02-01", "duration_hours": 696.0 },
                { "start_date": "2024-03-01", "duration_hours": 744.0 }
              ]
            }"#,
            expected: Expected::Carrier {
                thermal_stages: 1,
                thermal_bounds: 1,
                water: Water::Declared(2),
            },
        },
        Case {
            name: "current water only",
            json: r#"{ "water_bucket_stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }] }"#,
            expected: Expected::Carrier {
                thermal_stages: 0,
                thermal_bounds: 0,
                water: Water::Declared(1),
            },
        },
        Case {
            name: "empty object",
            json: "{}",
            expected: Expected::Carrier {
                thermal_stages: 0,
                thermal_bounds: 0,
                water: Water::Declared(0),
            },
        },
        Case {
            name: "mixed stages and thermal_stages",
            json: r#"{
              "stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }],
              "thermal_stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }],
              "thermal_bounds": []
            }"#,
            expected: Expected::Rejected {
                message: MIGRATION_SCRIPT,
            },
        },
        Case {
            name: "mixed stages and water_bucket_stages",
            json: r#"{
              "stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }],
              "thermal_bounds": [],
              "water_bucket_stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }]
            }"#,
            expected: Expected::Rejected {
                message: MIGRATION_SCRIPT,
            },
        },
        Case {
            name: "legacy calendar diagnostics name stages",
            json: r#"{
              "stages": [
                { "start_date": "2024-02-01", "duration_hours": 696.0 },
                { "start_date": "2024-04-01", "duration_hours": 720.0 }
              ],
              "thermal_bounds": []
            }"#,
            expected: Expected::Rejected {
                message: "(stages start_date=2024-04-01)",
            },
        },
        Case {
            name: "current calendar diagnostics name thermal_stages",
            json: r#"{
              "thermal_stages": [
                { "start_date": "2024-02-01", "duration_hours": 696.0 },
                { "start_date": "2024-04-01", "duration_hours": 720.0 }
              ],
              "thermal_bounds": []
            }"#,
            expected: Expected::Rejected {
                message: "(thermal_stages start_date=2024-04-01)",
            },
        },
        Case {
            name: "thermal_bounds without thermal_stages",
            json: r#"{ "thermal_bounds": [] }"#,
            expected: Expected::Rejected {
                message: THERMAL_PAIR_MESSAGE,
            },
        },
        Case {
            name: "thermal_stages without thermal_bounds",
            json: r#"{ "thermal_stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }] }"#,
            expected: Expected::Rejected {
                message: THERMAL_PAIR_MESSAGE,
            },
        },
        Case {
            name: "empty thermal pair",
            json: r#"{ "thermal_stages": [], "thermal_bounds": [] }"#,
            expected: Expected::Carrier {
                thermal_stages: 0,
                thermal_bounds: 0,
                water: Water::Declared(0),
            },
        },
        Case {
            name: "unknown key",
            json: r#"{ "thermal_stage": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }] }"#,
            expected: Expected::Rejected {
                message: "unknown field `thermal_stage`",
            },
        },
    ]
}

#[test]
fn post_study_file_format_table() {
    for case in cases() {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        write_file(dir.path(), "post_study_stages.json", case.json);
        let result = validate_case(dir.path());

        match case.expected {
            Expected::Carrier {
                thermal_stages,
                thermal_bounds,
                water,
            } => {
                let (system, _) = result.unwrap_or_else(|e| panic!("{}: {e}", case.name));
                let ps = system
                    .post_study_stages()
                    .unwrap_or_else(|| panic!("{}: carrier missing", case.name));
                assert_eq!(ps.thermal_stages.len(), thermal_stages, "{}", case.name);
                assert_eq!(ps.thermal_bounds.len(), thermal_bounds, "{}", case.name);
                match (water, &ps.water_bucket_stages) {
                    (Water::FromThermalStages, WaterBucketStages::FromThermalStages) => {}
                    (Water::Declared(n), WaterBucketStages::Declared(stages)) => {
                        assert_eq!(stages.len(), n, "{}", case.name);
                    }
                    (_, actual) => panic!("{}: unexpected water carrier {actual:?}", case.name),
                }
            }
            Expected::Rejected { message } => {
                let Err(LoadError::ConstraintError { description }) = result else {
                    panic!("{}: expected a validation error", case.name);
                };
                let [entry] = description.lines().collect::<Vec<_>>()[..] else {
                    panic!(
                        "{}: expected exactly one error, got:\n{description}",
                        case.name
                    );
                };
                assert!(
                    entry.contains("post_study_stages.json") && entry.contains(message),
                    "{}: unexpected error entry: {entry}",
                    case.name
                );
            }
        }
    }
}

#[test]
fn incomplete_thermal_pair_names_the_missing_key_as_entity() {
    for (json, missing) in [
        (r#"{ "thermal_bounds": [] }"#, "thermal_stages"),
        (r#"{ "thermal_stages": [] }"#, "thermal_bounds"),
    ] {
        let dir = TempDir::new().unwrap();
        make_minimal_case(&dir);
        write_file(dir.path(), "post_study_stages.json", json);
        let Err(LoadError::ConstraintError { description }) = validate_case(dir.path()) else {
            panic!("{json}: expected a validation error");
        };
        assert!(
            description.contains(&format!("post_study_stages.json ({missing}): ")),
            "{json}: {description}"
        );
    }
}

fn write_cascade_deck(dir: &TempDir, deck: &Deck) {
    make_minimal_case(dir);
    let travel_time = match deck {
        Deck::Arc => r#""travel_time_hours": 360.0,"#,
        Deck::NoArc => "",
    };
    let hydro = |id: u32, downstream: &str, travel_time: &str| {
        format!(
            r#"{{
              "id": {id},
              "name": "HYDRO_{id}",
              "operational_start_date": "2024-01-01",
              "downstream_id": {downstream},
              {travel_time}
              "reservoir": {{ "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 }},
              "outflow": {{ "min_outflow_m3s": 0.0, "max_outflow_m3s": null }},
              "generation": {{
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 200.0,
                "min_generation_mw": 0.0,
                "max_generation_mw": 200.0
              }},
              "unit_groups": [{{
                "id": 0,
                "name": "HYDRO_{id}",
                "bus_id": 1,
                "min_generation_mw": 0.0,
                "max_generation_mw": 200.0,
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 200.0
              }}]
            }}"#
        )
    };
    write_file(
        dir.path(),
        "system/hydros.json",
        &format!(
            r#"{{ "hydros": [{}, {}] }}"#,
            hydro(1, "2", travel_time),
            hydro(2, "null", "")
        ),
    );
    let production_model = |hydro_id: u32| {
        format!(
            r#"{{
              "hydro_id": {hydro_id},
              "selection_mode": "stage_ranges",
              "stage_ranges": [{{
                "start_stage_id": 0,
                "end_stage_id": null,
                "model": "constant_productivity",
                "productivity_mw_per_m3s": 0.9
              }}]
            }}"#
        )
    };
    write_file(
        dir.path(),
        "system/hydro_production_models.json",
        &format!(
            r#"{{ "production_models": [{}, {}] }}"#,
            production_model(1),
            production_model(2)
        ),
    );
    if matches!(deck, Deck::Arc) {
        write_file(
            dir.path(),
            "initial_conditions.json",
            r#"{
              "storage": [],
              "filling_storage": [],
              "past_defluences": [
                { "hydro_id": 1, "start_date": "2023-12-01", "end_date": "2024-01-01", "value_m3s": 200.0 }
              ]
            }"#,
        );
    }
}

fn water_cases() -> Vec<WaterCase> {
    vec![
        WaterCase {
            name: "arc deck, empty object",
            deck: Deck::Arc,
            post_study: Some("{}"),
            outcome: WaterOutcome::Rejected {
                fragments: &[
                    "(water_bucket_stages): ",
                    "omits `water_bucket_stages`",
                    "hydro 1",
                ],
            },
        },
        WaterCase {
            name: "arc deck, empty water calendar",
            deck: Deck::Arc,
            post_study: Some(r#"{ "water_bucket_stages": [] }"#),
            outcome: WaterOutcome::Rejected {
                fragments: &["(water_bucket_stages): ", "covers 0 h", "arc of 360 h"],
            },
        },
        WaterCase {
            name: "arc deck, one hour short",
            deck: Deck::Arc,
            post_study: Some(
                r#"{ "water_bucket_stages": [
                  { "start_date": "2024-02-01", "duration_hours": 168.0 },
                  { "start_date": "2024-02-08", "duration_hours": 191.0 }
                ] }"#,
            ),
            outcome: WaterOutcome::Rejected {
                fragments: &["(water_bucket_stages): ", "covers 359 h", "arc of 360 h"],
            },
        },
        WaterCase {
            name: "arc deck, exact equality",
            deck: Deck::Arc,
            post_study: Some(
                r#"{ "water_bucket_stages": [
                  { "start_date": "2024-02-01", "duration_hours": 168.0 },
                  { "start_date": "2024-02-08", "duration_hours": 192.0 }
                ] }"#,
            ),
            outcome: WaterOutcome::Accepted,
        },
        WaterCase {
            name: "arc deck, gap between water stages",
            deck: Deck::Arc,
            post_study: Some(
                r#"{ "water_bucket_stages": [
                  { "start_date": "2024-02-01", "duration_hours": 168.0 },
                  { "start_date": "2024-02-15", "duration_hours": 192.0 }
                ] }"#,
            ),
            outcome: WaterOutcome::Rejected {
                fragments: &["(water_bucket_stages start_date=2024-02-15): "],
            },
        },
        WaterCase {
            name: "no-arc deck, empty object",
            deck: Deck::NoArc,
            post_study: Some("{}"),
            outcome: WaterOutcome::Accepted,
        },
        WaterCase {
            name: "no-arc deck, gap between water stages",
            deck: Deck::NoArc,
            post_study: Some(
                r#"{ "water_bucket_stages": [
                  { "start_date": "2024-02-01", "duration_hours": 168.0 },
                  { "start_date": "2024-02-15", "duration_hours": 192.0 }
                ] }"#,
            ),
            outcome: WaterOutcome::Rejected {
                fragments: &["(water_bucket_stages start_date=2024-02-15): "],
            },
        },
        WaterCase {
            name: "arc deck, legacy file",
            deck: Deck::Arc,
            post_study: Some(
                r#"{
                  "stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }],
                  "thermal_bounds": []
                }"#,
            ),
            outcome: WaterOutcome::Accepted,
        },
        WaterCase {
            name: "arc deck, no file",
            deck: Deck::Arc,
            post_study: None,
            outcome: WaterOutcome::Accepted,
        },
    ]
}

#[test]
fn water_calendar_table() {
    for case in water_cases() {
        let dir = TempDir::new().unwrap();
        write_cascade_deck(&dir, &case.deck);
        if let Some(json) = case.post_study {
            write_file(dir.path(), "post_study_stages.json", json);
        }
        let result = validate_case(dir.path());

        match case.outcome {
            WaterOutcome::Accepted => {
                result.unwrap_or_else(|e| panic!("{}: {e}", case.name));
            }
            WaterOutcome::Rejected { fragments } => {
                let Err(LoadError::ConstraintError { description }) = result else {
                    panic!("{}: expected a validation error", case.name);
                };
                let [entry] = description.lines().collect::<Vec<_>>()[..] else {
                    panic!(
                        "{}: expected exactly one error, got:\n{description}",
                        case.name
                    );
                };
                for fragment in fragments {
                    assert!(
                        entry.contains("post_study_stages.json") && entry.contains(fragment),
                        "{}: `{fragment}` missing from: {entry}",
                        case.name
                    );
                }
            }
        }
    }
}

fn legacy_cases() -> Vec<LegacyCase> {
    vec![
        LegacyCase {
            name: "legacy file",
            deck: Deck::NoArc,
            post_study: Some(
                r#"{
                  "stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }],
                  "thermal_bounds": []
                }"#,
            ),
            outcome: LegacyOutcome::Warnings(1),
        },
        LegacyCase {
            name: "arc deck, no file",
            deck: Deck::Arc,
            post_study: None,
            outcome: LegacyOutcome::Warnings(1),
        },
        LegacyCase {
            name: "no-arc deck, no file",
            deck: Deck::NoArc,
            post_study: None,
            outcome: LegacyOutcome::Warnings(0),
        },
        LegacyCase {
            name: "current file",
            deck: Deck::NoArc,
            post_study: Some(
                r#"{ "thermal_stages": [], "thermal_bounds": [], "water_bucket_stages": [] }"#,
            ),
            outcome: LegacyOutcome::Warnings(0),
        },
        LegacyCase {
            name: "mixed file",
            deck: Deck::NoArc,
            post_study: Some(
                r#"{
                  "stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }],
                  "thermal_stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }],
                  "thermal_bounds": []
                }"#,
            ),
            outcome: LegacyOutcome::MixedRejected,
        },
    ]
}

#[test]
fn legacy_format_warning_table() {
    for case in legacy_cases() {
        let dir = TempDir::new().unwrap();
        write_cascade_deck(&dir, &case.deck);
        if let Some(json) = case.post_study {
            write_file(dir.path(), "post_study_stages.json", json);
        }
        let result = validate_case(dir.path());

        match case.outcome {
            LegacyOutcome::Warnings(expected) => {
                let (_, report) = result.unwrap_or_else(|e| panic!("{}: {e}", case.name));
                let deprecations: Vec<_> = report
                    .warnings
                    .iter()
                    .filter(|entry| {
                        entry.file.ends_with("post_study_stages.json")
                            && entry.message.contains("is deprecated")
                    })
                    .collect();
                assert_eq!(
                    deprecations.len(),
                    expected,
                    "{}: {deprecations:?}",
                    case.name
                );
                for entry in deprecations {
                    for name in [
                        "thermal_stages",
                        "water_bucket_stages",
                        "scripts/migrate/post_study_stages.py",
                    ] {
                        assert!(
                            entry.message.contains(name),
                            "{}: {}",
                            case.name,
                            entry.message
                        );
                    }
                }
            }
            LegacyOutcome::MixedRejected => {
                let Err(LoadError::ConstraintError { description }) = result else {
                    panic!("{}: expected a validation error", case.name);
                };
                assert!(
                    description.contains("cannot be mixed")
                        && !description.contains("is deprecated"),
                    "{}: {description}",
                    case.name
                );
            }
        }
    }
}

fn unused_cases() -> Vec<UnusedCase> {
    vec![
        UnusedCase {
            name: "no-arc deck, empty water calendar",
            deck: Deck::NoArc,
            post_study: r#"{ "water_bucket_stages": [] }"#,
            unused_keys: &["water_bucket_stages"],
        },
        UnusedCase {
            name: "no-arc deck, non-empty water calendar",
            deck: Deck::NoArc,
            post_study: r#"{ "water_bucket_stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }] }"#,
            unused_keys: &["water_bucket_stages"],
        },
        UnusedCase {
            name: "no anticipated thermal, empty thermal pair",
            deck: Deck::NoArc,
            post_study: r#"{ "thermal_stages": [], "thermal_bounds": [] }"#,
            unused_keys: &["thermal_stages"],
        },
        UnusedCase {
            name: "legacy file, deck with neither",
            deck: Deck::NoArc,
            post_study: r#"{
              "stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }],
              "thermal_bounds": []
            }"#,
            unused_keys: &[],
        },
        UnusedCase {
            name: "arc deck, covering water calendar",
            deck: Deck::Arc,
            post_study: r#"{ "water_bucket_stages": [
              { "start_date": "2024-02-01", "duration_hours": 168.0 },
              { "start_date": "2024-02-08", "duration_hours": 192.0 }
            ] }"#,
            unused_keys: &[],
        },
    ]
}

#[test]
fn unused_calendar_warning_table() {
    for case in unused_cases() {
        let dir = TempDir::new().unwrap();
        write_cascade_deck(&dir, &case.deck);
        write_file(dir.path(), "post_study_stages.json", case.post_study);
        let (_, report) =
            validate_case(dir.path()).unwrap_or_else(|e| panic!("{}: {e}", case.name));

        let unused: Vec<_> = report
            .warnings
            .iter()
            .filter(|entry| {
                entry.file.ends_with("post_study_stages.json")
                    && entry.message.contains("has no effect")
            })
            .collect();
        let entities: Vec<_> = unused
            .iter()
            .map(|entry| entry.entity.as_deref().unwrap())
            .collect();
        assert_eq!(entities, case.unused_keys, "{}: {unused:?}", case.name);
        for (entry, key) in unused.iter().zip(case.unused_keys) {
            assert!(
                entry.message.contains(&format!("`{key}`")),
                "{}: {}",
                case.name,
                entry.message
            );
        }
    }
}

#[test]
fn anticipated_thermal_deck_does_not_warn_on_thermal_stages() {
    let dir = TempDir::new().unwrap();
    write_cascade_deck(&dir, &Deck::NoArc);
    write_file(
        dir.path(),
        "system/thermals.json",
        r#"{ "thermals": [{
          "id": 86,
          "name": "T_ANT",
          "operational_start_date": "2024-01-01",
          "bus_id": 1,
          "cost_per_mwh": 10.0,
          "generation": { "min_mw": 0.0, "max_mw": 300.0 },
          "anticipated_config": { "lead_time_hours": 744.0 }
        }] }"#,
    );
    write_file(
        dir.path(),
        "initial_conditions.json",
        r#"{
          "storage": [],
          "filling_storage": [],
          "past_anticipated_commitments": [
            { "thermal_id": 86, "start_date": "2024-01-01", "end_date": "2024-02-01", "value_mw": 0.0 }
          ]
        }"#,
    );
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "thermal_stages": [{ "start_date": "2024-02-01", "duration_hours": 696.0 }],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#,
    );
    let (_, report) = validate_case(dir.path()).unwrap();
    assert!(
        !report
            .warnings
            .iter()
            .any(|entry| entry.message.contains("has no effect")),
        "{:?}",
        report.warnings
    );
}

#[test]
fn legacy_format_rule_is_a_registered_warning() {
    let rule = RULES
        .iter()
        .find(|rule| rule.id == "semantic.5a.56")
        .expect("semantic.5a.56 is listed");
    assert_eq!(rule.severity, Severity::Warning);
    assert_eq!(rule.kind, ErrorKind::ModelQuality);
}

#[test]
fn unused_calendar_rules_are_registered_warnings() {
    for id in ["semantic.5a.61", "semantic.5a.62"] {
        let rule = RULES
            .iter()
            .find(|rule| rule.id == id)
            .unwrap_or_else(|| panic!("{id} is listed"));
        assert_eq!(rule.severity, Severity::Warning, "{id}");
        assert_eq!(rule.kind, ErrorKind::ModelQuality, "{id}");
    }
}

#[test]
fn post_study_format_rules_are_registered_errors() {
    for id in [
        "semantic.5a.57",
        "semantic.5a.58",
        "semantic.5a.59",
        "semantic.5a.60",
    ] {
        let rule = RULES
            .iter()
            .find(|rule| rule.id == id)
            .unwrap_or_else(|| panic!("{id} is listed"));
        assert_eq!(rule.severity, Severity::Error, "{id}");
        assert_eq!(rule.kind, ErrorKind::BusinessRuleViolation, "{id}");
    }
}
