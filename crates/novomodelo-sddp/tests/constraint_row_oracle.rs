//! The constraint-row facts of the stage LPs of a `d02-single-hydro` copy with generic
//! constraints: rows by identity, the probes a solve evaluates, the JSON round trip and the
//! digest of the rest of the LP. The facts and the simulation outputs of the oracle cases are
//! compared with a golden in their own identity, captured once and never regenerated.
//! An ignored test does the same for a converted case that environment variables name, against
//! a snapshot kept outside the repository.

#![allow(clippy::expect_used, clippy::panic)]

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::path::{Path, PathBuf};

use novomodelo_core::System;
use novomodelo_sddp::StudySetup;
use novomodelo_sddp::test_support::constraint_row_facts::{
    BoundFacts, ConstraintRowFacts, RowFacts, StageRowFacts, constraint_row_facts,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tempfile::TempDir;

use common::constraint_oracle::{
    Cells, Divergence, OracleMap, Segment, compare_output_facts, compare_template_facts,
};
use common::constraint_oracle_cases::{
    OracleCase, OutputFacts, cells, default_slack, naming_case, oracle_cases, output_facts,
    write_generic,
};
use common::parquet_fixtures::write_seasonal_stats;
use common::{copy_case_with_stage_field, fresh_setup_with, fresh_system_and_setup_with_boundary};

fn d02() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/deterministic/d02-single-hydro")
}

/// 701 (`hydro_generation(0) >= lower_701`) and 703 (`hydro_generation(0) <= 1000000`), the
/// latter gated by `bus_load_value(0) <= 0`.
fn gated_pair(lower_701: f64) -> Value {
    let slack = default_slack(&d02());
    json!({
        "constraints": [
            { "id": 701, "name": "constraint_701",
              "expression": format!("hydro_generation(0) >= {lower_701}"), "slack": slack },
            { "id": 703, "name": "constraint_703", "expression": "hydro_generation(0) <= 1000000",
              "active_when": "bus_load_value(0) <= 0", "slack": slack },
        ],
    })
}

/// `d02-single-hydro` with `generic` as its generic constraints, each held at every block of every
/// study stage. d02's load is deterministic, so the copy draws it at 20% of its mean to make bus 0
/// a load-noise member that a probe sets.
fn case(generic: &Value) -> TempDir {
    let case = copy_case_with_stage_field(&d02(), "num_openings", &json!(1));

    let load_stats = case.path().join("scenarios/load_seasonal_stats.parquet");
    let rows: Vec<_> = novomodelo_io::parse_load_seasonal_stats(&load_stats)
        .expect("d02 load stats must parse")
        .into_iter()
        .map(|row| (row.bus_id.0, row.stage_id, row.mean_mw, 0.2 * row.mean_mw))
        .collect();
    write_seasonal_stats(&load_stats, "bus_id", "mean_mw", "std_mw", &rows);

    let bounds: Vec<_> = generic["constraints"]
        .as_array()
        .expect("a constraints array")
        .iter()
        .flat_map(|constraint| {
            let id = constraint["id"].as_i64().expect("an integer constraint id");
            let id = i32::try_from(id).expect("an i32 constraint id");
            cells(case.path(), id, .., None, None)
        })
        .collect();
    write_generic(&case, generic, &bounds);
    case
}

fn setup(case: &TempDir) -> StudySetup {
    fresh_setup_with(case.path(), |_| {})
}

/// One probe per entry of `loads`, setting every load-noise member at every block of every stage
/// to that load.
fn probes(setup: &StudySetup, loads: &[f64]) -> Vec<Vec<Vec<f64>>> {
    let templates = &setup.inputs.stage_data.stage_templates;
    loads
        .iter()
        .map(|&load| {
            templates
                .geometry_per_stage
                .iter()
                .map(|geometry| vec![load; templates.n_load_buses() * geometry.n_blks])
                .collect()
        })
        .collect()
}

fn row(stage: &StageRowFacts, constraint_id: i32) -> &RowFacts {
    stage
        .rows
        .iter()
        .find(|row| row.constraint_id == constraint_id)
        .unwrap_or_else(|| panic!("stage {}: no row {constraint_id}", stage.stage_id))
}

#[test]
fn facts_list_every_row_of_the_family_by_identity() {
    let setup = setup(&case(&gated_pair(1000.0)));
    let facts = constraint_row_facts(&setup, &[]);

    assert_eq!(facts.stages.len(), 2, "d02 has two study stages");
    for stage in &facts.stages {
        let keys: Vec<_> = stage
            .rows
            .iter()
            .map(|row| (row.constraint_id, row.block_id))
            .collect();
        assert_eq!(keys, [(701, 0), (703, 0)], "stage {}", stage.stage_id);
        assert!(row(stage, 703).gated, "stage {}", stage.stage_id);
        assert!(!row(stage, 701).gated, "stage {}", stage.stage_id);
        assert_eq!(
            row(stage, 701).lower,
            BoundFacts::Fixed("1000.0".to_string()),
            "stage {}",
            stage.stage_id
        );
    }
}

#[test]
fn a_probe_reports_the_gate_and_the_bounds_it_evaluates() {
    let setup = setup(&case(&gated_pair(1000.0)));
    let probes = probes(&setup, &[0.0, 1e9]);
    let facts = constraint_row_facts(&setup, &probes);

    assert_eq!(facts, constraint_row_facts(&setup, &probes));
    for stage in &facts.stages {
        let [zero, huge] = stage.probes.as_slice() else {
            panic!("stage {}: two probes", stage.stage_id);
        };
        assert_eq!(zero.loads, ["0.0"], "stage {}", stage.stage_id);
        assert_eq!(huge.loads, ["1000000000.0"], "stage {}", stage.stage_id);
        let [at_zero] = zero.rows.as_slice() else {
            panic!("stage {}: only 703 is gated", stage.stage_id);
        };
        let [at_huge] = huge.rows.as_slice() else {
            panic!("stage {}: only 703 is gated", stage.stage_id);
        };
        assert_eq!((at_zero.constraint_id, at_zero.block_id), (703, 0));
        assert!(at_zero.active, "stage {}", stage.stage_id);
        assert_eq!(at_zero.upper, "1000000.0", "stage {}", stage.stage_id);
        assert_eq!((at_huge.constraint_id, at_huge.block_id), (703, 0));
        assert!(!at_huge.active, "stage {}", stage.stage_id);
        assert_eq!(at_huge.lower, "-inf", "stage {}", stage.stage_id);
        assert_eq!(at_huge.upper, "inf", "stage {}", stage.stage_id);
    }
}

#[test]
fn facts_round_trip_through_json_bit_for_bit() {
    let setup = setup(&case(&gated_pair(1000.0)));
    let specials = [-0.0_f64, f64::INFINITY, f64::NEG_INFINITY];
    let facts = constraint_row_facts(&setup, &probes(&setup, &specials));

    let text = serde_json::to_string_pretty(&facts).expect("facts must serialize");
    let read: ConstraintRowFacts = serde_json::from_str(&text).expect("facts must deserialize");

    assert_eq!(read, facts);
    for stage in &read.stages {
        let bits: Vec<u64> = stage
            .probes
            .iter()
            .flat_map(|probe| &probe.loads)
            .map(|load| load.parse::<f64>().expect("a load parses").to_bits())
            .collect();
        assert_eq!(bits, specials.map(f64::to_bits), "stage {}", stage.stage_id);
    }
}

#[test]
fn a_family_change_leaves_the_outside_digest_unchanged() {
    let at_1000 = constraint_row_facts(&setup(&case(&gated_pair(1000.0))), &[]);
    let at_1001 = constraint_row_facts(&setup(&case(&gated_pair(1001.0))), &[]);
    let wider_thermal = case(&gated_pair(1000.0));
    let thermals = wider_thermal.path().join("system/thermals.json");
    let mut document: Value =
        serde_json::from_str(&std::fs::read_to_string(&thermals).expect("read thermals"))
            .expect("parse thermals");
    document["thermals"][0]["generation"]["max_mw"] = json!(200.0);
    std::fs::write(&thermals, document.to_string()).expect("write thermals");
    let wider_thermal = constraint_row_facts(&setup(&wider_thermal), &[]);

    assert_eq!(at_1000.stages.len(), at_1001.stages.len());
    for ((a, b), twin) in at_1000
        .stages
        .iter()
        .zip(&at_1001.stages)
        .zip(&wider_thermal.stages)
    {
        assert_eq!(
            a.outside_family_sha256, b.outside_family_sha256,
            "stage {}",
            a.stage_id
        );
        assert_ne!(a.rows, b.rows, "stage {}", a.stage_id);
        assert_ne!(
            a.outside_family_sha256, twin.outside_family_sha256,
            "stage {}: the digest reads the thermal bounds outside the family",
            a.stage_id
        );
    }
}

/// derived: 705 reads `-1 * bus_load_value(0)`, so a solve writes `10 - (-1 * load)` on its upper
/// side and leaves the absent lower side open.
#[test]
fn a_probe_reports_the_bound_of_a_row_that_reads_a_bus_load_value() {
    let slack = default_slack(&d02());
    let setup = setup(&case(&json!({
        "constraints": [
            { "id": 701, "name": "constraint_701", "expression": "hydro_generation(0) >= 1000",
              "slack": slack },
            { "id": 705, "name": "constraint_705",
              "expression": "hydro_generation(0) - bus_load_value(0) <= 10", "slack": slack },
        ],
    })));
    let loads = [0.0, 50.0];
    let load_coefficient = -1.0;
    let facts = constraint_row_facts(&setup, &probes(&setup, &loads));

    for stage in &facts.stages {
        assert_eq!(
            (&row(stage, 705).lower, &row(stage, 705).upper),
            (&BoundFacts::Absent, &BoundFacts::PerSolve),
            "stage {}",
            stage.stage_id
        );
        assert_eq!(stage.probes.len(), loads.len(), "stage {}", stage.stage_id);
        for (probe, load) in stage.probes.iter().zip(loads) {
            let [shifted] = probe.rows.as_slice() else {
                panic!(
                    "stage {}: only 705 reads a load, got {:?}",
                    stage.stage_id, probe.rows
                );
            };
            assert_eq!((shifted.constraint_id, shifted.block_id), (705, 0));
            assert!(shifted.active, "stage {}", stage.stage_id);
            assert_eq!(shifted.lower, "-inf", "stage {}", stage.stage_id);
            assert_eq!(
                shifted.upper,
                format!("{:?}", 10.0 - load_coefficient * load),
                "stage {}: load {load}",
                stage.stage_id
            );
        }
    }
}

#[test]
#[should_panic(expected = "stage 0: probe 0 has no load vector for it")]
fn a_probe_without_a_stage_vector_panics_naming_the_stage() {
    let setup = setup(&case(&gated_pair(1000.0)));
    let _ = constraint_row_facts(&setup, &[Vec::new()]);
}

const TEMPLATE_FACTS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/constraint_row_golden/template_facts.json"
);
const OUTPUTS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/constraint_row_golden/outputs.json"
);
const MAX_REPORTED: usize = 50;

fn case_facts(case: &OracleCase) -> ConstraintRowFacts {
    naming_case(case.name, || {
        constraint_row_facts(&setup(&case.dir), &case.probes)
    })
}

fn read_frozen<T: DeserializeOwned>(path: &Path) -> T {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read the snapshot {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("the snapshot {} must parse: {error}", path.display()))
}

/// The golden's entries by case name, each left as JSON until its case reads it.
fn golden_entries(golden: &str) -> BTreeMap<String, Value> {
    read_frozen(Path::new(golden))
}

fn golden_entry<T: DeserializeOwned>(
    entries: &BTreeMap<String, Value>,
    golden: &str,
    name: &str,
) -> T {
    let entry = entries
        .get(name)
        .unwrap_or_else(|| panic!("the golden has no case {name}"));
    T::deserialize(entry)
        .unwrap_or_else(|error| panic!("the golden {golden}: case {name} must parse: {error}"))
}

/// The golden entry of every catalog case.
fn golden_by_name<T: DeserializeOwned>(golden: &str) -> BTreeMap<String, T> {
    let entries = golden_entries(golden);
    oracle_cases()
        .iter()
        .map(|case| {
            (
                case.name.to_string(),
                golden_entry(&entries, golden, case.name),
            )
        })
        .collect()
}

/// The count of `lines` and the first [`MAX_REPORTED`] of them.
fn report(lines: &[impl Display]) -> String {
    let shown: Vec<String> = lines
        .iter()
        .take(MAX_REPORTED)
        .map(ToString::to_string)
        .collect();
    format!(
        "{} divergence(s), first {} shown:\n  {}",
        lines.len(),
        shown.len(),
        shown.join("\n  ")
    )
}

/// `capture` of every case of `cases` must have no divergence from its entry in the golden `file`.
fn check_golden<T: DeserializeOwned>(
    file: &str,
    cases: &[OracleCase],
    capture: impl Fn(&OracleCase) -> T,
    compare: fn(&T, &T) -> Vec<Divergence>,
) {
    let captured: Vec<(&'static str, T)> = cases
        .iter()
        .map(|case| (case.name, capture(case)))
        .collect();

    let entries = golden_entries(file);
    let divergences: Vec<String> = captured
        .iter()
        .flat_map(|(name, facts)| {
            let golden: T = golden_entry(&entries, file, name);
            compare(&golden, facts)
                .into_iter()
                .map(move |divergence| format!("case {name}: {divergence}"))
        })
        .collect();
    assert!(
        divergences.is_empty(),
        "the facts diverge from the golden {file}: {}",
        report(&divergences)
    );
}

fn compare_unmapped(golden: &ConstraintRowFacts, current: &ConstraintRowFacts) -> Vec<Divergence> {
    compare_template_facts(golden, current, &OracleMap::default())
}

#[test]
fn golden_template_facts() {
    check_golden(
        TEMPLATE_FACTS,
        &oracle_cases(),
        case_facts,
        compare_unmapped,
    );
}

#[test]
fn golden_outputs() {
    check_golden(OUTPUTS, &oracle_cases(), output_facts, compare_output_facts);
}

#[test]
fn every_golden_entry_has_a_case() {
    let catalog: BTreeSet<&str> = oracle_cases().iter().map(|case| case.name).collect();
    for golden in [TEMPLATE_FACTS, OUTPUTS] {
        let entries = golden_entries(golden);
        let without_a_case: BTreeSet<&str> = entries
            .keys()
            .map(String::as_str)
            .filter(|name| !catalog.contains(name))
            .collect();
        assert_eq!(without_a_case, BTreeSet::new(), "the golden {golden}");
    }
}

fn golden_case<T: DeserializeOwned>(golden: &str, name: &str) -> T {
    golden_entry(&golden_entries(golden), golden, name)
}

#[test]
fn identity_has_no_divergence() {
    let facts: ConstraintRowFacts = golden_case(TEMPLATE_FACTS, "inequalities_and_horizon");
    let outputs: OutputFacts = golden_case(OUTPUTS, "inequalities_and_horizon");
    let map = OracleMap::default();

    let template = compare_template_facts(&facts, &facts, &map);
    let outputs = compare_output_facts(&outputs, &outputs);

    assert!(template.is_empty(), "{}", report(&template));
    assert!(outputs.is_empty(), "{}", report(&outputs));
}

#[test]
fn a_one_ulp_change_is_located() {
    let golden: ConstraintRowFacts = golden_case(TEMPLATE_FACTS, "inequalities_and_horizon");
    let mut current = golden.clone();
    let entry = current
        .stages
        .iter_mut()
        .flat_map(|stage| &mut stage.rows)
        .find(|row| row.constraint_id == 501)
        .and_then(|row| row.entries.first_mut())
        .expect("501 has an entry");
    let value: f64 = entry.value.parse().expect("a coefficient parses");
    entry.value = format!("{:?}", value.next_up());

    let divergences = compare_template_facts(&golden, &current, &OracleMap::default());

    let [divergence] = divergences.as_slice() else {
        panic!("{}", report(&divergences));
    };
    assert_eq!(divergence.constraint_id, Some(501), "{divergence}");
    assert!(divergence.fact.starts_with("entries"), "{divergence}");
    assert_eq!(divergence.ulp_distance, Some(1), "{divergence}");
}

#[test]
fn a_division_map_matches_split_rows() {
    let golden: ConstraintRowFacts = golden_case(TEMPLATE_FACTS, "range_overrides");
    let mut current = golden.clone();
    stage_mut(&mut current, 1)
        .rows
        .iter_mut()
        .find(|row| (row.constraint_id, row.block_id) == (6022, 0))
        .expect("6022 has a row at block 0 of stage 1")
        .constraint_id = 6023;
    let other_cells: Vec<(i32, usize)> = golden
        .stages
        .iter()
        .flat_map(|stage| {
            let rows = stage.rows.iter().filter(|row| row.constraint_id == 6022);
            rows.map(|row| (stage.stage_id, row.block_id))
        })
        .filter(|&cell| cell != (1, 0))
        .collect();
    let map = OracleMap {
        constraints: BTreeMap::from([(
            6022,
            vec![
                Segment {
                    id: 6022,
                    cells: Cells::Listed(other_cells),
                },
                Segment {
                    id: 6023,
                    cells: Cells::Listed(vec![(1, 0)]),
                },
            ],
        )]),
    };

    let split = compare_template_facts(&golden, &current, &map);
    let unmapped = compare_template_facts(&golden, &current, &OracleMap::default());

    assert!(split.is_empty(), "{}", report(&split));
    assert!(
        unmapped
            .iter()
            .any(|d| d.constraint_id == Some(6023) && d.fact == "extra row"),
        "{}",
        report(&unmapped)
    );
}

/// Another value that deserializes into the same field: the neighbouring float, the other per-solve
/// marker of a bound, or a longer text.
fn next_value(text: &str) -> String {
    match text {
        "Absent" => "PerSolve".to_string(),
        "PerSolve" => "Absent".to_string(),
        _ => match text.parse::<f64>() {
            Ok(value) if value < f64::INFINITY => format!("{:?}", value.next_up()),
            Ok(value) => format!("{:?}", value.next_down()),
            Err(_) => format!("{text}x"),
        },
    }
}

/// A float moved beyond the solver tolerance, or a longer text.
fn far_value(text: &str) -> String {
    match text.parse::<f64>() {
        Ok(value) if value.is_finite() => format!("{:?}", -1e12 * (1.0 + value.abs())),
        Ok(value) => format!("{:?}", -value),
        Err(_) => format!("{text}x"),
    }
}

/// Changes the `skip`-th non-null leaf of `value` outside the `ignored` fields, depth first, with
/// `change` on a text; `false` when there is none.
fn change_leaf(
    value: &mut Value,
    skip: &mut usize,
    ignored: &[&str],
    change: fn(&str) -> String,
) -> bool {
    match value {
        Value::Array(items) => items
            .iter_mut()
            .any(|item| change_leaf(item, skip, ignored, change)),
        Value::Object(fields) => fields
            .iter_mut()
            .filter(|(name, _)| !ignored.contains(&name.as_str()))
            .any(|(_, field)| change_leaf(field, skip, ignored, change)),
        Value::Null => false,
        _ if *skip > 0 => {
            *skip -= 1;
            false
        }
        Value::Bool(flag) => {
            *flag = !*flag;
            true
        }
        Value::Number(number) => {
            let number = number.as_i64().expect("an integer fact");
            *value = json!(number + 1);
            true
        }
        Value::String(text) => {
            *text = change(text);
            true
        }
    }
}

/// Asserts that changing any one leaf of a case outside the `ignored` fields gives a divergence
/// under the empty map.
fn assert_every_leaf_is_compared<T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(
    cases: &BTreeMap<String, T>,
    ignored: &[&str],
    change: fn(&str) -> String,
    compare: fn(&T, &T) -> Vec<Divergence>,
) {
    for (name, facts) in cases {
        let value = serde_json::to_value(facts).expect("the facts serialize");
        for leaf in 0.. {
            let mut changed = value.clone();
            if !change_leaf(&mut changed, &mut { leaf }, ignored, change) {
                break;
            }
            let current: T = serde_json::from_value(changed).expect("a changed leaf deserializes");
            assert_ne!(current, *facts, "case {name}: leaf {leaf} did not change");
            assert!(
                !compare(facts, &current).is_empty(),
                "case {name}: a change of leaf {leaf} has no divergence"
            );
        }
    }
}

/// With the empty map, an empty list means the facts are equal, as `==` meant before.
#[test]
fn every_template_fact_is_compared() {
    assert_every_leaf_is_compared(
        &golden_by_name(TEMPLATE_FACTS),
        &[],
        next_value,
        compare_unmapped,
    );
}

/// Names and term rows are not compared, so neither is how many term rows a row has: each row
/// keeps one. Every other output fact is compared beyond the tolerance.
#[test]
fn every_output_fact_but_names_and_terms_is_compared() {
    let mut cases: BTreeMap<String, OutputFacts> = golden_by_name(OUTPUTS);
    for facts in cases.values_mut() {
        let mut rows = BTreeSet::new();
        facts
            .echo
            .retain(|row| rows.insert((row.stage_id, row.block_id, row.constraint_id)));
    }
    let names_and_terms = [
        "constraint_name",
        "term_index",
        "variable_kind",
        "variable",
        "coefficient",
        "value",
    ];
    assert_every_leaf_is_compared(&cases, &names_and_terms, far_value, compare_output_facts);
}

/// 701 has a condition, 702 a per-solve upper side and 703 a per-solve lower side.
#[test]
fn a_conditioned_or_per_solve_activation_row_stays_expected() {
    let golden: OutputFacts = golden_case(OUTPUTS, "gated_and_selected");
    for id in [701, 702, 703] {
        let mut current = golden.clone();
        current.stages[0].activation.retain(|row| row.0 != id);

        let divergence = only(compare_output_facts(&golden, &current));

        assert_eq!(
            (divergence.constraint_id, divergence.fact.as_str()),
            (Some(id), "scenario 0, activation: missing row"),
            "{divergence}"
        );
    }
}

#[test]
fn a_condition_change_is_located_in_the_echo() {
    let golden: OutputFacts = golden_case(OUTPUTS, "gated_and_selected");
    let mut current = golden.clone();
    for row in current
        .echo
        .iter_mut()
        .filter(|row| row.constraint_id == 701)
    {
        row.has_active_when = false;
    }

    let divergences = compare_output_facts(&golden, &current);

    assert!(
        !divergences.is_empty()
            && divergences
                .iter()
                .all(|d| d.constraint_id == Some(701) && d.fact == "echo: has_active_when"),
        "{}",
        report(&divergences)
    );
}

#[test]
fn a_map_naming_an_unknown_id_diverges() {
    let facts: ConstraintRowFacts = golden_case(TEMPLATE_FACTS, "inequalities_and_horizon");
    let map = OracleMap {
        constraints: BTreeMap::from([(
            999,
            vec![Segment {
                id: 999,
                cells: Cells::All,
            }],
        )]),
    };

    let divergences = compare_template_facts(&facts, &facts, &map);

    let [divergence] = divergences.as_slice() else {
        panic!("{}", report(&divergences));
    };
    assert_eq!(
        (divergence.constraint_id, divergence.fact.as_str()),
        (Some(999), "map"),
        "{divergence}"
    );
}

fn only(divergences: Vec<Divergence>) -> Divergence {
    let [divergence] = <[Divergence; 1]>::try_from(divergences)
        .unwrap_or_else(|divergences| panic!("{}", report(&divergences)));
    divergence
}

#[test]
fn a_duplicate_current_row_diverges() {
    let facts: ConstraintRowFacts = golden_case(TEMPLATE_FACTS, "inequalities_and_horizon");
    let outputs: OutputFacts = golden_case(OUTPUTS, "inequalities_and_horizon");
    let mut repeated_row = facts.clone();
    let stage = stage_mut(&mut repeated_row, 0);
    stage.rows.push(stage.rows[0].clone());
    let mut repeated_stage = facts.clone();
    repeated_stage.stages.push(facts.stages[0].clone());
    let mut repeated_violation = outputs.clone();
    let scenario_stage = &mut repeated_violation.stages[0];
    scenario_stage
        .violations
        .push(scenario_stage.violations[0].clone());
    let map = OracleMap::default();

    let row = only(compare_template_facts(&facts, &repeated_row, &map));
    let stage = only(compare_template_facts(&facts, &repeated_stage, &map));
    let violation = only(compare_output_facts(&outputs, &repeated_violation));

    assert_eq!(
        (
            row.stage_id,
            row.constraint_id,
            row.block_id,
            row.fact.as_str()
        ),
        (Some(0), Some(501), Some(0), "duplicate row"),
        "{row}"
    );
    assert_eq!(stage.fact, "duplicate stage 0", "{stage}");
    assert_eq!(
        (violation.constraint_id, violation.fact.as_str()),
        (Some(501), "scenario 0, violations: duplicate row"),
        "{violation}"
    );
}

#[test]
#[should_panic(expected = "snapshot stage 0, constraint 501, block 0: entries")]
fn an_unparseable_snapshot_float_panics_naming_the_field() {
    let current: ConstraintRowFacts = golden_case(TEMPLATE_FACTS, "inequalities_and_horizon");
    let mut golden = current.clone();
    stage_mut(&mut golden, 0)
        .rows
        .iter_mut()
        .find(|row| row.constraint_id == 501)
        .and_then(|row| row.entries.first_mut())
        .expect("501 has an entry")
        .value = "one".to_string();

    let _ = compare_template_facts(&golden, &current, &OracleMap::default());
}

fn scale(text: &mut String, factor: f64) {
    let value: f64 = text.parse().expect("a slack value parses");
    *text = format!("{:?}", value * factor);
}

#[test]
fn solver_dependent_fields_have_tolerance() {
    let golden: OutputFacts = golden_case(OUTPUTS, "inequalities_and_horizon");
    let mut current = golden.clone();
    let mut nonzero = current
        .stages
        .iter_mut()
        .flat_map(|stage| {
            let stage_id = stage.stage_id;
            stage.violations.iter_mut().map(move |row| (stage_id, row))
        })
        .filter(|(_, row)| row.2 != "0.0");
    let (_, within) = nonzero.next().expect("a nonzero slack value");
    let (stage_id, beyond) = nonzero.next().expect("a second nonzero slack value");
    let (within_text, beyond_text) = (within.2.clone(), beyond.2.clone());
    scale(&mut within.2, 1.0 + 1e-9);
    scale(&mut beyond.2, 1.0 + 1e-3);
    let beyond_id = beyond.0;
    assert_ne!(within.2, within_text, "the 1e-9 change must move the value");
    assert_ne!(beyond.2, beyond_text);

    let divergences = compare_output_facts(&golden, &current);

    let [divergence] = divergences.as_slice() else {
        panic!("{}", report(&divergences));
    };
    assert_eq!(
        (divergence.stage_id, divergence.constraint_id),
        (
            Some(i32::try_from(stage_id).expect("a stage id fits i32")),
            Some(beyond_id)
        ),
        "{divergence}"
    );
    assert!(divergence.fact.ends_with("slack_value"), "{divergence}");
}

fn stage(facts: &ConstraintRowFacts, stage_id: i32) -> &StageRowFacts {
    facts
        .stages
        .iter()
        .find(|stage| stage.stage_id == stage_id)
        .unwrap_or_else(|| panic!("no stage {stage_id}"))
}

fn stage_mut(facts: &mut ConstraintRowFacts, stage_id: i32) -> &mut StageRowFacts {
    facts
        .stages
        .iter_mut()
        .find(|stage| stage.stage_id == stage_id)
        .unwrap_or_else(|| panic!("no stage {stage_id}"))
}

fn rows_of(stage: &StageRowFacts, constraint_id: i32) -> Vec<&RowFacts> {
    let rows: Vec<_> = stage
        .rows
        .iter()
        .filter(|row| row.constraint_id == constraint_id)
        .collect();
    assert!(
        !rows.is_empty(),
        "stage {}: no row {constraint_id}",
        stage.stage_id
    );
    rows
}

#[test]
fn every_case_exercises_what_it_names() {
    let golden: BTreeMap<String, ConstraintRowFacts> = golden_by_name(TEMPLATE_FACTS);
    let case = |name: &str| {
        golden
            .get(name)
            .unwrap_or_else(|| panic!("the golden has no case {name}"))
    };

    let range_overrides = case("range_overrides");
    let [block_0] = rows_of(stage(range_overrides, 1), 6012)[..] else {
        panic!("range_overrides: 6012 has one row at stage 1");
    };
    let [block_1] = rows_of(stage(range_overrides, 1), 6013)[..] else {
        panic!("range_overrides: 6013 has one row at stage 1");
    };
    assert_eq!((block_0.block_id, block_1.block_id), (0, 1));
    assert!(
        block_0.entries != block_1.entries
            || block_0.lower != block_1.lower
            || block_0.upper != block_1.upper,
        "range_overrides: the two blocks of stage 1 must hold different rows"
    );
    assert!(
        block_0.slack.is_none() && block_1.slack.is_none(),
        "range_overrides: 6012 and 6013 are hard"
    );
    for row in rows_of(stage(range_overrides, 0), 6011) {
        let penalty = row.slack.as_ref().map(|slack| slack.penalty.as_str());
        assert_eq!(penalty, Some("1500.0"), "range_overrides: 6011 at stage 0");
    }

    for stage in &case("gated_and_selected").stages {
        let states: Vec<bool> = stage
            .probes
            .iter()
            .flat_map(|probe| &probe.rows)
            .filter(|row| row.constraint_id == 701)
            .map(|row| row.active)
            .collect();
        assert!(
            states.contains(&true) && states.contains(&false),
            "gated_and_selected: stage {}: 701 must be active under one probe and inactive under \
             another, got {states:?}",
            stage.stage_id
        );
    }

    let formula_with_bounds = case("formula_with_bounds");
    for row in rows_of(stage(formula_with_bounds, 0), 411) {
        assert_eq!(
            (&row.lower, &row.upper),
            (&BoundFacts::PerSolve, &BoundFacts::PerSolve),
            "formula_with_bounds: 411 at stage 0"
        );
    }
    for row in rows_of(stage(formula_with_bounds, 1), 401) {
        assert_eq!(
            row.lower,
            BoundFacts::Absent,
            "formula_with_bounds: 401 at stage 1"
        );
    }
}

#[test]
fn outputs_cover_violation_and_activation() {
    let golden: BTreeMap<String, OutputFacts> = golden_by_name(OUTPUTS);
    let case = |name: &str| {
        golden
            .get(name)
            .unwrap_or_else(|| panic!("the golden has no case {name}"))
    };

    assert!(
        case("inequalities_and_horizon")
            .stages
            .iter()
            .flat_map(|stage| &stage.violations)
            .any(|(_, _, slack_value, _)| slack_value != "0.0"),
        "inequalities_and_horizon: no violation has a nonzero slack"
    );

    let states: BTreeSet<bool> = case("gated_and_selected")
        .stages
        .iter()
        .flat_map(|stage| &stage.activation)
        .filter(|&&(constraint_id, ..)| constraint_id == 701)
        .map(|&(_, _, active, ..)| active)
        .collect();
    assert_eq!(
        states,
        BTreeSet::from([false, true]),
        "gated_and_selected: 701 must be active in some simulated stage and inactive in another"
    );
}

const CASE_VAR: &str = "NOVOMODELO_CONSTRAINT_ORACLE_CASE";
const SNAPSHOT_VAR: &str = "NOVOMODELO_CONSTRAINT_ORACLE_SNAPSHOT";
const MAP_VAR: &str = "NOVOMODELO_CONSTRAINT_ORACLE_MAP";

/// Member-major and block-minor, as [`constraint_row_facts`] reads a probe.
fn mean_load_probes(case_dir: &Path, system: &System, setup: &StudySetup) -> Vec<Vec<Vec<f64>>> {
    let means: BTreeMap<(i32, i32), f64> = novomodelo_io::parse_load_seasonal_stats(
        &case_dir.join("scenarios/load_seasonal_stats.parquet"),
    )
    .expect("the case's load stats must parse")
    .into_iter()
    .map(|row| ((row.bus_id.0, row.stage_id), row.mean_mw))
    .collect();
    let stochastic = &setup.inputs.stochastic;
    let members = &stochastic.entity_order()[stochastic.class_dimensions().load_bus_range()];
    let geometry = &setup.inputs.stage_data.stage_templates.geometry_per_stage;
    [0.5, 1.0, 1.5]
        .into_iter()
        .map(|factor| {
            system
                .stages()
                .iter()
                .filter(|stage| stage.id >= 0)
                .zip(geometry)
                .map(|(stage, geometry)| {
                    members
                        .iter()
                        .flat_map(|bus| {
                            let mean = means.get(&(bus.0, stage.id)).unwrap_or_else(|| {
                                panic!("bus {}: no mean load at stage {}", bus.0, stage.id)
                            });
                            std::iter::repeat_n(factor * mean, geometry.n_blks)
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn read_map(path: Option<&Path>) -> OracleMap {
    let Some(path) = path else {
        return OracleMap::default();
    };
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read the map {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("the map {} must parse: {error}", path.display()))
}

#[test]
fn a_map_file_round_trips() {
    let map = OracleMap {
        constraints: BTreeMap::from([(
            449,
            vec![
                Segment {
                    id: 1001,
                    cells: Cells::Listed(vec![(0, 0), (0, 1)]),
                },
                Segment {
                    id: 1002,
                    cells: Cells::All,
                },
            ],
        )]),
    };
    let dir = TempDir::new().expect("a temporary directory");
    let path = dir.path().join("map.json");
    let text = serde_json::to_string_pretty(&map).expect("the map must serialize");
    std::fs::write(&path, &text).expect("write the map");

    assert_eq!(
        serde_json::from_str::<Value>(&text).expect("the map is JSON"),
        json!({ "constraints": { "449": [
            { "id": 1001, "cells": { "Listed": [[0, 0], [0, 1]] } },
            { "id": 1002, "cells": "All" },
        ] } })
    );
    assert_eq!(read_map(Some(&path)), map);
    assert_eq!(read_map(None), OracleMap::default());
}

/// An external snapshot taken before the facts stopped listing load-coupling buses and columns
/// holds both, empty, in every stage.
#[test]
fn a_snapshot_with_empty_coupling_fields_reads_as_its_facts() {
    let setup = setup(&case(&gated_pair(1000.0)));
    let facts = constraint_row_facts(&setup, &probes(&setup, &[0.0]));
    let mut snapshot = serde_json::to_value(&facts).expect("the facts serialize");
    for stage in snapshot["stages"].as_array_mut().expect("a stages array") {
        stage["coupling_buses"] = json!([]);
        stage["coupling_columns"] = json!([]);
    }
    let dir = TempDir::new().expect("a temporary directory");
    let path = dir.path().join("snapshot.json");
    std::fs::write(&path, snapshot.to_string()).expect("write the snapshot");

    let read: ConstraintRowFacts = read_frozen(&path);

    assert_eq!(read, facts);
}

#[test]
#[ignore = "needs a converted case; set NOVOMODELO_CONSTRAINT_ORACLE_CASE to its directory, \
            NOVOMODELO_CONSTRAINT_ORACLE_SNAPSHOT to its facts file and, when the case renames \
            snapshot cells, NOVOMODELO_CONSTRAINT_ORACLE_MAP to its map file"]
fn external_case_template_facts() {
    let case_dir = std::env::var(CASE_VAR).expect(CASE_VAR);
    let snapshot = std::env::var(SNAPSHOT_VAR).expect(SNAPSHOT_VAR);
    let (case_dir, snapshot) = (Path::new(&case_dir), Path::new(&snapshot));
    let frozen: ConstraintRowFacts = read_frozen(snapshot);
    let map = read_map(std::env::var_os(MAP_VAR).map(PathBuf::from).as_deref());

    let config = novomodelo_io::parse_config(&case_dir.join("config.json"))
        .expect("the case's config must parse");
    let boundary = novomodelo_sddp::resolve_boundary_state_requirements(case_dir, &config)
        .expect("the case's boundary must resolve");
    let (system, setup) = fresh_system_and_setup_with_boundary(case_dir, boundary, |_| {});
    let facts = constraint_row_facts(&setup, &mean_load_probes(case_dir, &system, &setup));

    let divergences = compare_template_facts(&frozen, &facts, &map);
    assert!(
        divergences.is_empty(),
        "the facts diverge from the frozen snapshot {}: {}",
        snapshot.display(),
        report(&divergences)
    );
}
