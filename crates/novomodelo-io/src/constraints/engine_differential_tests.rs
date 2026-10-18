//! Compares the expression engine with the old generic-constraint parser: per constraint, every
//! variable's coefficient and each bound endpoint, evaluated under three leaf valuations.

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use novomodelo_core::{
    AffineBound, CoefficientRef, EntityId, GenericConstraint, Line, VariableRef, is_exact_zero,
};
use serde_json::Value;

use crate::constraints::{LineBusPairIndex, build_line_bus_pair_index, parse_generic_constraints};
use crate::expression::lower::{LoweredRelation, NamedTable, Scope, lower_relation};
use crate::parse_scalar_parameters_json;
use crate::test_support::write_json;

type Leaf = fn(EntityId) -> f64;

fn golden(id: EntityId) -> f64 {
    1.0 + f64::from(id.0) * 0.618_033_988_749_894_9
}

fn positive_zero(_: EntityId) -> f64 {
    0.0
}

fn negative_zero(_: EntityId) -> f64 {
    -0.0
}

const VALUATIONS: [(&str, Leaf); 3] = [
    ("golden", golden),
    ("positive-zero", positive_zero),
    ("negative-zero", negative_zero),
];

fn announce_valuations() {
    for (name, _) in VALUATIONS {
        println!("valuation: {name}");
    }
}

/// One value the two paths disagree on. A `None` side holds no coefficient or no endpoint.
struct Mismatch {
    valuation: &'static str,
    constraint: EntityId,
    place: String,
    old: Option<f64>,
    new: Option<f64>,
}

impl Mismatch {
    fn within_last_bits(&self) -> bool {
        match (self.old, self.new) {
            (Some(old), Some(new)) => {
                (old - new).abs() <= 4.0 * f64::EPSILON * old.abs().max(new.abs())
            }
            _ => false,
        }
    }
}

fn show(value: Option<f64>) -> String {
    value.map_or_else(
        || "absent".to_owned(),
        |v| format!("{v:e} ({:#018x})", v.to_bits()),
    )
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: constraint {}, {}: old {} vs engine {}",
            self.valuation,
            self.constraint.0,
            self.place,
            show(self.old),
            show(self.new)
        )
    }
}

fn same_bits(old: Option<f64>, new: Option<f64>) -> bool {
    old.map(f64::to_bits) == new.map(f64::to_bits)
}

/// The builder's sum over the old terms on each variable, in canonical order and folded from the
/// first value; an all-literal sum that is exactly zero is no coefficient.
fn old_coefficients(constraint: &GenericConstraint, leaf: Leaf) -> Vec<(VariableRef, f64)> {
    let mut sums: Vec<(VariableRef, f64, bool)> = Vec::new();
    for term in &constraint.expression.terms {
        let (c, literal) = match term.coefficient {
            CoefficientRef::Literal(v) => (v, true),
            CoefficientRef::Parameter(id) => (leaf(id), false),
        };
        let value = c * term.scale;
        match sums.iter_mut().find(|(v, _, _)| *v == term.variable) {
            Some((_, sum, all_literal)) => {
                *sum += value;
                *all_literal &= literal;
            }
            None => sums.push((term.variable, value, literal)),
        }
    }
    sums.into_iter()
        .filter(|&(_, sum, all_literal)| !(all_literal && is_exact_zero(sum)))
        .map(|(variable, sum, _)| (variable, sum))
        .collect()
}

/// The layout's evaluation of an old endpoint: the constant, then each term, then each product.
fn old_endpoint(bound: &AffineBound, leaf: Leaf) -> f64 {
    let linear = bound
        .terms
        .iter()
        .fold(bound.constant, |acc, &(k, p)| acc + k * leaf(p));
    bound
        .products
        .iter()
        .fold(linear, |acc, &(s, l, r)| acc + s * leaf(l) * leaf(r))
}

fn compare_constraint(
    old: &GenericConstraint,
    new: &LoweredRelation,
    mismatches: &mut Vec<Mismatch>,
) {
    let id = old.id;
    let old_present = old_coefficients(old, golden);
    for (variable, value) in &old_present {
        if !new.terms.iter().any(|(v, _)| v == variable) {
            mismatches.push(Mismatch {
                valuation: "every valuation",
                constraint: id,
                place: format!("coefficient of {variable:?}"),
                old: Some(*value),
                new: None,
            });
        }
    }
    for (variable, coefficient) in &new.terms {
        if !old_present.iter().any(|(v, _)| v == variable) {
            mismatches.push(Mismatch {
                valuation: "every valuation",
                constraint: id,
                place: format!("coefficient of {variable:?}"),
                old: None,
                new: Some(coefficient.evaluate(golden)),
            });
        }
    }
    for (valuation, leaf) in VALUATIONS {
        for (variable, old_value) in old_coefficients(old, leaf) {
            let Some((_, coefficient)) = new.terms.iter().find(|(v, _)| *v == variable) else {
                continue;
            };
            let new_value = coefficient.evaluate(leaf);
            if !same_bits(Some(old_value), Some(new_value)) {
                mismatches.push(Mismatch {
                    valuation,
                    constraint: id,
                    place: format!("coefficient of {variable:?}"),
                    old: Some(old_value),
                    new: Some(new_value),
                });
            }
        }
        let endpoints = [
            ("lower endpoint", &old.bound_lower, &new.lower),
            ("upper endpoint", &old.bound_upper, &new.upper),
        ];
        for (place, old_bound, new_bound) in endpoints {
            let old_value = old_bound.as_ref().map(|b| old_endpoint(b, leaf));
            let new_value = new_bound.as_ref().map(|e| e.evaluate(leaf));
            if !same_bits(old_value, new_value) {
                mismatches.push(Mismatch {
                    valuation,
                    constraint: id,
                    place: place.to_owned(),
                    old: old_value,
                    new: new_value,
                });
            }
        }
    }
}

/// Runs the old loader and the engine over one constraint file and returns every value whose
/// bits differ. Either side rejecting the file fails the test, naming `label`.
fn compare_file(
    label: &str,
    json: &str,
    params: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
) -> Vec<Mismatch> {
    let file = write_json(json);
    let old = parse_generic_constraints(file.path(), params, line_index)
        .unwrap_or_else(|e| panic!("{label}: the old parser rejects the file: {e}"));

    let root: Value = serde_json::from_str(json).expect("valid JSON");
    let decls: Vec<(&str, &str)> = root
        .get("expressions")
        .and_then(Value::as_array)
        .map(|declarations| {
            declarations
                .iter()
                .map(|d| {
                    (
                        d["name"].as_str().expect("a name"),
                        d["expression"].as_str().expect("an expression"),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let named = NamedTable::declare(&decls, params, line_index).unwrap_or_else(|(j, d)| {
        panic!(
            "{label}: the engine rejects expressions[{j}]: {}",
            d.render(decls[j].1)
        )
    });
    let scope = Scope {
        params,
        line_index,
        named: &named,
    };

    let constraints = root["constraints"].as_array().expect("a constraints array");
    assert_eq!(
        constraints.len(),
        old.len(),
        "{label}: the two paths load different constraint counts"
    );
    let mut mismatches = Vec::new();
    for constraint in constraints {
        let id = EntityId(
            i32::try_from(constraint["id"].as_i64().expect("an integer id")).expect("an i32 id"),
        );
        let src = constraint["expression"].as_str().expect("an expression");
        let new = lower_relation(src, &scope).unwrap_or_else(|d| {
            panic!(
                "{label}: the engine rejects constraint {}: {}",
                id.0,
                d.render(src)
            )
        });
        let old = old
            .iter()
            .find(|c| c.id == id)
            .unwrap_or_else(|| panic!("{label}: the old parser has no constraint {}", id.0));
        compare_constraint(old, &new, &mut mismatches);
    }
    mismatches
}

/// Every `generic_constraints.json` under `root`, sorted, skipping build output, the
/// gitignored working tree under `plans/` and dot-directories.
fn repository_constraint_files(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let entries = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: cannot list: {e}", dir.display()));
        for entry in entries {
            let path = entry.expect("a directory entry").path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if path.is_dir() {
                if !(name.starts_with('.') || name == "target" || name == "plans") {
                    pending.push(path);
                }
            } else if name == "generic_constraints.json" {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// characterization: the old loader is the oracle; every coefficient and endpoint of every
/// generic-constraint file in the repository keeps its bits under the golden, `+0.0` and `-0.0`
/// valuations.
#[test]
fn committed_constraint_files_lower_bit_identically() {
    announce_valuations();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let files = repository_constraint_files(&root);
    assert!(
        !files.is_empty(),
        "no generic_constraints.json under {}",
        root.display()
    );
    let mut report = Vec::new();
    for path in &files {
        let relative = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .display()
            .to_string();
        let relative = relative.as_str();
        let json = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("{relative}: cannot read: {e}"));
        let parameters = path.with_file_name("generic_parameters.json");
        let params: HashMap<String, EntityId> = if parameters.exists() {
            parse_scalar_parameters_json(&parameters)
                .unwrap_or_else(|e| panic!("{relative}: parameters do not load: {e}"))
                .into_iter()
                .map(|p| (p.name, p.id))
                .collect()
        } else {
            HashMap::new()
        };
        let root: Value = serde_json::from_str(&json).expect("valid JSON");
        assert!(
            root["constraints"]
                .as_array()
                .is_some_and(|c| !c.is_empty()),
            "{relative}: holds no constraint"
        );
        let mismatches = compare_file(relative, &json, &params, &LineBusPairIndex::default());
        report.extend(mismatches.iter().map(|m| format!("{relative}: {m}")));
    }
    assert!(
        report.is_empty(),
        "committed files moved bits:\n{}",
        report.join("\n")
    );
}

/// A shape whose last bit may move because the engine evaluates it in a different order or
/// shape than the old parser and builder.
#[derive(Clone, Copy)]
enum LastBit {
    NestedLiteralScales,
    RemainderSummedInAnotherOrder,
    RepeatedParameterOnOneVariable,
    RepeatedLiteralsOnOneVariable,
    ParameterAndLiteralOnOneVariable,
}

impl LastBit {
    fn label(self) -> &'static str {
        match self {
            LastBit::NestedLiteralScales => "nested literal scales",
            LastBit::RemainderSummedInAnotherOrder => {
                "multi-term remainder summed in another order"
            }
            LastBit::RepeatedParameterOnOneVariable => "repeated parameter on one variable",
            LastBit::RepeatedLiteralsOnOneVariable => "repeated literals on one variable",
            LastBit::ParameterAndLiteralOnOneVariable => "parameter and literal on one variable",
        }
    }
}

struct Entry {
    expression: &'static str,
    named: &'static [(&'static str, &'static str)],
    params: &'static [(&'static str, i32)],
    category: Option<LastBit>,
}

const fn entry(
    expression: &'static str,
    named: &'static [(&'static str, &'static str)],
    params: &'static [(&'static str, i32)],
    category: Option<LastBit>,
) -> Entry {
    Entry {
        expression,
        named,
        params,
        category,
    }
}

const PQ: &[(&str, i32)] = &[("p", 1), ("q", 2)];
const A: &[(&str, i32)] = &[("a", 3)];
const AB: &[(&str, i32)] = &[("a", 3), ("b", 4)];

const HAND_CORPUS: &[Entry] = &[
    entry(
        "2.5 * @p * hydro_generation(1) + thermal_generation(2)",
        &[],
        PQ,
        None,
    ),
    entry("@p * thermal_generation(2) >= 10", &[], PQ, None),
    entry(
        "0.5 * @e",
        &[(
            "e",
            "hydro_generation(1) + 2 * thermal_generation(2) + @p * bus_deficit(0)",
        )],
        PQ,
        None,
    ),
    entry(
        "@e",
        &[("e", "hydro_generation(1) - @p * thermal_generation(2)")],
        PQ,
        None,
    ),
    entry("0.87 * hydro_generation(140)", &[], &[], None),
    entry(
        "2 * (hydro_generation(1) - 0.5 * thermal_generation(2))",
        &[],
        &[],
        None,
    ),
    entry(
        "3 * (hydro_generation(1) + 2 * (thermal_generation(2) - bus_deficit(0)))",
        &[],
        &[],
        None,
    ),
    entry(
        "(hydro_generation(1) + thermal_generation(2)) - bus_deficit(0)",
        &[],
        &[],
        None,
    ),
    entry("hydro_generation(1)", &[], &[], None),
    entry("hydro_generation(1) <= 73", &[], &[], None),
    entry("5 <= hydro_generation(1)", &[], &[], None),
    entry("hydro_generation(1) <= 2 * @p * @q", &[], PQ, None),
    entry(
        "hydro_generation(0) >= @fnese",
        &[("fnese", "hydro_generation(10) + hydro_generation(11)")],
        &[],
        None,
    ),
    entry(
        "line_exchange(source_bus=1, target_bus=2) - line_exchange(source_bus=4, target_bus=3) <= 50",
        &[],
        &[],
        None,
    ),
    entry(
        "hydro_turbined(1, bus=2) + hydro_generation(1, bus=2)",
        &[],
        &[],
        None,
    ),
    entry(
        "hydro_turbined(3, 0, bus=4) - hydro_generation(3, 1, bus=4) >= 0",
        &[],
        &[],
        None,
    ),
    entry("anticipated_decision(5) <= 1", &[], &[], None),
    entry(
        "thermal_generation(5) <= 0.87 * hydro_generation(140) + 73",
        &[],
        &[],
        None,
    ),
    entry(
        "hydro_generation(0) - hydro_generation(0) <= 5",
        &[],
        &[],
        None,
    ),
    entry(
        "bus_deficit(0) <= -0.316 * bus_deficit(1) + 0.5 * bus_deficit(2) + 12352",
        &[],
        &[],
        None,
    ),
    entry(
        "bus_deficit(0) <= 11000 - 0.04 * @val_demanda_3 + 0.06 * @ger_pee_11",
        &[],
        &[("val_demanda_3", 50), ("ger_pee_11", 51)],
        None,
    ),
    entry("bus_deficit(0) <= 73", &[], &[], None),
    entry("hydro_generation(1) + hydro_generation(0)", &[], &[], None),
    entry(
        "hydro_generation(1) + 0 <= thermal_generation(2)",
        &[],
        &[],
        None,
    ),
    entry(
        "hydro_generation(1) >= thermal_generation(2) - @p * @q",
        &[],
        PQ,
        None,
    ),
    entry("hydro_generation(1) >= @p", &[], PQ, None),
    entry("hydro_generation(1) <= -2 * @p * @q", &[], PQ, None),
    entry("hydro_generation(1) <= @a + @b", &[], AB, None),
    entry("hydro_generation(1) <= -@a", &[], A, None),
    entry("hydro_generation(1) <= 2 * @a", &[], A, None),
    entry(
        "hydro_generation(1) - @a >= thermal_generation(2)",
        &[],
        A,
        None,
    ),
    entry(
        "0.1 * (0.3 * (@a * hydro_generation(1)))",
        &[],
        A,
        Some(LastBit::NestedLiteralScales),
    ),
    entry(
        "hydro_generation(1) + 0.1 <= @a + @b + 0.7",
        &[],
        AB,
        Some(LastBit::RemainderSummedInAnotherOrder),
    ),
    entry(
        "0.1 * @a * hydro_generation(1) + 0.2 * @a * hydro_generation(1) >= thermal_generation(2)",
        &[],
        A,
        Some(LastBit::RepeatedParameterOnOneVariable),
    ),
    entry(
        "0.3 * hydro_generation(1) + 0.6 * hydro_generation(1) + 0.1 * hydro_generation(1)",
        &[],
        &[],
        Some(LastBit::RepeatedLiteralsOnOneVariable),
    ),
    entry("line_exchange(3) >= @target", &[], &[("target", 99)], None),
    entry(
        "0.1 * hydro_generation(1) + @a * hydro_generation(1) + 0.2 * hydro_generation(1) <= 1",
        &[],
        A,
        Some(LastBit::ParameterAndLiteralOnOneVariable),
    ),
];

fn two_line_index() -> LineBusPairIndex {
    let line = |id: i32, source: i32, target: i32| Line {
        id: EntityId(id),
        name: format!("L{id}"),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
        source_bus_id: EntityId(source),
        target_bus_id: EntityId(target),
        entry_stage_id: None,
        exit_stage_id: None,
        direct_capacity_mw: 100.0,
        reverse_capacity_mw: 100.0,
        losses_percent: 0.0,
        exchange_cost: 0.0,
    };
    build_line_bus_pair_index(&[line(7, 1, 2), line(8, 3, 4)]).expect("distinct bus pairs")
}

fn entry_json(entry: &Entry) -> String {
    let constraint = serde_json::json!({
        "id": 1,
        "name": "c",
        "expression": entry.expression,
        "slack": { "enabled": false },
    });
    let expressions: Vec<Value> = entry
        .named
        .iter()
        .map(|(name, expression)| serde_json::json!({ "name": name, "expression": expression }))
        .collect();
    serde_json::json!({ "expressions": expressions, "constraints": [constraint] }).to_string()
}

/// characterization: the old loader is the oracle; an untagged entry keeps every bit under the
/// golden, `+0.0` and `-0.0` valuations, and a tagged entry stays within four ulps relative.
#[test]
fn hand_corpus_lowers_like_the_old_parser() {
    announce_valuations();
    let line_index = two_line_index();
    let mut report = Vec::new();
    for entry in HAND_CORPUS {
        let params: HashMap<String, EntityId> = entry
            .params
            .iter()
            .map(|&(name, id)| (name.to_owned(), EntityId(id)))
            .collect();
        let mismatches = compare_file(entry.expression, &entry_json(entry), &params, &line_index);
        match entry.category {
            None => report.extend(
                mismatches
                    .iter()
                    .map(|m| format!("`{}` (untagged): {m}", entry.expression)),
            ),
            Some(category) => {
                report.extend(
                    mismatches
                        .iter()
                        .filter(|m| !m.within_last_bits())
                        .map(|m| format!("`{}` ({}): {m}", entry.expression, category.label())),
                );
                let moved: Vec<&str> = mismatches.iter().map(|m| m.valuation).collect();
                println!(
                    "category {}: `{}`: bits moved under {}",
                    category.label(),
                    entry.expression,
                    if moved.is_empty() {
                        "no valuation".to_owned()
                    } else {
                        moved.join(", ")
                    }
                );
            }
        }
    }
    assert!(
        report.is_empty(),
        "the engine lowers hand entries differently:\n{}",
        report.join("\n")
    );
}
