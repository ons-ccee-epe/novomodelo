//! Compares current constraint-row facts and simulation outputs with their golden by identity, and
//! template facts with an external snapshot under a declarative map that renames the cells of a
//! snapshot constraint. Template facts must hold the same bits; the output fields that depend on
//! the solver's vertex agree within a relative `1e-6` with an absolute floor of `1e-6`.

#![allow(clippy::panic)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Debug, Display};

use novomodelo_sddp::test_support::constraint_row_facts::{
    BoundFacts, ConstraintRowFacts, ProbeFacts, RowFacts, SlackColumnFacts, StageRowFacts,
};
use serde::{Deserialize, Serialize};

use super::constraint_oracle_cases::{EchoFacts, OutputFacts, ScenarioStageFacts};

const REL_TOLERANCE: f64 = 1e-6;
const ABS_TOLERANCE: f64 = 1e-6;

/// `(constraint_id, block_id)`.
type RowKey = (i32, Option<usize>);

/// The current ids of the snapshot's constraints. A constraint without an entry keeps its id in
/// every cell; one with an entry needs exactly one segment holding each of its cells.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OracleMap {
    pub constraints: BTreeMap<i32, Vec<Segment>>,
}

/// The cells of a snapshot constraint that the current facts call `id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub id: i32,
    pub cells: Cells,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Cells {
    All,
    /// `(stage_id, block_id)`.
    Listed(Vec<(i32, usize)>),
}

impl Cells {
    fn contains(&self, (stage_id, block_id): (i32, Option<usize>)) -> bool {
        match self {
            Cells::All => true,
            Cells::Listed(cells) => {
                block_id.is_some_and(|block_id| cells.contains(&(stage_id, block_id)))
            }
        }
    }
}

/// One fact that differs. A `None` location field is a fact that belongs to no stage, constraint
/// or block; a field divergence of a renamed row carries the snapshot's id.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Divergence {
    pub stage_id: Option<i32>,
    pub constraint_id: Option<i32>,
    pub block_id: Option<usize>,
    pub fact: String,
    pub expected: String,
    pub actual: String,
    pub ulp_distance: Option<u64>,
}

impl Display for Divergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: expected {}, actual {}",
            location(self.stage_id, self.constraint_id, self.block_id, &self.fact),
            self.expected,
            self.actual
        )?;
        match self.ulp_distance {
            Some(ulp) => write!(f, " ({ulp} ulp)"),
            None => Ok(()),
        }
    }
}

fn location(
    stage_id: Option<i32>,
    constraint_id: Option<i32>,
    block_id: Option<usize>,
    fact: &str,
) -> String {
    let dash = |id: Option<String>| id.unwrap_or_else(|| "-".to_string());
    format!(
        "stage {}, constraint {}, block {}: {fact}",
        dash(stage_id.map(|id| id.to_string())),
        dash(constraint_id.map(|id| id.to_string())),
        dash(block_id.map(|id| id.to_string()))
    )
}

/// Every divergence of `current` from `frozen`, ordered by `(stage_id, constraint_id, block_id,
/// fact)`. Rows pair by identity, never by position.
///
/// # Panics
///
/// Naming the field, when a float of either side does not parse.
#[must_use]
pub fn compare_template_facts(
    frozen: &ConstraintRowFacts,
    current: &ConstraintRowFacts,
    map: &OracleMap,
) -> Vec<Divergence> {
    let mut diffs = Diffs::default();
    let known = frozen
        .stages
        .iter()
        .flat_map(|stage| &stage.rows)
        .map(|row| row.constraint_id)
        .collect();
    diffs.unknown_map_ids(map, &known);
    let stages = paired(
        &mut diffs,
        NOWHERE,
        "stage",
        (&frozen.stages, &current.stages),
        |stage| (stage.stage_id, stage),
    );
    for (stage_id, expected, actual) in stages {
        if let Some((expected, actual)) = diffs.both(At::stage(stage_id), "stage", expected, actual)
        {
            compare_stage(&mut diffs, map, expected, actual);
        }
    }
    diffs.into_sorted()
}

/// Every divergence of `current` from `golden`, ordered as [`compare_template_facts`] orders
/// them. The echo compares one record of bound and configuration fields per row; names and terms
/// are not compared.
///
/// # Panics
///
/// Naming the field, when a float of either side does not parse.
#[must_use]
pub fn compare_output_facts(golden: &OutputFacts, current: &OutputFacts) -> Vec<Divergence> {
    let mut diffs = Diffs::default();
    compare_echo(&mut diffs, &golden.echo, &current.echo);
    let stages = paired(
        &mut diffs,
        NOWHERE,
        "scenario stage",
        (&golden.stages, &current.stages),
        |stage| ((stage.scenario_id, stage.stage_id), stage),
    );
    for ((scenario_id, stage_id), expected, actual) in stages {
        let stage_id = i32::try_from(stage_id).expect("a stage id fits i32");
        let scenario = format!("scenario {scenario_id}");
        if let Some((expected, actual)) =
            diffs.both(At::stage(stage_id), &scenario, expected, actual)
        {
            compare_scenario_stage(&mut diffs, stage_id, &scenario, expected, actual);
        }
    }
    diffs.into_sorted()
}

fn compare_stage(
    diffs: &mut Diffs,
    map: &OracleMap,
    frozen: &StageRowFacts,
    current: &StageRowFacts,
) {
    let stage_id = frozen.stage_id;
    let at = At::stage(stage_id);
    diffs.same(at, "n_blocks", &frozen.n_blocks, &current.n_blocks);
    diffs.same(
        at,
        "outside_family_sha256",
        &frozen.outside_family_sha256,
        &current.outside_family_sha256,
    );
    compare_keyed(
        diffs,
        map,
        stage_id,
        "",
        (&frozen.rows, &current.rows),
        |row| (row.constraint_id, Some(row.block_id)),
        compare_row,
    );
    diffs.same(
        at,
        "probe count",
        &frozen.probes.len(),
        &current.probes.len(),
    );
    for (p, (expected, actual)) in frozen.probes.iter().zip(&current.probes).enumerate() {
        compare_probe(
            diffs,
            map,
            stage_id,
            &format!("probes[{p}]: "),
            expected,
            actual,
        );
    }
}

fn compare_row(diffs: &mut Diffs, at: At<'_>, frozen: &RowFacts, current: &RowFacts) {
    diffs.same(
        at,
        "is_stage_level",
        &frozen.is_stage_level,
        &current.is_stage_level,
    );
    diffs.same(at, "gated", &frozen.gated, &current.gated);
    let entries = paired(
        diffs,
        at,
        "entry",
        (&frozen.entries, &current.entries),
        |entry| (entry.column, entry.value.as_str()),
    );
    for (column, expected, actual) in entries {
        diffs.bits(at, &format!("entries[{column:?}]"), expected, actual);
    }
    diffs.bound(at, "lower", &frozen.lower, &current.lower);
    diffs.bound(at, "upper", &frozen.upper, &current.upper);
    let Some((expected, actual)) =
        diffs.both(at, "slack", frozen.slack.as_ref(), current.slack.as_ref())
    else {
        return;
    };
    diffs.bits(
        at,
        "slack.penalty",
        Some(&expected.penalty),
        Some(&actual.penalty),
    );
    diffs.same(at, "slack.basis", &expected.basis, &actual.basis);
    diffs.slack_column(at, "slack.plus", &expected.plus, &actual.plus);
    if let Some((expected, actual)) = diffs.both(
        at,
        "slack.minus",
        expected.minus.as_ref(),
        actual.minus.as_ref(),
    ) {
        diffs.slack_column(at, "slack.minus", expected, actual);
    }
}

fn compare_probe(
    diffs: &mut Diffs,
    map: &OracleMap,
    stage_id: i32,
    prefix: &str,
    frozen: &ProbeFacts,
    current: &ProbeFacts,
) {
    let at = At {
        prefix,
        ..At::stage(stage_id)
    };
    diffs.same(at, "load count", &frozen.loads.len(), &current.loads.len());
    for (i, (expected, actual)) in frozen.loads.iter().zip(&current.loads).enumerate() {
        diffs.bits(at, &format!("loads[{i}]"), Some(expected), Some(actual));
    }
    compare_keyed(
        diffs,
        map,
        stage_id,
        prefix,
        (&frozen.rows, &current.rows),
        |row| (row.constraint_id, Some(row.block_id)),
        |diffs, at, expected, actual| {
            diffs.same(at, "active", &expected.active, &actual.active);
            diffs.bits(at, "lower", Some(&expected.lower), Some(&actual.lower));
            diffs.bits(at, "upper", Some(&expected.upper), Some(&actual.upper));
        },
    );
}

/// The echo repeats a row's bound and configuration fields on each of its term rows, so every
/// snapshot term row of a row is compared with every current one.
fn compare_echo(diffs: &mut Diffs, frozen: &[EchoFacts], current: &[EchoFacts]) {
    let stage_ids: BTreeSet<i32> = frozen
        .iter()
        .chain(current)
        .map(|row| row.stage_id)
        .collect();
    for stage_id in stage_ids {
        compare_keyed(
            diffs,
            &OracleMap::default(),
            stage_id,
            "echo: ",
            (&echo_rows(frozen, stage_id), &echo_rows(current, stage_id)),
            |&(key, _)| key,
            |diffs, at, (_, expected), (_, actual)| {
                for expected in expected {
                    for actual in actual {
                        compare_echo_row(diffs, at, expected, actual);
                    }
                }
            },
        );
    }
}

/// The term rows of each row of the echo at `stage_id`.
fn echo_rows(echo: &[EchoFacts], stage_id: i32) -> Vec<(RowKey, Vec<&EchoFacts>)> {
    let mut rows: BTreeMap<RowKey, Vec<&EchoFacts>> = BTreeMap::new();
    for row in echo.iter().filter(|row| row.stage_id == stage_id) {
        rows.entry((row.constraint_id, echo_block(row)))
            .or_default()
            .push(row);
    }
    rows.into_iter().collect()
}

fn echo_block(row: &EchoFacts) -> Option<usize> {
    row.block_id
        .map(|block_id| usize::try_from(block_id).expect("a block id is non-negative"))
}

fn compare_echo_row(diffs: &mut Diffs, at: At<'_>, frozen: &EchoFacts, current: &EchoFacts) {
    diffs.bits(
        at,
        "bound_lower",
        frozen.bound_lower.as_deref(),
        current.bound_lower.as_deref(),
    );
    diffs.bits(
        at,
        "bound_upper",
        frozen.bound_upper.as_deref(),
        current.bound_upper.as_deref(),
    );
    diffs.same(
        at,
        "lower_per_solve",
        &frozen.lower_per_solve,
        &current.lower_per_solve,
    );
    diffs.same(
        at,
        "upper_per_solve",
        &frozen.upper_per_solve,
        &current.upper_per_solve,
    );
    diffs.same(
        at,
        "derived_shape",
        &frozen.derived_shape,
        &current.derived_shape,
    );
    diffs.same(
        at,
        "has_active_when",
        &frozen.has_active_when,
        &current.has_active_when,
    );
    diffs.same(
        at,
        "slack_enabled",
        &frozen.slack_enabled,
        &current.slack_enabled,
    );
    diffs.bits(
        at,
        "slack_penalty",
        frozen.slack_penalty.as_deref(),
        current.slack_penalty.as_deref(),
    );
    diffs.same(
        at,
        "slack_penalty_basis",
        &frozen.slack_penalty_basis,
        &current.slack_penalty_basis,
    );
}

fn compare_scenario_stage(
    diffs: &mut Diffs,
    stage_id: i32,
    scenario: &str,
    frozen: &ScenarioStageFacts,
    current: &ScenarioStageFacts,
) {
    compare_keyed(
        diffs,
        &OracleMap::default(),
        stage_id,
        &format!("{scenario}, violations: "),
        (&frozen.violations, &current.violations),
        |row| (row.0, row.1.map(|block_id| block_id as usize)),
        |diffs, at, expected, actual| {
            diffs.tolerant_texts(at, "slack_value", &expected.2, &actual.2);
            diffs.tolerant_texts(at, "slack_cost", &expected.3, &actual.3);
        },
    );
    compare_keyed(
        diffs,
        &OracleMap::default(),
        stage_id,
        &format!("{scenario}, activation: "),
        (&frozen.activation, &current.activation),
        |row| (row.0, row.1.map(|block_id| block_id as usize)),
        |diffs, at, expected, actual| {
            diffs.same(at, "active", &expected.2, &actual.2);
            diffs.bits(
                at,
                "bound_lower",
                expected.3.as_deref(),
                actual.3.as_deref(),
            );
            diffs.bits(
                at,
                "bound_upper",
                expected.4.as_deref(),
                actual.4.as_deref(),
            );
        },
    );

    let prefix = format!("{scenario}, violation_cost: ");
    let costs = paired(
        diffs,
        At {
            prefix: &prefix,
            ..At::stage(stage_id)
        },
        "block",
        (&frozen.violation_cost, &current.violation_cost),
        |(block_id, cost)| (*block_id, cost.as_str()),
    );
    for (block_id, expected, actual) in costs {
        let at = At {
            block_id: block_id.map(|block_id| block_id as usize),
            prefix: &prefix,
            ..At::stage(stage_id)
        };
        if let Some((expected, actual)) = diffs.both(at, "block", expected, actual) {
            diffs.tolerant_texts(at, "generic_violation_cost", expected, actual);
        }
    }

    let prefix = format!("{scenario}, spot_price: ");
    let prices = paired(
        diffs,
        At {
            prefix: &prefix,
            ..At::stage(stage_id)
        },
        "bus and block",
        (&frozen.spot_price, &current.spot_price),
        |(bus_id, block_id, price)| ((*bus_id, *block_id), price.as_str()),
    );
    for ((bus_id, block_id), expected, actual) in prices {
        let at = At {
            block_id: block_id.map(|block_id| block_id as usize),
            prefix: &prefix,
            ..At::stage(stage_id)
        };
        let bus = format!("bus {bus_id}");
        if let Some((expected, actual)) = diffs.both(at, &bus, expected, actual) {
            diffs.tolerant_texts(at, &bus, expected, actual);
        }
    }
}

/// Pairs each snapshot record `(id, block)` with the current record `(target, block)`, `target`
/// being the id of the one segment holding the cell `(stage_id, block)`, and compares the pair
/// with `fields`. Reports a cell held by no segment or by several, a record with no target, a
/// current record that is the target of no snapshot record or of several, and a current key that
/// repeats (`duplicate row`).
fn compare_keyed<T>(
    diffs: &mut Diffs,
    map: &OracleMap,
    stage_id: i32,
    prefix: &str,
    (frozen, current): (&[T], &[T]),
    key: impl Fn(&T) -> RowKey,
    mut fields: impl FnMut(&mut Diffs, At<'_>, &T, &T),
) {
    let row_at = |id, block_id| At {
        stage_id: Some(stage_id),
        constraint_id: Some(id),
        block_id,
        prefix,
    };
    let mut current_by_key: BTreeMap<RowKey, &T> = BTreeMap::new();
    for row in current {
        let (id, block_id) = key(row);
        if current_by_key.insert((id, block_id), row).is_some() {
            let at = row_at(id, block_id);
            diffs.push(at, "duplicate row", "one row".to_string(), several(), None);
        }
    }
    let mut claims: BTreeMap<RowKey, Vec<i32>> = BTreeMap::new();
    for expected in frozen {
        let (id, block_id) = key(expected);
        let at = row_at(id, block_id);
        let Some(target) = map.target(diffs, at, id, (stage_id, block_id)) else {
            continue;
        };
        claims.entry((target, block_id)).or_default().push(id);
        match current_by_key.get(&(target, block_id)) {
            Some(actual) => fields(diffs, at, expected, actual),
            None => diffs.push(at, "missing row", format!("row {target}"), none(), None),
        }
    }
    for &(id, block_id) in current_by_key.keys() {
        let at = row_at(id, block_id);
        match claims.get(&(id, block_id)).map(Vec::as_slice) {
            None => diffs.push(at, "extra row", none(), format!("row {id}"), None),
            Some([_]) => {}
            Some(ids) => diffs.push(
                at,
                "map",
                "one snapshot row".to_string(),
                format!("snapshot rows {ids:?}"),
                None,
            ),
        }
    }
}

impl OracleMap {
    /// The current id of the snapshot constraint `id` at `cell`; `None`, after reporting it,
    /// when no segment or several hold the cell.
    fn target(
        &self,
        diffs: &mut Diffs,
        at: At<'_>,
        id: i32,
        cell: (i32, Option<usize>),
    ) -> Option<i32> {
        let Some(segments) = self.constraints.get(&id) else {
            return Some(id);
        };
        let ids: Vec<i32> = segments
            .iter()
            .filter(|segment| segment.cells.contains(cell))
            .map(|segment| segment.id)
            .collect();
        if let [target] = ids[..] {
            return Some(target);
        }
        diffs.push(
            at,
            "map",
            "one segment".to_string(),
            format!("segments {ids:?}"),
            None,
        );
        None
    }
}

/// Every key that `entry` gives an item of either side, ascending, with its value on each side. A
/// key that repeats on the current side is reported at `at` as `duplicate {what} {key:?}`.
fn paired<'a, T, K: Ord + Copy + Debug, V>(
    diffs: &mut Diffs,
    at: At<'_>,
    what: &str,
    (frozen, current): (&'a [T], &'a [T]),
    entry: impl Fn(&'a T) -> (K, V),
) -> Vec<(K, Option<V>, Option<V>)> {
    let mut frozen: BTreeMap<K, V> = frozen.iter().map(&entry).collect();
    let mut current_by_key = BTreeMap::new();
    for (key, value) in current.iter().map(&entry) {
        if current_by_key.insert(key, value).is_some() {
            let fact = format!("duplicate {what} {key:?}");
            diffs.push(at, &fact, "one".to_string(), several(), None);
        }
    }
    let mut current = current_by_key;
    let keys: BTreeSet<K> = frozen.keys().chain(current.keys()).copied().collect();
    keys.into_iter()
        .map(|key| (key, frozen.remove(&key), current.remove(&key)))
        .collect()
}

/// Where a fact sits; `prefix` names the table or the probe before the fact.
#[derive(Clone, Copy)]
struct At<'a> {
    stage_id: Option<i32>,
    constraint_id: Option<i32>,
    block_id: Option<usize>,
    prefix: &'a str,
}

const NOWHERE: At<'static> = At {
    stage_id: None,
    constraint_id: None,
    block_id: None,
    prefix: "",
};

impl At<'static> {
    fn stage(stage_id: i32) -> Self {
        At {
            stage_id: Some(stage_id),
            ..NOWHERE
        }
    }
}

fn none() -> String {
    "none".to_string()
}

fn several() -> String {
    "several".to_string()
}

fn parse(at: At<'_>, fact: &str, side: &str, text: &str) -> f64 {
    text.parse().unwrap_or_else(|_| {
        panic!(
            "{side} {}: {text:?} is not a float",
            location(
                at.stage_id,
                at.constraint_id,
                at.block_id,
                &format!("{}{fact}", at.prefix)
            )
        )
    })
}

/// Representable `f64` values from `a` to `b`; `-0.0` and `0.0` are 0 apart.
fn ulp_distance(a: f64, b: f64) -> u64 {
    let ordered = |value: f64| {
        let bits = value.to_bits().cast_signed();
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    ordered(a).abs_diff(ordered(b))
}

#[derive(Default)]
struct Diffs(Vec<Divergence>);

impl Diffs {
    fn push(
        &mut self,
        at: At<'_>,
        fact: &str,
        expected: String,
        actual: String,
        ulp_distance: Option<u64>,
    ) {
        self.0.push(Divergence {
            stage_id: at.stage_id,
            constraint_id: at.constraint_id,
            block_id: at.block_id,
            fact: format!("{}{fact}", at.prefix),
            expected,
            actual,
            ulp_distance,
        });
    }

    fn unknown_map_ids(&mut self, map: &OracleMap, known: &BTreeSet<i32>) {
        for &id in map.constraints.keys().filter(|id| !known.contains(id)) {
            let at = At {
                constraint_id: Some(id),
                ..NOWHERE
            };
            self.push(at, "map", "a snapshot constraint".to_string(), none(), None);
        }
    }

    fn same<T: PartialEq + Debug>(&mut self, at: At<'_>, fact: &str, expected: &T, actual: &T) {
        if expected != actual {
            self.push(
                at,
                fact,
                format!("{expected:?}"),
                format!("{actual:?}"),
                None,
            );
        }
    }

    /// Two floats, each its `{:?}` text, compared by their bits.
    fn bits(&mut self, at: At<'_>, fact: &str, expected: Option<&str>, actual: Option<&str>) {
        if expected == actual {
            return;
        }
        let ulp_distance = expected.zip(actual).map(|(expected, actual)| {
            ulp_distance(
                parse(at, fact, "snapshot", expected),
                parse(at, fact, "current", actual),
            )
        });
        let text = |value: Option<&str>| value.map_or_else(none, str::to_string);
        self.push(at, fact, text(expected), text(actual), ulp_distance);
    }

    fn bound(&mut self, at: At<'_>, fact: &str, expected: &BoundFacts, actual: &BoundFacts) {
        match (expected, actual) {
            (BoundFacts::Fixed(expected), BoundFacts::Fixed(actual)) => {
                self.bits(at, fact, Some(expected), Some(actual));
            }
            _ => self.same(at, fact, expected, actual),
        }
    }

    fn slack_column(
        &mut self,
        at: At<'_>,
        column: &str,
        expected: &SlackColumnFacts,
        actual: &SlackColumnFacts,
    ) {
        for (field, expected, actual) in [
            ("coefficient", &expected.coefficient, &actual.coefficient),
            ("objective", &expected.objective, &actual.objective),
            ("lower", &expected.lower, &actual.lower),
            ("upper", &expected.upper, &actual.upper),
        ] {
            self.bits(
                at,
                &format!("{column}.{field}"),
                Some(expected),
                Some(actual),
            );
        }
    }

    fn tolerant(&mut self, at: At<'_>, fact: &str, expected: f64, actual: f64) {
        let close = expected.to_bits() == actual.to_bits()
            || (expected - actual).abs()
                <= (REL_TOLERANCE * expected.abs().max(actual.abs())).max(ABS_TOLERANCE);
        if !close {
            self.push(
                at,
                fact,
                format!("{expected:?}"),
                format!("{actual:?}"),
                Some(ulp_distance(expected, actual)),
            );
        }
    }

    fn tolerant_texts(&mut self, at: At<'_>, fact: &str, expected: &str, actual: &str) {
        self.tolerant(
            at,
            fact,
            parse(at, fact, "snapshot", expected),
            parse(at, fact, "current", actual),
        );
    }

    /// Both sides of a fact that may be absent; reports `missing {what}` or `extra {what}` when
    /// only one side holds it.
    fn both<'v, V: ?Sized>(
        &mut self,
        at: At<'_>,
        what: &str,
        expected: Option<&'v V>,
        actual: Option<&'v V>,
    ) -> Option<(&'v V, &'v V)> {
        match (expected, actual) {
            (Some(expected), Some(actual)) => return Some((expected, actual)),
            (Some(_), None) => {
                self.push(
                    at,
                    &format!("missing {what}"),
                    "present".to_string(),
                    none(),
                    None,
                );
            }
            (None, Some(_)) => {
                self.push(
                    at,
                    &format!("extra {what}"),
                    none(),
                    "present".to_string(),
                    None,
                );
            }
            (None, None) => {}
        }
        None
    }

    fn into_sorted(mut self) -> Vec<Divergence> {
        self.0.sort();
        self.0.dedup();
        self.0
    }
}
