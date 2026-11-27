//! Golden regression snapshot of the generic-constraint outputs of a converted case run.
//!
//! Without an external oracle the regression oracle is the run itself: three sorted CSV
//! summaries that a human reviews once, the operator commits, and every later run is
//! compared against. The resolved bounds, the activation and the plant availability come
//! from deterministic arithmetic on the case and compare exactly; the violation summary
//! depends on the solver's vertex and compares within a relative 1e-6 (absolute floor 1e-6).
//! Floats are written with `{:?}`, which round-trips exactly, and an empty cell is null.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

use super::run_case;
use super::scenario_constraint_case::{
    ActivationRow, ConstraintOutputs, EchoRow, ViolationRow, assert_constraint_run_invariants,
    read_constraint_outputs, write_scenario_constraint_case,
};

const BASELINE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/scenario_constraint_case_golden"
);

const REL_TOLERANCE: f64 = 1e-6;
const ABS_TOLERANCE: f64 = 1e-6;
const MAX_REPORTED: usize = 50;

/// A row key; `None` sorts first, as a null block does in the outputs.
type Key = Vec<Option<i32>>;

trait Csv: Sized {
    fn put(&self) -> String;
    fn get(cell: &str) -> Result<Self, String>;
}

macro_rules! csv_via_debug_and_parse {
    ($($ty:ty),+) => {$(
        impl Csv for $ty {
            fn put(&self) -> String {
                format!("{self:?}")
            }

            fn get(cell: &str) -> Result<Self, String> {
                cell.parse().map_err(|e| format!("`{cell}`: {e}"))
            }
        }
    )+};
}

csv_via_debug_and_parse!(i32, usize, bool, f64);

impl Csv for String {
    fn put(&self) -> String {
        self.clone()
    }

    fn get(cell: &str) -> Result<Self, String> {
        Ok(cell.to_owned())
    }
}

impl<T: Csv> Csv for Option<T> {
    fn put(&self) -> String {
        self.as_ref().map_or_else(String::new, T::put)
    }

    fn get(cell: &str) -> Result<Self, String> {
        if cell.is_empty() {
            Ok(None)
        } else {
            T::get(cell).map(Some)
        }
    }
}

trait GoldenRow: Sized {
    const FILE: &'static str;
    const HEADER: &'static [&'static str];
    /// The non-key cells are solver values, compared within the tolerance; otherwise exact.
    const TOLERANT: bool;
    fn key(&self) -> Key;
    fn fields(&self) -> Vec<String>;
    fn parse(line: &str) -> Result<Self, String>;
}

fn take<'a, T: Csv>(
    cells: &mut impl Iterator<Item = (&'a str, &'static str)>,
) -> Result<T, String> {
    let (cell, column) = cells.next().expect("the cell count was checked");
    T::get(cell).map_err(|e| format!("column `{column}`: {e}"))
}

macro_rules! golden_row {
    (
        $(#[$doc:meta])*
        $name:ident, $file:literal, tolerant: $tolerant:literal,
        key { $($key:ident: $key_ty:ty),+ $(,)? },
        values { $($field:ident: $ty:ty),+ $(,)? }
    ) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq)]
        pub(crate) struct $name {
            $(pub(crate) $key: $key_ty,)+
            $(pub(crate) $field: $ty,)+
        }

        impl GoldenRow for $name {
            const FILE: &'static str = $file;
            const HEADER: &'static [&'static str] =
                &[$(stringify!($key),)+ $(stringify!($field)),+];
            const TOLERANT: bool = $tolerant;

            fn key(&self) -> Key {
                vec![$(Option::<i32>::from(self.$key)),+]
            }

            fn fields(&self) -> Vec<String> {
                vec![$(self.$key.put(),)+ $(self.$field.put()),+]
            }

            fn parse(line: &str) -> Result<Self, String> {
                let cells: Vec<&str> = line.split(',').collect();
                if cells.len() != Self::HEADER.len() {
                    return Err(format!(
                        "expected {} cells, found {}",
                        Self::HEADER.len(),
                        cells.len()
                    ));
                }
                let mut cells = cells.into_iter().zip(Self::HEADER.iter().copied());
                Ok(Self {
                    $($key: take(&mut cells)?,)+
                    $($field: take(&mut cells)?,)+
                })
            }
        }
    };
}

golden_row! {
    /// One `(constraint, stage, block)` cell of the echo: the resolved bounds, the slack
    /// and its penalty, whether `active_when` gates it, the number of LP terms and the sum in
    /// MW of the `hydro_availability` capacities the cell read.
    BoundsSummary, "resolved_bounds.csv", tolerant: false,
    key { constraint_id: i32, stage_id: i32, block_id: Option<i32> },
    values {
        bound_lower: Option<f64>,
        bound_upper: Option<f64>,
        lower_per_solve: bool,
        upper_per_solve: bool,
        has_active_when: bool,
        slack_enabled: bool,
        slack_penalty: Option<f64>,
        slack_penalty_basis: Option<String>,
        n_terms: usize,
        hydro_availability_mw: f64,
    }
}

golden_row! {
    /// One `(constraint, stage, block)` cell over the simulated scenarios: how many
    /// scenarios it covers, in how many it is active, and the extremes of its bounds
    /// (null when no scenario carries one).
    ActivationSummary, "activation.csv", tolerant: false,
    key { constraint_id: i32, stage_id: i32, block_id: Option<i32> },
    values {
        n_scenarios: usize,
        n_active: usize,
        min_bound_lower: Option<f64>,
        max_bound_lower: Option<f64>,
        min_bound_upper: Option<f64>,
        max_bound_upper: Option<f64>,
    }
}

golden_row! {
    /// One `(constraint, stage)` pair over the simulated scenarios and blocks: the sum and
    /// the largest magnitude of the slack, and the sum of its cost.
    ViolationSummary, "violation.csv", tolerant: true,
    key { constraint_id: i32, stage_id: i32 },
    values {
        sum_slack_value: f64,
        max_abs_slack_value: f64,
        sum_slack_cost: f64,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GoldenSnapshot {
    pub(crate) resolved_bounds: Vec<BoundsSummary>,
    pub(crate) activation: Vec<ActivationSummary>,
    pub(crate) violation: Vec<ViolationSummary>,
}

fn group<T, K: Ord>(rows: &[T], key: impl Fn(&T) -> K) -> BTreeMap<K, Vec<&T>> {
    let mut groups: BTreeMap<K, Vec<&T>> = BTreeMap::new();
    for row in rows {
        groups.entry(key(row)).or_default().push(row);
    }
    groups
}

fn is_availability_read(row: &EchoRow) -> bool {
    row.variable_kind.as_deref() == Some("hydro_availability")
}

fn bounds_summaries(echo: &[EchoRow]) -> Vec<BoundsSummary> {
    group(echo, |r| (r.constraint_id, r.stage_id, r.block_id))
        .into_iter()
        .map(|((constraint_id, stage_id, block_id), rows)| {
            let first = rows[0];
            // Not `sum()`: the sum of no capacity must be +0.0, not -0.0.
            let mut hydro_availability_mw = 0.0;
            for value in rows
                .iter()
                .filter(|r| is_availability_read(r))
                .filter_map(|r| r.value)
            {
                hydro_availability_mw += value;
            }
            BoundsSummary {
                constraint_id,
                stage_id,
                block_id,
                bound_lower: first.bound_lower,
                bound_upper: first.bound_upper,
                lower_per_solve: first.lower_per_solve,
                upper_per_solve: first.upper_per_solve,
                has_active_when: first.has_active_when,
                slack_enabled: first.slack_enabled,
                slack_penalty: first.slack_penalty,
                slack_penalty_basis: first.slack_penalty_basis.clone(),
                n_terms: rows
                    .iter()
                    .filter(|r| r.variable_kind.is_some() && !is_availability_read(r))
                    .count(),
                hydro_availability_mw,
            }
        })
        .collect()
}

fn activation_summaries(activation: &[ActivationRow]) -> Vec<ActivationSummary> {
    group(activation, |r| (r.constraint_id, r.stage_id, r.block_id))
        .into_iter()
        .map(|((constraint_id, stage_id, block_id), rows)| {
            let extreme = |bound: fn(&ActivationRow) -> Option<f64>, pick: fn(f64, f64) -> f64| {
                rows.iter().filter_map(|r| bound(r)).reduce(pick)
            };
            ActivationSummary {
                constraint_id,
                stage_id,
                block_id,
                n_scenarios: rows.len(),
                n_active: rows.iter().filter(|r| r.active).count(),
                min_bound_lower: extreme(|r| r.bound_lower, f64::min),
                max_bound_lower: extreme(|r| r.bound_lower, f64::max),
                min_bound_upper: extreme(|r| r.bound_upper, f64::min),
                max_bound_upper: extreme(|r| r.bound_upper, f64::max),
            }
        })
        .collect()
}

fn violation_summaries(violations: &[ViolationRow]) -> Vec<ViolationSummary> {
    group(violations, |r| (r.constraint_id, r.stage_id))
        .into_iter()
        .map(|((constraint_id, stage_id), rows)| {
            let (mut sum_slack_value, mut max_abs_slack_value, mut sum_slack_cost) =
                (0.0, 0.0, 0.0);
            for row in rows {
                sum_slack_value += row.slack_value;
                max_abs_slack_value = f64::max(max_abs_slack_value, row.slack_value.abs());
                sum_slack_cost += row.slack_cost;
            }
            ViolationSummary {
                constraint_id,
                stage_id,
                sum_slack_value,
                max_abs_slack_value,
                sum_slack_cost,
            }
        })
        .collect()
}

/// Summarises the generic-constraint outputs of a run; every table is sorted by its key and
/// every aggregate is summed in that order, so the snapshot of one run is deterministic.
pub(crate) fn snapshot(outputs: &ConstraintOutputs) -> GoldenSnapshot {
    GoldenSnapshot {
        resolved_bounds: bounds_summaries(&outputs.echo),
        activation: activation_summaries(&outputs.activation),
        violation: violation_summaries(&outputs.violations),
    }
}

fn write_table<R: GoldenRow>(dir: &Path, rows: &[R]) {
    let mut text = format!("{}\n", R::HEADER.join(","));
    for row in rows {
        text.push_str(&row.fields().join(","));
        text.push('\n');
    }
    fs::write(dir.join(R::FILE), text)
        .unwrap_or_else(|e| panic!("cannot write {} under {}: {e}", R::FILE, dir.display()));
}

fn write_golden(dir: &Path, s: &GoldenSnapshot) {
    fs::create_dir_all(dir).unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
    write_table(dir, &s.resolved_bounds);
    write_table(dir, &s.activation);
    write_table(dir, &s.violation);
}

fn read_table<R: GoldenRow>(dir: &Path) -> Result<Vec<R>, String> {
    let path = dir.join(R::FILE);
    let text =
        fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut lines = text.lines();
    let header = R::HEADER.join(",");
    if lines.next() != Some(header.as_str()) {
        return Err(format!("{} must start with the header `{header}`", R::FILE));
    }
    let rows = lines
        .enumerate()
        .map(|(i, line)| R::parse(line).map_err(|e| format!("{} line {}: {e}", R::FILE, i + 2)))
        .collect::<Result<Vec<R>, String>>()?;
    if !rows.windows(2).all(|w| w[0].key() < w[1].key()) {
        return Err(format!("{} rows must ascend strictly by key", R::FILE));
    }
    Ok(rows)
}

fn read_golden(dir: &Path) -> Result<GoldenSnapshot, String> {
    Ok(GoldenSnapshot {
        resolved_bounds: read_table(dir)?,
        activation: read_table(dir)?,
        violation: read_table(dir)?,
    })
}

fn cells_agree<R: GoldenRow>(expected: &str, actual: &str) -> bool {
    if !R::TOLERANT {
        return expected == actual;
    }
    let number = |cell: &str| cell.parse::<f64>().expect("a tolerant cell is a number");
    let (expected, actual) = (number(expected), number(actual));
    (expected - actual).abs()
        <= (REL_TOLERANCE * expected.abs().max(actual.abs())).max(ABS_TOLERANCE)
}

fn compare_table<R: GoldenRow>(expected: &[R], actual: &[R], diffs: &mut Vec<String>) {
    let by_key = |rows: &[R]| -> BTreeMap<Key, Vec<String>> {
        rows.iter().map(|r| (r.key(), r.fields())).collect()
    };
    let (expected, actual) = (by_key(expected), by_key(actual));
    for (key, want) in &expected {
        let Some(got) = actual.get(key) else {
            diffs.push(format!("{}: key {key:?} is missing from the run", R::FILE));
            continue;
        };
        for ((column, want), got) in R::HEADER.iter().zip(want).zip(got).skip(key.len()) {
            if !cells_agree::<R>(want, got) {
                diffs.push(format!(
                    "{}: key {key:?}, column {column}: expected {want}, got {got}",
                    R::FILE
                ));
            }
        }
    }
    for key in actual.keys().filter(|k| !expected.contains_key(*k)) {
        diffs.push(format!("{}: key {key:?} is not in the baseline", R::FILE));
    }
}

/// One message per key missing from or extra in `actual` and per diverging cell. The
/// resolved bounds and the activation compare exactly; the violation compares within a
/// relative 1e-6 with an absolute floor of 1e-6.
fn compare(expected: &GoldenSnapshot, actual: &GoldenSnapshot) -> Vec<String> {
    let mut diffs = Vec::new();
    compare_table(
        &expected.resolved_bounds,
        &actual.resolved_bounds,
        &mut diffs,
    );
    compare_table(&expected.activation, &actual.activation, &mut diffs);
    compare_table(&expected.violation, &actual.violation, &mut diffs);
    diffs
}

fn power_counts(s: &GoldenSnapshot) -> [(&'static str, usize); 3] {
    [
        (
            "cell inactive in every scenario",
            s.activation.iter().filter(|c| c.n_active == 0).count(),
        ),
        (
            "cell active in every scenario",
            s.activation
                .iter()
                .filter(|c| c.n_scenarios > 0 && c.n_active == c.n_scenarios)
                .count(),
        ),
        (
            "constraint reading a positive `hydro_availability`",
            s.resolved_bounds
                .iter()
                .filter(|b| b.hydro_availability_mw > 0.0)
                .count(),
        ),
    ]
}

fn assert_has_power(s: &GoldenSnapshot) {
    let missing: Vec<&str> = power_counts(s)
        .into_iter()
        .filter(|&(_, n)| n == 0)
        .map(|(what, _)| what)
        .collect();
    assert!(
        missing.is_empty(),
        "golden has no power: the run has no {}",
        missing.join(" and no ")
    );
}

/// characterization: the baseline under `tests/fixtures/scenario_constraint_case_golden` is
/// recorded from a run a human reviewed (the `scenario_constraint_case_golden_regen` test writes
/// it), so a match proves the outputs did not move, not that they are right. The companion
/// invariants are `assert_constraint_run_invariants` and the checks of the converted case test,
/// which every run compared here has passed; the non-degeneracy guard is `assert_has_power`,
/// which refuses a run with no cell inactive in every scenario, no cell active in every
/// scenario, or no positive `hydro_availability`, since drift in the activation or the
/// availability would go unseen. A missing or malformed baseline and a divergence panic; the
/// baseline is never rewritten here.
pub(crate) fn assert_matches_baseline(outputs: &ConstraintOutputs) {
    let actual = snapshot(outputs);
    assert_has_power(&actual);
    let expected = read_golden(Path::new(BASELINE_DIR)).unwrap_or_else(|e| {
        panic!(
            "the scenario constraint case golden baseline is unusable: {e}; review the run, then \
             write it with the ignored `scenario_constraint_case_golden_regen` test \
             (NOVOMODELO_CONVERTED_CASE set)"
        )
    });
    let diffs = compare(&expected, &actual);
    assert!(
        diffs.is_empty(),
        "the run diverges from the scenario constraint case golden baseline in {} place(s), \
         first {} shown:\n  {}\nif the change is intended, review it and rewrite the baseline \
         with the ignored `scenario_constraint_case_golden_regen` test",
        diffs.len(),
        diffs.len().min(MAX_REPORTED),
        diffs
            .iter()
            .take(MAX_REPORTED)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

#[test]
#[ignore = "rewrites the committed scenario constraint case golden — run explicitly after reviewing the change"]
fn scenario_constraint_case_golden_regen() {
    let case = PathBuf::from(
        std::env::var("NOVOMODELO_CONVERTED_CASE")
            .expect("NOVOMODELO_CONVERTED_CASE: set it to the directory of a converted case"),
    );
    let out = TempDir::new().unwrap();
    run_case(&case, out.path());

    let outputs = read_constraint_outputs(out.path());
    assert_constraint_run_invariants(out.path(), &outputs);
    let snapshot = snapshot(&outputs);
    assert_has_power(&snapshot);
    write_golden(Path::new(BASELINE_DIR), &snapshot);

    eprintln!(
        "scenario constraint case golden written to {BASELINE_DIR}: resolved_bounds.csv={} \
         rows, activation.csv={} rows, violation.csv={} rows; {:?}",
        snapshot.resolved_bounds.len(),
        snapshot.activation.len(),
        snapshot.violation.len(),
        power_counts(&snapshot),
    );
}

fn fixture_snapshot() -> GoldenSnapshot {
    let case = TempDir::new().unwrap();
    write_scenario_constraint_case(case.path());
    let out = TempDir::new().unwrap();
    run_case(case.path(), out.path());
    let outputs = read_constraint_outputs(out.path());
    assert_constraint_run_invariants(out.path(), &outputs);
    snapshot(&outputs)
}

#[test]
fn the_golden_snapshot_is_reproducible_and_detects_drift() {
    let base = fixture_snapshot();
    let second = fixture_snapshot();
    assert_has_power(&base);
    assert_eq!(
        compare(&base, &second),
        Vec::<String>::new(),
        "two runs of the same case must agree"
    );
    assert_eq!(
        base, second,
        "two runs of the same case must agree bit for bit"
    );

    let dir = TempDir::new().unwrap();
    let golden = dir.path().join("nested/golden");
    write_golden(&golden, &base);
    assert_eq!(read_golden(&golden), Ok(base.clone()));
    assert!(
        read_golden(&dir.path().join("absent")).is_err(),
        "a missing baseline must be an error"
    );
    fs::write(golden.join("violation.csv"), "constraint_id,stage_id\n").unwrap();
    assert!(
        read_golden(&golden).is_err(),
        "a baseline with the wrong header must be an error"
    );

    let violated = base
        .violation
        .iter()
        .position(|v| v.constraint_id == 701)
        .expect("the fixture violates constraint 701");
    assert!(
        base.violation[violated].sum_slack_value.abs() > 1.0,
        "the drift below needs a slack well above the absolute floor"
    );
    let violation_drift = |relative: f64| {
        let mut drifted = base.clone();
        drifted.violation[violated].sum_slack_value *= 1.0 + relative;
        compare(&base, &drifted)
    };
    assert_eq!(violation_drift(2e-6).len(), 1);
    assert!(violation_drift(1e-7).is_empty());

    let mut inactive_changed = base.clone();
    inactive_changed.activation[0].n_active += 1;
    assert_eq!(compare(&base, &inactive_changed).len(), 1);

    let fixed_upper = base
        .resolved_bounds
        .iter()
        .position(|b| b.bound_upper.is_some())
        .expect("the fixture has a constraint with a fixed upper bound");
    let mut one_ulp = base.clone();
    let bound = one_ulp.resolved_bounds[fixed_upper]
        .bound_upper
        .as_mut()
        .unwrap();
    *bound = f64::from_bits(bound.to_bits() + 1);
    assert_eq!(
        compare(&base, &one_ulp).len(),
        1,
        "a data cell compares exactly, not within the solver tolerance"
    );

    let mut dropped = base.clone();
    dropped.violation.remove(0);
    assert_eq!(compare(&base, &dropped).len(), 1);
    assert_eq!(compare(&dropped, &base).len(), 1);

    let degenerate: [fn(&mut GoldenSnapshot); 3] = [
        |s| {
            for cell in &mut s.activation {
                cell.n_active = cell.n_scenarios;
            }
        },
        |s| {
            for cell in &mut s.activation {
                cell.n_active = 0;
            }
        },
        |s| {
            for row in &mut s.resolved_bounds {
                row.hydro_availability_mw = 0.0;
            }
        },
    ];
    for (guarded, degrade) in degenerate.into_iter().enumerate() {
        let mut degraded = base.clone();
        degrade(&mut degraded);
        assert!(
            power_counts(&degraded)
                .iter()
                .enumerate()
                .all(|(i, &(_, n))| (n == 0) == (i == guarded)),
            "degenerating the fixture must zero exactly power count {guarded}: {:?}",
            power_counts(&degraded)
        );
    }
}
