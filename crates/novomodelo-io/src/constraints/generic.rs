//! Parsing for `constraints/generic_constraints.json` — user-defined linear constraints.
//!
//! [`parse_generic_constraints`] reads `constraints/generic_constraints.json` and
//! returns a sorted `Vec<GenericConstraint>`.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "constraints": [
//!     {
//!       "id": 0,
//!       "name": "min_southeast_hydro",
//!       "description": "...",
//!       "expression": "hydro_generation(0) + hydro_generation(1) <= @generation_cap",
//!       "active_when": "bus_load_value(0) > 82000",
//!       "slack": { "enabled": true, "penalty": 5000.0, "penalty_basis": "per_hour" }
//!     }
//!   ]
//! }
//! ```
//!
//! Every `expression` string, in a constraint or in a named expression of the optional
//! top-level `expressions` array, is parsed and lowered by the crate's expression engine
//! (`crate::expression`). A relational operator is authored inline; the engine moves every
//! variable term to the left and assigns the remainder to the bound the operator targets.
//!
//! An expression may read scenario data: a data term such as `bus_load_value(1)` stands for a
//! scalar-parameter id allocated above every id of `name_to_id`, and an `if(cond, a, b)` lowers
//! to a selector, a further id that stands for the branch its data-only condition picks. A named
//! expression that reads no LP variable is data, written in place wherever it is referenced.
//! `hydro_availability(h)` is the plant's capacity, read as data through a scalar-parameter id of
//! its own, allocated as a data term's is: the term leaves the left side, and its coefficient
//! times that id is subtracted from each bound, or from the endpoint shift.
//!
//! An expression without an operator is bounded by the bounds table. Its constant and data part,
//! negated, is the constraint's endpoint shift in the [`ScenarioLayer`]: it moves each endpoint
//! the table gives and never creates one.
//!
//! A constraint's optional `active_when` is a condition over data only, lowered by the same
//! engine without `if()`. It sits beside the shift in the constraint's layer entry, and where it
//! is false the constraint does not hold. The layer also lists the data atoms, capacities and
//! selectors by id; a file that reads no data and no capacity and has no `if()`, no
//! `active_when` and no shift leaves it empty.
//!
//! Deferred validations (not performed here):
//!
//! - Entity ID existence in entity registries — Layer 3.
//! - Block ID validity for the referenced stage — Layer 3/5.

use novomodelo_core::{
    ConstraintExpression, EntityId, GenericConstraint, LinearTerm, PenaltyBasis, SlackConfig,
    is_exact_zero,
};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use super::reference_cycles::detect_cycles;
use super::scenario::{
    AtomScan, ConstraintScenario, Ids, ScenarioLayer, allocate_ids, fold_availability,
};
use crate::LoadError;
use crate::expression::Diagnostic;
use crate::expression::catalog::{DataAtom, LineBusPairIndex};
use crate::expression::lower::{
    NamedSource, NamedTable, Scope, lower_activation_condition, lower_relation_from,
};
use crate::expression::parser::{parse, parse_condition};

/// The named expressions each constraint reached, by constraint id; a constraint that reached
/// none has no entry.
pub(crate) type NamedSourcesByConstraint = BTreeMap<EntityId, Vec<NamedSource>>;

#[derive(Default)]
pub(crate) struct LoadedGenericConstraints {
    pub(crate) constraints: Vec<GenericConstraint>,
    pub(crate) named_sources: NamedSourcesByConstraint,
    pub(crate) scenario: ScenarioLayer,
}

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Root object of `generic_constraints.json`.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct RawGenericConstraintsFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Array of constraint entries.
    constraints: Vec<RawConstraint>,

    /// Named linear-expression declarations sharing the `@name` namespace with
    /// scalar parameters. Absent ⇒ empty ⇒ no named expressions.
    #[serde(default)]
    expressions: Vec<RawNamedExpression>,
}

/// A single constraint entry.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
struct RawConstraint {
    /// Constraint identifier. Must be unique within the file.
    id: i32,

    /// Short name used in reports and log output.
    name: String,

    /// Optional human-readable description.
    description: Option<String>,

    /// Linear expression over LP variables, with numbers and `@parameters` combined by `+`, `-`,
    /// `*`, `/` and parentheses, e.g. `"2.5 * thermal_generation(5) - @k * hydro_generation(3) <= @cap"`.
    /// To constrain an anticipated thermal's commitment, use `"anticipated_decision(5)"` (stage-level scalar, no block index).
    /// To constrain one bus of a plant split across several buses, use `"hydro_turbined(5, bus=2)"`
    /// or `"hydro_generation(5, bus=2)"` — the only two variables accepting a `bus=` selector.
    /// For a stage volume in hm³ use `"hydro_outflow_volume(5)"` (turbined plus spilled),
    /// `"hydro_diversion_volume(5)"` or `"pumping_volume(2)"`: each block's m³/s flow times its
    /// duration, summed over the stage, so a coefficient multiplies hm³. A constraint with one
    /// takes only stage volumes, `hydro_storage`, and `hydro_storage_*` or
    /// `hydro_useful_volume_*` without a block argument; price its slack `"per_unit"`.
    /// Besides numbers and `@parameters` it may read scenario data: `bus_load_value(n)` is the
    /// load of bus `n` in the scenario being solved, `bus_load_value(n, k)` that load at block
    /// `k`, and the load of several buses is their written sum, e.g.
    /// `bus_load_value(0) + bus_load_value(1)`. It may choose a value with `if(cond, a, b)`:
    /// `cond` reads data only, as `active_when` does, and the branches read the same LP
    /// variables, differing only in coefficients and constants.
    /// An optional inline relational operator (`<=`, `>=`, `==`) moves every variable term to
    /// the left and assigns the remainder to the bound it targets. Without one,
    /// `constraints/generic_constraint_bounds.parquet` bounds the whole expression: its constant
    /// and data part shifts each endpoint the table gives and creates none.
    ///
    /// `hydro_availability(h)` is data, not an LP variable: at each stage and block it is the sum
    /// of the maximum generation of plant `h`'s unit groups. A group's maximum is the one the
    /// study resolves for the block or the stage from
    /// `constraints/hydro_unit_group_bounds.parquet`, where it resolves one, and the group's
    /// declared maximum elsewhere. The value is 0 at a stage where the plant does not operate.
    /// The term reads the capacity of the row's block; `hydro_availability(h, k)` reads that of
    /// block `k`. The term leaves the left side: its coefficient times that value moves into the
    /// bounds. A condition does not read it.
    expression: String,

    /// Optional condition under which the constraint holds, tested per stage, block and scenario;
    /// where it is false the constraint does not hold. It reads data only: the scenario bus loads
    /// `bus_load_value(n)`, `@parameters` and named expressions that read no LP variable,
    /// compared with `<`, `<=`, `>`, `>=` or `==` and joined by `&` and `|`, e.g.
    /// `"82000 < bus_load_value(0) + bus_load_value(1) & bus_load_value(0) + bus_load_value(1) <= 95000"`.
    /// It reads no LP variable, no `hydro_availability` and no `if()`.
    active_when: Option<String>,

    /// Slack variable configuration.
    slack: RawSlackConfig,
}

/// The slack configuration.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
struct RawSlackConfig {
    /// Whether a slack variable is allowed.
    enabled: bool,

    /// Penalty per unit of violation. Required and > 0.0 when `enabled` is `true`;
    /// absent otherwise.
    penalty: Option<f64>,

    /// How `penalty` is charged: `"per_hour"` multiplies it by the hours the row
    /// spans (rows in MW or m³/s); `"per_unit"` does not (rows in `MWh` or hm³).
    /// Required when `enabled` is `true`; absent otherwise.
    penalty_basis: Option<PenaltyBasis>,
}

/// A named linear-expression declaration.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
struct RawNamedExpression {
    /// Name bound to the expression; unique in the file and distinct from every scalar-parameter name.
    name: String,

    /// Linear expression under the same arithmetic as a constraint's `expression`, scenario data
    /// and `if()` included, without a relational operator. One that reads no LP variable is data:
    /// `active_when` may read it, and in a constraint it joins the constant and data part.
    expression: String,

    /// Optional human-readable description.
    #[serde(rename = "description")]
    _description: Option<String>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load and validate `constraints/generic_constraints.json` from `path`, returning
/// constraints sorted by `id` ascending (declaration-order invariance).
///
/// # Errors
///
/// | Condition                                      | Error variant              |
/// | ---------------------------------------------- | -------------------------- |
/// | File not found / read failure                  | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field  | [`LoadError::ParseError`]  |
/// | Duplicate `id` within the constraints array    | [`LoadError::SchemaError`] |
/// | `slack.enabled = true` with absent or <= 0 penalty, or absent `penalty_basis` | [`LoadError::SchemaError`] |
/// | `slack.enabled = false` with `penalty` or `penalty_basis` set | [`LoadError::SchemaError`] |
/// | Named expression: duplicate or parameter-colliding name | [`LoadError::SchemaError`] |
/// | No scalar-parameter id left above `name_to_id` for the data terms and selectors | [`LoadError::SchemaError`] |
/// | Named expression: body rejected (syntax error, unknown name, not linear, `if()` whose condition reads an LP variable or whose branches read different ones) | [`LoadError::SchemaError`] |
/// | Named expression: reference cycle              | [`LoadError::SchemaError`] |
/// | Constraint expression rejected (syntax error, unknown variable or `@name`, not linear, `if()` as for a named expression) | [`LoadError::SchemaError`] |
/// | Constraint `active_when` rejected (syntax error, unknown name, reads an LP variable, `hydro_availability` or `if()`) | [`LoadError::SchemaError`] |
///
/// Faults are reported in this order, each in file order: ids and slacks, named-expression
/// names, id room, named-expression bodies, reference cycles, constraint expressions, each
/// followed by its `active_when`.
///
/// # Parameters
///
/// `name_to_id` maps parameter definition names to their [`EntityId`]. Pass
/// `&HashMap::new()` when no parameters have been loaded; expressions that
/// contain `@name` tokens will then fail with a schema error. The case load
/// builds the map from the loaded scalar parameters.
///
/// `line_index` resolves the `line_exchange(source_bus=X, target_bus=Y)` addressing
/// form to a line id; pass [`LineBusPairIndex::default`] when no line topology is
/// available and any pair form then errors as an unmatched pair.
///
/// # Examples
///
/// ```no_run
/// use novomodelo_io::constraints::{LineBusPairIndex, parse_generic_constraints};
/// use std::collections::HashMap;
/// use std::path::Path;
///
/// let constraints = parse_generic_constraints(
///     Path::new("case/constraints/generic_constraints.json"),
///     &HashMap::new(),
///     &LineBusPairIndex::default(),
/// ).expect("valid generic constraints file");
/// println!("loaded {} generic constraints", constraints.len());
/// ```
#[allow(clippy::implicit_hasher)]
pub fn parse_generic_constraints(
    path: &Path,
    name_to_id: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
) -> Result<Vec<GenericConstraint>, LoadError> {
    parse_generic_constraints_with_provenance(path, name_to_id, line_index)
        .map(|loaded| loaded.constraints)
}

/// [`parse_generic_constraints`] that also returns, for each constraint that reached a named
/// expression, the named expressions it reached, and the scenario layer.
pub(crate) fn parse_generic_constraints_with_provenance(
    path: &Path,
    name_to_id: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
) -> Result<LoadedGenericConstraints, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawGenericConstraintsFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw(&raw, path)?;
    check_named_expression_names(&raw.expressions, path, name_to_id)?;

    let AtomScan {
        atoms, capacities, ..
    } = scan_atoms(&raw, line_index);
    let ids =
        allocate_ids(name_to_id, atoms, capacities).ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "constraints".to_string(),
            message:
                "the scalar-parameter ids leave no room for the data terms and selectors of the \
                 constraints"
                    .to_string(),
        })?;

    let table =
        declare_named_expressions(&raw.expressions, path, name_to_id, line_index, &ids.data)?;
    detect_cycles(&table.reference_graph(), "named-expression").map_err(|message| {
        LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "expressions".to_string(),
            message,
        }
    })?;

    convert(raw, path, name_to_id, &table, line_index, ids)
}

/// The data atoms and plant capacities read by every named-expression body, constraint
/// expression and `active_when` that parses; one that does not parse is reported when it is
/// lowered.
fn scan_atoms<'a>(
    raw: &RawGenericConstraintsFile,
    line_index: &'a LineBusPairIndex,
) -> AtomScan<'a> {
    let mut scan = AtomScan::new(line_index);
    let texts = raw.expressions.iter().map(|entry| &entry.expression).chain(
        raw.constraints
            .iter()
            .map(|constraint| &constraint.expression),
    );
    for relation in texts.filter_map(|text| parse(text).ok()) {
        scan.visit_expr(&relation.lhs);
        if let Some((_, rhs)) = &relation.rhs {
            scan.visit_expr(rhs);
        }
    }
    let conditions = raw
        .constraints
        .iter()
        .filter_map(|constraint| constraint.active_when.as_deref());
    for condition in conditions.filter_map(|text| parse_condition(text).ok()) {
        scan.visit_condition(&condition);
    }
    scan
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Validate the invariants on the raw deserialized constraint data that do not
/// involve an expression.
fn validate_raw(raw: &RawGenericConstraintsFile, path: &Path) -> Result<(), LoadError> {
    validate_no_duplicate_ids(&raw.constraints, path)?;
    for (i, constraint) in raw.constraints.iter().enumerate() {
        resolve_slack(&constraint.slack, i, path)?;
    }
    Ok(())
}

/// Check that no two constraints share the same `id`.
fn validate_no_duplicate_ids(constraints: &[RawConstraint], path: &Path) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, constraint) in constraints.iter().enumerate() {
        if !seen.insert(constraint.id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("constraints[{i}].id"),
                message: format!("duplicate id {} in constraints array", constraint.id),
            });
        }
    }
    Ok(())
}

/// Resolve a raw slack block: `enabled` requires a positive `penalty` and a
/// `penalty_basis`; a disabled slack carries neither.
fn resolve_slack(
    slack: &RawSlackConfig,
    constraint_index: usize,
    path: &Path,
) -> Result<Option<SlackConfig>, LoadError> {
    let error = |key: &str, message: String| LoadError::SchemaError {
        path: path.to_path_buf(),
        field: format!("constraints[{constraint_index}].slack.{key}"),
        message,
    };
    if !slack.enabled {
        if slack.penalty.is_some() {
            return Err(error(
                "penalty",
                "slack.penalty is set but slack.enabled is false".to_string(),
            ));
        }
        if slack.penalty_basis.is_some() {
            return Err(error(
                "penalty_basis",
                "slack.penalty_basis is set but slack.enabled is false".to_string(),
            ));
        }
        return Ok(None);
    }
    let penalty = match slack.penalty {
        None => {
            return Err(error(
                "penalty",
                "slack.enabled is true but slack.penalty is absent".to_string(),
            ));
        }
        Some(p) if p <= 0.0 => {
            return Err(error(
                "penalty",
                format!("slack.penalty must be > 0.0 when enabled, got {p}"),
            ));
        }
        Some(p) => p,
    };
    let Some(penalty_basis) = slack.penalty_basis else {
        return Err(error(
            "penalty_basis",
            "slack.enabled is true but slack.penalty_basis is absent: use \"per_hour\" \
             for a row in MW or m³/s, \"per_unit\" for a row in MWh or hm³"
                .to_string(),
        ));
    };
    Ok(Some(SlackConfig {
        penalty,
        penalty_basis,
    }))
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert the validated raw data to `Vec<GenericConstraint>`, sorted by `id`, the named
/// expressions each constraint reached, keyed by constraint `id`, and the scenario layer.
fn convert(
    raw: RawGenericConstraintsFile,
    path: &Path,
    name_to_id: &HashMap<String, EntityId>,
    table: &NamedTable,
    line_index: &LineBusPairIndex,
    ids: Ids,
) -> Result<LoadedGenericConstraints, LoadError> {
    let Ids {
        data,
        capacities,
        first_selector,
    } = ids;
    let scope = Scope {
        params: name_to_id,
        line_index,
        named: table,
        data_ids: Some(&data),
    };
    let mut result = Vec::with_capacity(raw.constraints.len());
    let mut named_sources = BTreeMap::new();
    let mut selectors = Vec::new();
    let mut entries = Vec::new();
    let mut next_selector = first_selector;

    for (i, c) in raw.constraints.into_iter().enumerate() {
        let error = |key: &str, message: String| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("constraints[{i}].{key}"),
            message,
        };
        let equated = parse(&c.expression)
            .map_err(|diagnostic| error("expression", diagnostic.render(&c.expression)))?
            .rhs
            .is_none();
        // An expression without an operator lowers as `expression == 0`, which leaves its
        // constant and data part, negated, as the remainder on both endpoints.
        let mut lowered = if equated {
            lower_relation_from(
                &format!("{} == 0", c.expression),
                &scope,
                EntityId(next_selector),
            )
            .map_err(|diagnostic| within(diagnostic, c.expression.len()))
        } else {
            lower_relation_from(&c.expression, &scope, EntityId(next_selector))
        }
        .map_err(|diagnostic| error("expression", diagnostic.render(&c.expression)))?;
        next_selector = i32::try_from(lowered.selectors.len())
            .ok()
            .and_then(|count| next_selector.checked_add(count))
            .ok_or_else(|| {
                error(
                    "expression",
                    "the scalar-parameter ids leave no room for the selectors of the constraints"
                        .to_string(),
                )
            })?;
        let mut expression = ConstraintExpression {
            terms: lowered
                .terms
                .into_iter()
                .map(|(variable, coefficient)| LinearTerm {
                    coefficient,
                    variable,
                })
                .collect(),
        };
        expression.canonicalize();
        expression.terms = fold_availability(
            expression.terms,
            &mut lowered.lower,
            &mut lowered.upper,
            &capacities,
        )
        .map_err(|message| error("expression", message))?;
        let active_when = c
            .active_when
            .as_deref()
            .map(|text| {
                parse_condition(text)
                    .and_then(|condition| lower_activation_condition(&condition, &scope))
                    .map_err(|diagnostic| error("active_when", diagnostic.render(text)))
            })
            .transpose()?;

        let slack = resolve_slack(&c.slack, i, path)?;

        let id = EntityId::from(c.id);
        if !lowered.named_sources.is_empty() {
            named_sources.insert(id, lowered.named_sources);
        }
        selectors.extend(lowered.selectors);
        // The shift moves only the endpoints the bounds table gives; as a bound it would create
        // one.
        let (endpoint_shift, bound_lower, bound_upper) = if equated {
            let shift = lowered
                .lower
                .filter(|remainder| !remainder.as_literal().is_some_and(is_exact_zero));
            (shift, None, None)
        } else {
            (None, lowered.lower, lowered.upper)
        };
        if active_when.is_some() || endpoint_shift.is_some() {
            entries.push(ConstraintScenario {
                constraint_id: id,
                active_when,
                endpoint_shift,
            });
        }
        result.push(GenericConstraint {
            id,
            name: c.name,
            description: c.description,
            expression,
            slack,
            bound_lower,
            bound_upper,
        });
    }

    result.sort_by_key(|gc| gc.id.0);

    Ok(LoadedGenericConstraints {
        constraints: result,
        named_sources,
        scenario: ScenarioLayer::new(data, capacities, selectors, entries),
    })
}

/// `diagnostic`, located in a source whose first `len` bytes are the text it is rendered
/// against, with its span clamped to that text's end.
fn within(mut diagnostic: Diagnostic, len: usize) -> Diagnostic {
    diagnostic.span = diagnostic.span.start.min(len)..diagnostic.span.end.min(len);
    diagnostic
}

// ── Named-expression declarations ───────────────────────────────────────────

/// Check that no two `expressions` declarations share a name and that none takes a
/// scalar-parameter name.
///
/// # Errors
///
/// [`LoadError::SchemaError`] with `field = "expressions[i].name"` for the first duplicate or
/// parameter-colliding name, in file order.
fn check_named_expression_names(
    entries: &[RawNamedExpression],
    path: &Path,
    name_to_id: &HashMap<String, EntityId>,
) -> Result<(), LoadError> {
    let mut seen: HashSet<&str> = HashSet::new();

    for (i, entry) in entries.iter().enumerate() {
        if !seen.insert(entry.name.as_str()) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("expressions[{i}].name"),
                message: format!(
                    "duplicate named expression \"{}\" in expressions array",
                    entry.name
                ),
            });
        }

        if name_to_id.contains_key(&entry.name) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("expressions[{i}].name"),
                message: format!(
                    "name \"{}\" is declared as both a scalar parameter and a named expression; the \"@name\" namespace is shared",
                    entry.name
                ),
            });
        }
    }
    Ok(())
}

/// Parse, type and lower each `expressions` body into a [`NamedTable`], resolving its data
/// terms through `data_ids`. Reference cycles are checked over the table afterwards.
///
/// # Errors
///
/// [`LoadError::SchemaError`] with `field = "expressions[i].expression"` for the first body,
/// in file order, the engine rejects.
fn declare_named_expressions(
    entries: &[RawNamedExpression],
    path: &Path,
    name_to_id: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
    data_ids: &BTreeMap<DataAtom, EntityId>,
) -> Result<NamedTable, LoadError> {
    let declarations: Vec<(&str, &str)> = entries
        .iter()
        .map(|entry| (entry.name.as_str(), entry.expression.as_str()))
        .collect();
    NamedTable::declare(&declarations, name_to_id, line_index, data_ids).map_err(
        |(i, diagnostic)| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("expressions[{i}].expression"),
            message: diagnostic.render(&entries[i].expression),
        },
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use crate::constraints::scenario::{
        AvailabilityAtom, CompareOp, ConditionalScalar, ResolvedCondition,
    };
    use crate::expression::catalog::build_line_bus_pair_index;
    use crate::expression::lower::lower_relation;
    use crate::test_support::{make_line, write_json};
    use crate::{LoadedCase, load_case_with_artifacts};
    use novomodelo_core::{ScalarExpr, VariableRef};
    use std::fmt::Write as _;

    fn lit(term: &LinearTerm) -> f64 {
        term.coefficient
            .as_literal()
            .expect("expected a literal coefficient")
    }

    fn eval(expr: &ScalarExpr, leaves: &[(i32, f64)]) -> f64 {
        expr.evaluate(|id| {
            let &(_, value) = leaves
                .iter()
                .find(|(key, _)| *key == id.0)
                .unwrap_or_else(|| panic!("no leaf value for parameter {id:?}"));
            value
        })
    }

    fn params_of(expr: &ScalarExpr) -> Vec<EntityId> {
        expr.params().collect()
    }

    fn one_param_table() -> HashMap<String, EntityId> {
        let mut m = HashMap::new();
        m.insert("rho_eq".to_string(), EntityId(7));
        m.insert("rho".to_string(), EntityId(7));
        m
    }

    const VALID_JSON: &str = r#"{
  "constraints": [
    {
      "id": 1,
      "name": "min_hydro",
      "expression": "hydro_generation(10) + hydro_generation(11)",
      "slack": { "enabled": false }
    },
    {
      "id": 0,
      "name": "max_thermal",
      "expression": "2.5 * thermal_generation(5) - hydro_generation(3)",
      "slack": { "enabled": true, "penalty": 5000.0, "penalty_basis": "per_unit" }
    }
  ]
}"#;

    /// `thermal_generation(5) <= 0.87 * hydro_generation(140) + 73` normalizes
    /// to the same terms as the hand-flattened LHS
    /// `thermal_generation(5) - 0.87 * hydro_generation(140)` after
    /// canonicalize; `bound_upper` carries the constant,
    /// `bound_lower` stays `None`.
    #[test]
    fn inline_constant_and_variable_rhs_normalizes() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "c0", "expression": "thermal_generation(5) <= 0.87 * hydro_generation(140) + 73", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();

        let mut hand_flattened = ConstraintExpression {
            terms: vec![
                LinearTerm::literal(
                    1.0,
                    VariableRef::ThermalGeneration {
                        thermal_id: EntityId(5),
                        block_id: None,
                    },
                ),
                LinearTerm::literal(
                    -0.87,
                    VariableRef::HydroGeneration {
                        hydro_id: EntityId(140),
                        block_id: None,
                        bus_id: None,
                    },
                ),
            ],
        };
        hand_flattened.canonicalize();

        assert_eq!(result[0].expression.terms, hand_flattened.terms);
        assert_eq!(result[0].bound_upper, Some(ScalarExpr::literal(73.0)));
        assert_eq!(result[0].bound_lower, None);
    }

    /// `line_exchange(3) >= @target` where `target` is a loaded parameter
    /// reads that parameter on the lower endpoint, leaving the upper `None`.
    #[test]
    fn inline_lower_bound_bare_param_reference() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "c0", "expression": "line_exchange(3) >= @target", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let mut tbl = HashMap::new();
        tbl.insert("target".to_string(), EntityId(99));
        let result =
            parse_generic_constraints(f.path(), &tbl, &LineBusPairIndex::default()).unwrap();
        let bound = result[0].bound_lower.as_ref().expect("lower bound present");
        assert_eq!(params_of(bound), [EntityId(99)]);
        assert_eq!(eval(bound, &[(99, 3.5)]).to_bits(), 3.5_f64.to_bits());
        assert_eq!(result[0].bound_upper, None);
    }

    /// `hydro_generation(0) - hydro_generation(0) <= 5` merges to an empty
    /// term list (the net-zero column dropped) and `bound_upper` carries
    /// the constant `5`.
    #[test]
    fn net_zero_column_drops_and_constant_bound_survives() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "c0", "expression": "hydro_generation(0) - hydro_generation(0) <= 5", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert!(result[0].expression.terms.is_empty());
        assert_eq!(result[0].bound_upper, Some(ScalarExpr::literal(5.0)));
        assert_eq!(result[0].bound_lower, None);
    }

    /// A right side mixing two decision-variable terms and a constant.
    #[test]
    fn decision_variable_rhs() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "mixed_rhs", "expression": "bus_deficit(0) <= -0.316 * bus_deficit(1) + 0.5 * bus_deficit(2) + 12352", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        let terms = &result[0].expression.terms;
        assert_eq!(terms.len(), 3);
        let find = |bus_id: i32| {
            terms
                .iter()
                .find(|t| {
                    t.variable
                        == VariableRef::BusDeficit {
                            bus_id: EntityId(bus_id),
                            block_id: None,
                        }
                })
                .unwrap_or_else(|| panic!("expected bus_deficit({bus_id}) in {terms:?}"))
        };
        assert!((lit(find(0)) - 1.0).abs() < f64::EPSILON);
        assert!((lit(find(1)) - 0.316).abs() < 1e-10);
        assert!((lit(find(2)) - (-0.5)).abs() < 1e-10);
        assert_eq!(result[0].bound_upper, Some(ScalarExpr::literal(12352.0)));
        assert_eq!(result[0].bound_lower, None);
    }

    /// Affine parameters plus a constant, with no decision variable on the right side.
    #[test]
    fn affine_param_rhs() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "affine_rhs", "expression": "bus_deficit(0) <= 11000 - 0.04 * @external_load + 0.06 * @small_plants", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let mut tbl = HashMap::new();
        tbl.insert("external_load".to_string(), EntityId(50));
        tbl.insert("small_plants".to_string(), EntityId(51));
        let result =
            parse_generic_constraints(f.path(), &tbl, &LineBusPairIndex::default()).unwrap();

        assert_eq!(result[0].expression.terms.len(), 1);
        assert!((lit(&result[0].expression.terms[0]) - 1.0).abs() < f64::EPSILON);

        let bound = result[0].bound_upper.as_ref().expect("upper bound present");
        assert_eq!(params_of(bound), [EntityId(50), EntityId(51)]);
        let (demand, generation) = (2.0_f64, 3.0_f64);
        assert_eq!(
            eval(bound, &[(50, demand), (51, generation)]).to_bits(),
            (11000.0 - 0.04 * demand + 0.06 * generation).to_bits()
        );
        assert_eq!(result[0].bound_lower, None);
    }

    #[test]
    fn pure_constant_rhs() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "constant_rhs", "expression": "bus_deficit(0) <= 73", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result[0].expression.terms.len(), 1);
        assert_eq!(result[0].bound_upper, Some(ScalarExpr::literal(73.0)));
        assert_eq!(result[0].bound_lower, None);
    }

    /// Valid 2-constraint file. First has 2 hydro_generation terms.
    #[test]
    fn test_parse_valid_two_constraints() {
        let f = write_json(VALID_JSON);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();

        // Should be 2 constraints, sorted by id ascending: 0, 1.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, EntityId(0)); // id=0 was second in JSON
        assert_eq!(result[1].id, EntityId(1)); // id=1 was first in JSON

        // The first constraint (id=1, "min_hydro") has 2 hydro_generation terms.
        // After sorting, result[1] is the "min_hydro" constraint.
        let min_hydro = &result[1];
        assert_eq!(min_hydro.expression.terms.len(), 2);
        assert!((lit(&min_hydro.expression.terms[0]) - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            min_hydro.expression.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(10),
                block_id: None,
                bus_id: None,
            }
        );
        assert_eq!(
            min_hydro.expression.terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(11),
                block_id: None,
                bus_id: None,
            }
        );
    }

    /// Authoring-order invariance: two constraints differing only in the order their
    /// same terms are written parse to element-for-element-equal `terms`, because
    /// `convert` canonicalizes every expression.
    #[test]
    fn canonical_parse_order_invariance_hydro_generation() {
        let forward = r#"{
  "constraints": [
    { "id": 0, "name": "c", "expression": "hydro_generation(0) + hydro_generation(1)", "slack": { "enabled": false } }
  ]
}"#;
        let reversed = r#"{
  "constraints": [
    { "id": 0, "name": "c", "expression": "hydro_generation(1) + hydro_generation(0)", "slack": { "enabled": false } }
  ]
}"#;
        let ff = write_json(forward);
        let rf = write_json(reversed);
        let a = parse_generic_constraints(ff.path(), &HashMap::new(), &LineBusPairIndex::default())
            .unwrap();
        let b = parse_generic_constraints(rf.path(), &HashMap::new(), &LineBusPairIndex::default())
            .unwrap();
        assert_eq!(a[0].expression.terms, b[0].expression.terms);
        // Canonical order is by entity id: hydro_generation(0) then hydro_generation(1).
        assert_eq!(
            a[0].expression.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: None,
            }
        );
        assert_eq!(
            a[0].expression.terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(1),
                block_id: None,
                bus_id: None,
            }
        );
    }

    /// Expression `"2.5 * thermal_generation(5) - hydro_generation(3)"` — after
    /// canonicalization the terms are content-ordered: hydro_generation (variant tag 5)
    /// precedes thermal_generation (tag 8), regardless of authoring order.
    #[test]
    fn test_parse_coefficient_and_subtraction_expression() {
        let f = write_json(VALID_JSON);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();

        // result[0] is id=0 "max_thermal"
        let max_thermal = &result[0];
        assert_eq!(max_thermal.expression.terms.len(), 2);
        assert!((lit(&max_thermal.expression.terms[0]) - (-1.0)).abs() < f64::EPSILON);
        assert_eq!(
            max_thermal.expression.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(3),
                block_id: None,
                bus_id: None,
            }
        );
        assert!((lit(&max_thermal.expression.terms[1]) - 2.5).abs() < 1e-10);
        assert_eq!(
            max_thermal.expression.terms[1].variable,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: None,
            }
        );
    }

    /// A JSON constraint mixing a literal term and an `@rho_eq`-scaled parameter term
    /// parses; after canonicalization the terms are content-ordered, so
    /// hydro_generation (tag 5) precedes thermal_generation (tag 8).
    #[test]
    fn test_parse_param_ref_in_json_constraint() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "mixed",
      "expression": "thermal_generation(5) - @rho_eq * hydro_generation(3)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let tbl = one_param_table();
        let result =
            parse_generic_constraints(f.path(), &tbl, &LineBusPairIndex::default()).unwrap();
        assert_eq!(result.len(), 1);
        let expr = &result[0].expression;
        assert_eq!(expr.terms.len(), 2);
        // First term (canonical): hydro_generation(3) with coefficient -@rho_eq.
        assert_eq!(params_of(&expr.terms[0].coefficient), [EntityId(7)]);
        assert_eq!(
            eval(&expr.terms[0].coefficient, &[(7, 2.5)]).to_bits(),
            (-2.5_f64).to_bits()
        );
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(3),
                block_id: None,
                bus_id: None,
            }
        );
        // Second term (canonical): thermal_generation(5) literal 1.0.
        assert!((lit(&expr.terms[1]) - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[1].variable,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: None,
            }
        );
    }

    /// An inline `<=` against a bare `@name` reads that parameter on the upper
    /// endpoint; the lower endpoint stays `None`.
    #[test]
    fn test_parse_inline_upper_bound_resolves_to_entity_id() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "demand_cap",
      "expression": "hydro_generation(3) <= @rho_eq",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let tbl = one_param_table();
        let result =
            parse_generic_constraints(f.path(), &tbl, &LineBusPairIndex::default()).unwrap();
        assert_eq!(result.len(), 1);
        let bound = result[0].bound_upper.as_ref().expect("upper bound present");
        assert_eq!(params_of(bound), [EntityId(7)]);
        assert_eq!(eval(bound, &[(7, 4.25)]).to_bits(), 4.25_f64.to_bits());
        assert_eq!(result[0].bound_lower, None);
    }

    /// An inline `>=` against a bare `@name` resolves the same way, on the lower
    /// endpoint.
    #[test]
    fn test_parse_inline_lower_bound_resolves_to_entity_id() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "demand_floor",
      "expression": "hydro_generation(3) >= @rho_eq",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let tbl = one_param_table();
        let result =
            parse_generic_constraints(f.path(), &tbl, &LineBusPairIndex::default()).unwrap();
        let bound = result[0].bound_lower.as_ref().expect("lower bound present");
        assert_eq!(params_of(bound), [EntityId(7)]);
        assert_eq!(eval(bound, &[(7, 4.25)]).to_bits(), 4.25_f64.to_bits());
    }

    /// A bound-position `@name` that is neither a parameter in `name_to_id` nor a
    /// declared named expression is a `SchemaError` on the constraint's
    /// `expression` field, naming it.
    #[test]
    fn test_parse_unknown_inline_bound_returns_schema_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "bad_bound",
      "expression": "hydro_generation(0) >= @missing_param",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("expression"),
                    "field should name the expression, got: {field}"
                );
                assert!(
                    message.contains("unknown name \"@missing_param\""),
                    "message should name the missing parameter, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A relational side referencing a declared named expression inlines it, so
    /// its variable terms fold into the merged LHS rather than the bound
    /// remainder.
    #[test]
    fn test_parse_bound_side_named_expression_inlines_into_merged_lhs() {
        let json = r#"{
  "expressions": [
    { "name": "fnese", "expression": "hydro_generation(10) + hydro_generation(11)" }
  ],
  "constraints": [
    { "id": 0, "name": "c0", "expression": "hydro_generation(0) >= @fnese", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].bound_lower,
            Some(ScalarExpr::literal(0.0)),
            "an all-variable RHS supplies an explicit zero bound, not \"no bound\""
        );
        assert_eq!(result[0].bound_upper, None);
        let terms = &result[0].expression.terms;
        assert_eq!(terms.len(), 3, "hydro_generation(0) plus fnese's two terms");
        for hydro_id in [0, 10, 11] {
            assert!(
                terms.iter().any(|t| t.variable
                    == VariableRef::HydroGeneration {
                        hydro_id: EntityId(hydro_id),
                        block_id: None,
                        bus_id: None,
                    }),
                "expected hydro_generation({hydro_id}) in the merged terms, got: {terms:?}"
            );
        }
    }

    /// A `bound_upper_ref` JSON field is an unknown field under
    /// `deny_unknown_fields`: a symbolic endpoint is authored inline in the
    /// expression.
    #[test]
    fn test_parse_bound_upper_ref_field_is_unknown_field_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "a",
      "expression": "hydro_generation(0)",
      "slack": { "enabled": false },
      "bound_upper_ref": "cap"
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for the \"bound_upper_ref\" field, got: {err:?}"
        );
    }

    /// A constraint whose expression carries no inline bound parses with
    /// `bound_lower` and `bound_upper` both `None`.
    #[test]
    fn test_parse_absent_bound_refs_are_none() {
        let f = write_json(VALID_JSON);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        for gc in &result {
            assert_eq!(gc.bound_lower, None);
            assert_eq!(gc.bound_upper, None);
        }
    }

    /// Unknown `@param` in JSON expression → SchemaError with "expression" field and
    /// the unknown name in the message.
    #[test]
    fn test_parse_unknown_param_returns_schema_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "bad_ref",
      "expression": "@missing * hydro_generation(0)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("expression"),
                    "field should contain 'expression', got: {field}"
                );
                assert!(
                    message.contains("unknown name \"@missing\""),
                    "message should name the unknown parameter, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Invalid expression → SchemaError with "expression" in field.
    #[test]
    fn test_parse_invalid_expression_returns_schema_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "bad",
      "expression": "invalid_var(0)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("expression"),
                    "field should contain 'expression', got: {field}"
                );
                assert!(
                    message.contains("unknown variable"),
                    "message should contain 'unknown variable', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A repeated block argument → SchemaError on the constraint and on the named-expression field.
    #[test]
    fn parse_generic_constraints_repeated_block_argument_is_schema_error() {
        let in_constraint = r#"{
  "constraints": [
    { "id": 0, "name": "bad", "expression": "thermal_generation(0, 0, 1) <= 10", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(in_constraint);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "constraints[0].expression");
                assert!(
                    message.contains("repeated block argument in variable \"thermal_generation\""),
                    "message should name the repeated block argument, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }

        let in_named_expression = r#"{
  "expressions": [
    { "name": "e", "expression": "hydro_turbined(5, 0, 1)" }
  ],
  "constraints": [
    { "id": 0, "name": "c0", "expression": "@e <= 10", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(in_named_expression);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "expressions[0].expression");
                assert!(
                    message.contains("repeated block argument in variable \"hydro_turbined\""),
                    "message should name the repeated block argument, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A constant inside a group on a relational side scales into the bound: the
    /// group's leading coefficient multiplies the constant like every other term.
    #[test]
    fn parse_generic_constraints_relational_constant_inside_group_scales_into_the_bound() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "scaled",
      "expression": "thermal_generation(5) <= 2 * (hydro_generation(140) + 73)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        let terms = &result[0].expression.terms;
        assert_eq!(terms.len(), 2);
        assert_eq!(
            terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(140),
                block_id: None,
                bus_id: None,
            }
        );
        assert_eq!(lit(&terms[0]).to_bits(), (-2.0_f64).to_bits());
        assert_eq!(
            terms[1].variable,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: None,
            }
        );
        assert_eq!(lit(&terms[1]).to_bits(), 1.0_f64.to_bits());
        // derived: 0 + 2·73
        assert_eq!(result[0].bound_upper, Some(ScalarExpr::literal(146.0)));
        assert_eq!(result[0].bound_lower, None);
    }

    /// Duplicate IDs → SchemaError.
    #[test]
    fn test_parse_duplicate_ids_returns_schema_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "a",
      "expression": "hydro_generation(0)",
      "slack": { "enabled": false }
    },
    {
      "id": 0,
      "name": "b",
      "expression": "thermal_generation(1)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("id"),
                    "field should contain 'id', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Slack enabled without penalty → SchemaError.
    #[test]
    fn test_parse_slack_enabled_without_penalty_returns_schema_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "a",
      "expression": "hydro_generation(0)",
      "slack": { "enabled": true }
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("penalty"),
                    "field should contain 'penalty', got: {field}"
                );
                assert!(
                    message.contains("absent") || message.contains("enabled"),
                    "message should explain the issue, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Clean break: a constraint object still carrying a `"sense"` key is rejected
    /// (`RawConstraint` is `deny_unknown_fields`, no alias) — shape is derived from
    /// the bounds parquet's endpoint pair, never authored on the constraint.
    #[test]
    fn test_parse_unknown_sense_field_returns_parse_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "a",
      "expression": "hydro_generation(0)",
      "sense": ">=",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for an unknown 'sense' field, got: {err:?}"
        );
    }

    /// `None` path → `Ok(Vec::new())` (tested via `load_generic_constraints`).
    /// The `load_*` wrapper is in `mod.rs`; here we test the `parse_*` function returns Ok
    /// for a valid empty constraints array.
    #[test]
    fn test_parse_empty_constraints_array() {
        let json = r#"{ "constraints": [] }"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert!(result.is_empty());
    }

    /// Sorted output: constraints come back in id-ascending order regardless of JSON order.
    #[test]
    fn test_parse_sorted_by_id() {
        let json = r#"{
  "constraints": [
    {
      "id": 5,
      "name": "c",
      "expression": "hydro_generation(0)",
      "slack": { "enabled": false }
    },
    {
      "id": 2,
      "name": "b",
      "expression": "thermal_generation(0)",
      "slack": { "enabled": false }
    },
    {
      "id": 0,
      "name": "a",
      "expression": "line_direct(0)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].id, EntityId(0));
        assert_eq!(result[1].id, EntityId(2));
        assert_eq!(result[2].id, EntityId(5));
    }

    /// Full JSON constraint with `line_exchange` parses correctly.
    #[test]
    fn test_parse_line_exchange_json_constraint() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "net_exchange",
      "expression": "line_exchange(0)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "net_exchange");
        assert_eq!(result[0].expression.terms.len(), 1);
        assert_eq!(
            result[0].expression.terms[0].variable,
            VariableRef::LineExchange {
                line_id: EntityId(0),
                block_id: None,
            }
        );
    }

    /// Slack with zero penalty → SchemaError.
    #[test]
    fn test_parse_slack_zero_penalty_returns_schema_error() {
        let (field, message) = slack_schema_error(
            r#"{ "enabled": true, "penalty": 0.0, "penalty_basis": "per_hour" }"#,
        );
        assert!(
            field.contains("penalty"),
            "field should contain 'penalty', got: {field}"
        );
        assert!(
            message.contains("> 0.0"),
            "message should mention > 0.0, got: {message}"
        );
    }

    /// An enabled slack with a penalty below zero fails on `slack.penalty`, naming the value.
    #[test]
    fn a_negative_slack_penalty_is_a_schema_error() {
        let (field, message) = slack_schema_error(
            r#"{ "enabled": true, "penalty": -1.0, "penalty_basis": "per_hour" }"#,
        );
        assert_eq!(field, "constraints[0].slack.penalty");
        assert_eq!(message, "slack.penalty must be > 0.0 when enabled, got -1");
    }

    /// An enabled slack resolves to its penalty and basis; a disabled one to `None`.
    #[test]
    fn test_parse_slack_resolves_penalty_and_basis() {
        let f = write_json(VALID_JSON);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(
            result[0].slack,
            Some(SlackConfig {
                penalty: 5000.0,
                penalty_basis: PenaltyBasis::PerUnit,
            })
        );
        assert_eq!(result[1].slack, None);
    }

    fn parse_slack_err(slack: &str) -> LoadError {
        let json = format!(
            r#"{{
  "constraints": [
    {{
      "id": 0,
      "name": "a",
      "expression": "hydro_generation(0)",
      "slack": {slack}
    }}
  ]
}}"#
        );
        let f = write_json(&json);
        parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
            .unwrap_err()
    }

    fn slack_schema_error(slack: &str) -> (String, String) {
        match parse_slack_err(slack) {
            LoadError::SchemaError { field, message, .. } => (field, message),
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_parse_slack_enabled_without_penalty_basis_returns_schema_error() {
        let (field, message) = slack_schema_error(r#"{ "enabled": true, "penalty": 5000.0 }"#);
        assert_eq!(field, "constraints[0].slack.penalty_basis");
        assert!(
            message.contains("per_hour") && message.contains("per_unit"),
            "message should name both bases, got: {message}"
        );
    }

    #[test]
    fn test_parse_disabled_slack_with_penalty_returns_schema_error() {
        let (field, _) = slack_schema_error(r#"{ "enabled": false, "penalty": 5000.0 }"#);
        assert_eq!(field, "constraints[0].slack.penalty");
    }

    #[test]
    fn test_parse_disabled_slack_with_penalty_basis_returns_schema_error() {
        let (field, _) = slack_schema_error(r#"{ "enabled": false, "penalty_basis": "per_unit" }"#);
        assert_eq!(field, "constraints[0].slack.penalty_basis");
    }

    #[test]
    fn test_parse_unknown_penalty_basis_returns_parse_error() {
        let err = parse_slack_err(
            r#"{ "enabled": true, "penalty": 5000.0, "penalty_basis": "per_stage" }"#,
        );
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for an unknown penalty_basis, got: {err:?}"
        );
    }

    /// Description is optional — absent means `None`.
    #[test]
    fn test_parse_description_optional() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "nodesc",
      "expression": "hydro_generation(0)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].description.is_none());
    }

    fn raw_named(name: &str, expression: &str) -> RawNamedExpression {
        RawNamedExpression {
            name: name.to_string(),
            expression: expression.to_string(),
            _description: None,
        }
    }

    /// Byte-neutral: no declarations ⇒ empty table, no error.
    #[test]
    fn declare_named_expressions_empty_is_ok() {
        let table = declare_named_expressions(
            &[],
            Path::new("generic_constraints.json"),
            &HashMap::new(),
            &LineBusPairIndex::default(),
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(table.reference_graph().is_empty());
    }

    /// Two distinct valid definitions ⇒ a two-entry table, each lowering to its own terms in
    /// authoring order.
    #[test]
    fn declare_named_expressions_two_valid_definitions() {
        let entries = [
            raw_named("fnese", "hydro_generation(0) + hydro_generation(1)"),
            raw_named("fns", "2.5 * thermal_generation(5) - hydro_generation(3)"),
        ];
        let params = HashMap::new();
        let line_index = LineBusPairIndex::default();
        let table = declare_named_expressions(
            &entries,
            Path::new("generic_constraints.json"),
            &params,
            &line_index,
            &BTreeMap::new(),
        )
        .unwrap();
        let names: Vec<&str> = table
            .reference_graph()
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(names, ["fnese", "fns"]);

        let scope = Scope {
            params: &params,
            line_index: &line_index,
            named: &table,
            data_ids: None,
        };
        let hydro = |id: i32| VariableRef::HydroGeneration {
            hydro_id: EntityId(id),
            block_id: None,
            bus_id: None,
        };
        let fnese = lower_relation("@fnese", &scope).unwrap();
        let variables: Vec<VariableRef> =
            fnese.terms.iter().map(|(variable, _)| *variable).collect();
        assert_eq!(variables, [hydro(0), hydro(1)]);
        let fns = lower_relation("@fns", &scope).unwrap();
        assert_eq!(
            fns.terms[0].0,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: None,
            }
        );
        assert_eq!(fns.terms[0].1.as_literal(), Some(2.5));
        assert_eq!(fns.terms[1].0, hydro(3));
        assert_eq!(fns.terms[1].1.as_literal(), Some(-1.0));
    }

    /// Two declarations sharing a name ⇒ SchemaError naming the duplicate, field under `expressions`.
    #[test]
    fn named_expression_names_duplicate_is_error() {
        let entries = [
            raw_named("fnese", "hydro_generation(0)"),
            raw_named("fnese", "hydro_generation(1)"),
        ];
        let err = check_named_expression_names(
            &entries,
            Path::new("generic_constraints.json"),
            &HashMap::new(),
        )
        .unwrap_err();
        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("expressions"), "field: {field}");
                assert!(message.contains("fnese"), "message: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A declaration whose name is already a scalar parameter ⇒ SchemaError naming it.
    #[test]
    fn named_expression_names_param_collision_is_error() {
        let entries = [raw_named("rho", "hydro_generation(0)")];
        let err = check_named_expression_names(
            &entries,
            Path::new("generic_constraints.json"),
            &one_param_table(),
        )
        .unwrap_err();
        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("expressions"), "field: {field}");
                assert!(message.contains("rho"), "message: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A file carrying both constraints and a valid `expressions` array parses; the
    /// returned constraints are unaffected by the declarations.
    #[test]
    fn parse_generic_constraints_with_expressions_key() {
        let json = r#"{
  "expressions": [
    { "name": "fnese", "expression": "hydro_generation(10) + hydro_generation(11)", "description": "SE net flow" }
  ],
  "constraints": [
    {
      "id": 0,
      "name": "c0",
      "expression": "hydro_generation(0)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, EntityId(0));
        assert_eq!(result[0].expression.terms.len(), 1);
    }

    /// A declaration reading an unknown name fails the load even when every
    /// constraint is valid and no constraint uses the declaration.
    #[test]
    fn parse_generic_constraints_bad_expression_declaration_fails_load() {
        let json = r#"{
  "expressions": [
    { "name": "bad", "expression": "@other + hydro_generation(0)" }
  ],
  "constraints": [
    { "id": 0, "name": "c0", "expression": "hydro_generation(0)", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        assert!(
            matches!(&err, LoadError::SchemaError { message, .. } if message.contains("unknown name \"@other\"")),
            "expected an unknown-name error naming \"other\", got: {err:?}"
        );
    }

    /// An empty `expressions` array is byte-neutral — same constraints as with the key absent.
    #[test]
    fn parse_generic_constraints_empty_expressions_is_byte_neutral() {
        let with_key = r#"{ "expressions": [], "constraints": [ { "id": 0, "name": "c0", "expression": "hydro_generation(0)", "slack": { "enabled": false } } ] }"#;
        let without_key = r#"{ "constraints": [ { "id": 0, "name": "c0", "expression": "hydro_generation(0)", "slack": { "enabled": false } } ] }"#;
        let with_file = write_json(with_key);
        let without_file = write_json(without_key);
        let a = parse_generic_constraints(
            with_file.path(),
            &HashMap::new(),
            &LineBusPairIndex::default(),
        )
        .unwrap();
        let b = parse_generic_constraints(
            without_file.path(),
            &HashMap::new(),
            &LineBusPairIndex::default(),
        )
        .unwrap();
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].id, b[0].id);
        assert_eq!(a[0].expression.terms.len(), b[0].expression.terms.len());
    }

    /// An unknown key inside a declaration is rejected (`deny_unknown_fields`).
    #[test]
    fn parse_generic_constraints_unknown_expression_field_returns_parse_error() {
        let json = r#"{
  "expressions": [
    { "name": "x", "expression": "hydro_generation(0)", "unexpected": 1 }
  ],
  "constraints": []
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for an unknown declaration field, got: {err:?}"
        );
    }

    // ── Named-expression inlining tests ───────────────────────────────────────

    /// Single-reference distribution: `2.0 * @fnese - thermal_generation(3)` with
    /// `fnese = hydro_generation(0) + hydro_generation(1)` flattens to the
    /// hand-written expansion — 3 terms, coefficients 2.0, 2.0, -1.0.
    #[test]
    fn parse_generic_constraints_inlines_single_reference() {
        let json = r#"{
  "expressions": [
    { "name": "fnese", "expression": "hydro_generation(0) + hydro_generation(1)" }
  ],
  "constraints": [
    { "id": 0, "name": "c0", "expression": "2.0 * @fnese - thermal_generation(3)", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        let terms = &result[0].expression.terms;
        assert_eq!(terms.len(), 3);
        assert!((lit(&terms[0]) - 2.0).abs() < 1e-10);
        assert!((lit(&terms[1]) - 2.0).abs() < 1e-10);
        assert!((lit(&terms[2]) - (-1.0)).abs() < 1e-10);
        assert_eq!(
            terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: None,
            }
        );
        assert_eq!(
            terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(1),
                block_id: None,
                bus_id: None,
            }
        );
        assert_eq!(
            terms[2].variable,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(3),
                block_id: None,
            }
        );
    }

    /// A group's inner term list may contain a standalone `@name` reference,
    /// exactly as a top-level term list may: the group's leading coefficient
    /// distributes into the reference's own `scale`, composing with the
    /// reference's own expansion at inline time.
    #[test]
    fn parse_generic_constraints_group_distributes_into_named_reference() {
        let json = r#"{
  "expressions": [
    { "name": "fnese", "expression": "hydro_generation(0) + hydro_generation(1)" }
  ],
  "constraints": [
    { "id": 0, "name": "c0", "expression": "2.0 * (@fnese - hydro_generation(2))", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        let terms = &result[0].expression.terms;
        assert_eq!(terms.len(), 3);
        assert!((lit(&terms[0]) - 2.0).abs() < 1e-10);
        assert!((lit(&terms[1]) - 2.0).abs() < 1e-10);
        assert!((lit(&terms[2]) - (-2.0)).abs() < 1e-10);
    }

    /// Composition resolves transitively: a constraint referencing `outer`, which
    /// references `inner`, which references a flat `base`.
    #[test]
    fn parse_generic_constraints_inlines_composition() {
        let json = r#"{
  "expressions": [
    { "name": "base", "expression": "hydro_generation(0)" },
    { "name": "inner", "expression": "3.0 * @base" },
    { "name": "outer", "expression": "@inner + hydro_generation(1)" }
  ],
  "constraints": [
    { "id": 0, "name": "c0", "expression": "2.0 * @outer", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        let terms = &result[0].expression.terms;
        assert_eq!(terms.len(), 2);
        assert!((lit(&terms[0]) - 6.0).abs() < 1e-10);
        assert!((lit(&terms[1]) - 2.0).abs() < 1e-10);
        assert_eq!(
            terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: None,
            }
        );
        assert_eq!(
            terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(1),
                block_id: None,
                bus_id: None,
            }
        );
    }

    /// A 2-node reference cycle ⇒ SchemaError whose message names both nodes and
    /// the cycle path.
    #[test]
    fn parse_generic_constraints_two_node_cycle_errors() {
        let json = r#"{
  "expressions": [
    { "name": "a", "expression": "@b" },
    { "name": "b", "expression": "@a" }
  ],
  "constraints": []
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains('a') && message.contains('b'),
                    "message: {message}"
                );
                assert!(
                    message.starts_with("named-expression reference cycle detected: "),
                    "message: {message}"
                );
                assert!(message.contains("a -> b -> a"), "message: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A self-reference is a cycle of length 1 ⇒ SchemaError naming the node.
    #[test]
    fn parse_generic_constraints_self_reference_errors() {
        let json = r#"{
  "expressions": [
    { "name": "e", "expression": "@e" }
  ],
  "constraints": []
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        assert!(
            matches!(&err, LoadError::SchemaError { message, .. }
                if message.starts_with("named-expression reference cycle detected: ")
                    && message.contains("e -> e")),
            "expected a self-reference cycle naming \"e\", got: {err:?}"
        );
    }

    /// A reference to an unknown name inside a constraint ⇒ SchemaError naming
    /// the missing name, on the constraint's expression field.
    #[test]
    fn parse_generic_constraints_undeclared_reference_in_constraint_errors() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "c0", "expression": "@nope", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("expression"), "field: {field}");
                assert!(
                    message.contains("unknown name \"@nope\""),
                    "message: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `@param * @name` scales the referenced body: the term keeps the parameter as its
    /// coefficient.
    #[test]
    fn parse_generic_constraints_param_times_reference_scales_the_body() {
        let json = r#"{
  "expressions": [
    { "name": "fnese", "expression": "hydro_generation(0)" }
  ],
  "constraints": [
    { "id": 0, "name": "c0", "expression": "@rho * @fnese", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &one_param_table(), &LineBusPairIndex::default())
                .unwrap();
        let terms = &result[0].expression.terms;
        assert_eq!(terms.len(), 1);
        assert_eq!(
            terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: None,
            }
        );
        assert_eq!(terms[0].coefficient.as_param(), Some(EntityId(7)));
    }

    /// A `levels`-deep doubling chain `e0 = @rho * hydro_generation(0)`,
    /// `e_k = @e_{k-1} + @e_{k-1}` as a JSON `expressions` array body. The base reads a
    /// parameter, so its coefficient program doubles at every level instead of folding to
    /// one literal.
    fn doubling_chain_expressions(levels: u32) -> String {
        let mut exprs =
            String::from(r#"{ "name": "e0", "expression": "@rho * hydro_generation(0)" }"#);
        for k in 1..=levels {
            let _ = write!(
                exprs,
                r#", {{ "name": "e{k}", "expression": "@e{prev} + @e{prev}" }}"#,
                prev = k - 1
            );
        }
        exprs
    }

    /// A declared-but-unreferenced doubling chain must load without expanding: if
    /// declaration-time validation force-expanded it, this 2^60-term chain would
    /// exhaust memory. Completing quickly proves declarations read references as empty forms.
    #[test]
    fn parse_generic_constraints_unreferenced_doubling_chain_parses_fast() {
        let json = format!(
            r#"{{ "expressions": [ {} ], "constraints": [ {{ "id": 0, "name": "c0", "expression": "hydro_generation(0)", "slack": {{ "enabled": false }} }} ] }}"#,
            doubling_chain_expressions(60)
        );
        let f = write_json(&json);
        let result =
            parse_generic_constraints(f.path(), &one_param_table(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].expression.terms.len(), 1);
    }

    /// A constraint that references the top of the doubling chain triggers the
    /// term-budget cap — loudly and fast, not by exhausting memory.
    #[test]
    fn parse_generic_constraints_referenced_doubling_chain_hits_budget_fast() {
        let json = format!(
            r#"{{ "expressions": [ {} ], "constraints": [ {{ "id": 0, "name": "c0", "expression": "@e60", "slack": {{ "enabled": false }} }} ] }}"#,
            doubling_chain_expressions(60)
        );
        let f = write_json(&json);
        let err =
            parse_generic_constraints(f.path(), &one_param_table(), &LineBusPairIndex::default())
                .unwrap_err();
        assert!(
            matches!(&err, LoadError::SchemaError { field, message, .. }
                if field.contains("constraints[0].expression")
                    && message.contains("more than")
                    && message.contains("100000")),
            "expected a budget-exceeded error on the constraint, got: {err:?}"
        );
    }

    /// A legitimate large-but-under-cap flat constraint expression parses: the cap
    /// rejects only pathological exponential expansion, never a big flat sum.
    #[test]
    fn parse_generic_constraints_large_flat_expression_under_cap_parses() {
        let mut expr = String::from("hydro_generation(0)");
        for id in 1..1000 {
            let _ = write!(expr, " + hydro_generation({id})");
        }
        let json = format!(
            r#"{{ "constraints": [ {{ "id": 0, "name": "big", "expression": "{expr}", "slack": {{ "enabled": false }} }} ] }}"#
        );
        let f = write_json(&json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result[0].expression.terms.len(), 1000);
    }

    fn constraint_json(expression: &str) -> String {
        format!(
            r#"{{ "constraints": [ {{ "id": 0, "name": "c0", "expression": "{expression}", "slack": {{ "enabled": false }} }} ] }}"#
        )
    }

    fn load(
        json: &str,
        params: &HashMap<String, EntityId>,
    ) -> Result<LoadedGenericConstraints, LoadError> {
        let file = write_json(json);
        parse_generic_constraints_with_provenance(file.path(), params, &LineBusPairIndex::default())
    }

    fn schema_error(json: &str, params: &HashMap<String, EntityId>) -> (String, String) {
        match load(json, params).err() {
            Some(LoadError::SchemaError { field, message, .. }) => (field, message),
            other => panic!("expected a SchemaError, got: {other:?}"),
        }
    }

    fn hydro_0() -> VariableRef {
        VariableRef::HydroGeneration {
            hydro_id: EntityId(0),
            block_id: None,
            bus_id: None,
        }
    }

    /// derived: `k` is id 4, so the load of bus 1 is id 5, and `@FLOW + @NET_LOAD == 0` leaves
    /// `0 - (L - 3)` on both endpoints.
    #[test]
    fn a_formula_with_a_data_part_shifts_its_table_endpoints() {
        let json = r#"{ "constraints": [
    { "id": 450, "name": "constraint_450", "expression": "@FLOW + @NET_LOAD",
      "slack": { "enabled": true, "penalty": 5000.0, "penalty_basis": "per_hour" } } ],
  "expressions": [
    { "name": "FLOW", "expression": "line_flow(source_bus=0, target_bus=1)" },
    { "name": "NET_LOAD", "expression": "bus_load_value(1) - @k" } ] }"#;
        let file = write_json(json);
        let line_index = build_line_bus_pair_index(&[make_line(3, 0, 1)]).unwrap();
        let LoadedGenericConstraints {
            constraints,
            scenario,
            ..
        } = parse_generic_constraints_with_provenance(
            file.path(),
            &HashMap::from([("k".to_owned(), EntityId(4))]),
            &line_index,
        )
        .unwrap();

        assert_eq!(constraints[0].id, EntityId(450));
        assert_eq!(
            constraints[0].expression.terms,
            [LinearTerm::literal(
                1.0,
                VariableRef::LineDirect {
                    line_id: EntityId(3),
                    block_id: None,
                }
            )]
        );
        assert_eq!(constraints[0].bound_lower, None);
        assert_eq!(constraints[0].bound_upper, None);
        assert_eq!(
            scenario.data_atoms,
            [(
                EntityId(5),
                DataAtom::BusLoadValue {
                    bus_id: EntityId(1),
                    block_id: None,
                }
            )]
        );
        assert!(scenario.availability_atoms.is_empty() && scenario.selectors.is_empty());
        let [
            ConstraintScenario {
                constraint_id: EntityId(450),
                active_when: None,
                endpoint_shift: Some(shift),
            },
        ] = scenario.constraints.as_slice()
        else {
            panic!(
                "expected one shift for 450, got: {:?}",
                scenario.constraints
            );
        };
        let load = 10.0_f64;
        assert_eq!(
            eval(shift, &[(5, load), (4, 3.0)]).to_bits(),
            (-(load - 3.0)).to_bits()
        );
    }

    /// derived: no parameter is declared, so the load of bus 0 the condition reads is id 0 and
    /// the selector is id 1.
    #[test]
    fn an_if_becomes_a_selector_read_by_the_bound() {
        let LoadedGenericConstraints {
            constraints,
            scenario,
            ..
        } = load(
            &constraint_json("hydro_generation(0) <= if(bus_load_value(0) > 100, 800, 600)"),
            &HashMap::new(),
        )
        .unwrap();

        assert_eq!(
            scenario.data_atoms,
            [(
                EntityId(0),
                DataAtom::BusLoadValue {
                    bus_id: EntityId(0),
                    block_id: None,
                }
            )]
        );
        assert_eq!(
            scenario.selectors,
            [(
                EntityId(1),
                ConditionalScalar {
                    cond: ResolvedCondition::Compare {
                        lhs: ScalarExpr::param(EntityId(0)),
                        op: CompareOp::Gt,
                        rhs: ScalarExpr::literal(100.0),
                    },
                    then: ScalarExpr::literal(800.0),
                    else_: ScalarExpr::literal(600.0),
                }
            )]
        );
        let bound = constraints[0].bound_upper.as_ref().expect("an upper bound");
        assert_eq!(params_of(bound), [EntityId(1)]);
        assert_eq!(constraints[0].bound_lower, None);
        assert!(scenario.constraints.is_empty());
    }

    /// derived: no parameter is declared, so the loads of buses 0 and 1 are ids 0 and 1 and the
    /// selector, which compares their sum with 90000, is id 2.
    #[test]
    fn a_selector_condition_reads_the_sum_of_bus_loads() {
        let LoadedGenericConstraints {
            constraints,
            scenario,
            ..
        } = load(
            &constraint_json(
                "hydro_generation(0) <= if((bus_load_value(0) + bus_load_value(1)) > 90000, 800, 600)",
            ),
            &HashMap::new(),
        )
        .unwrap();

        let bus = |id: i32| DataAtom::BusLoadValue {
            bus_id: EntityId(id),
            block_id: None,
        };
        assert_eq!(
            scenario.data_atoms,
            [(EntityId(0), bus(0)), (EntityId(1), bus(1))]
        );
        assert_eq!(
            scenario.selectors,
            [(
                EntityId(2),
                ConditionalScalar {
                    cond: ResolvedCondition::Compare {
                        lhs: ScalarExpr::param(EntityId(0)) + ScalarExpr::param(EntityId(1)),
                        op: CompareOp::Gt,
                        rhs: ScalarExpr::literal(90_000.0),
                    },
                    then: ScalarExpr::literal(800.0),
                    else_: ScalarExpr::literal(600.0),
                }
            )]
        );
        let bound = constraints[0].bound_upper.as_ref().expect("an upper bound");
        assert_eq!(params_of(bound), [EntityId(2)]);
        assert!(scenario.constraints.is_empty());
    }

    /// derived: `k` is id 0 and no data is read, so the selector of the constraint first in the
    /// file, 7, is id 1 and that of 3 is id 2.
    #[test]
    fn selectors_are_numbered_across_the_constraints_in_file_order() {
        let json = r#"{ "constraints": [
    { "id": 7, "name": "c7", "expression": "hydro_generation(0) <= if(@k > 1, 10, 20)", "slack": { "enabled": false } },
    { "id": 3, "name": "c3", "expression": "hydro_generation(1) >= if(@k > 1, 30, 40)", "slack": { "enabled": false } } ] }"#;
        let LoadedGenericConstraints {
            constraints,
            scenario,
            ..
        } = load(json, &HashMap::from([("k".to_owned(), EntityId(0))])).unwrap();

        let ids: Vec<i32> = scenario.selectors.iter().map(|(id, _)| id.0).collect();
        assert_eq!(ids, [1, 2]);
        assert_eq!(scenario.selectors[0].1.then, ScalarExpr::literal(10.0));
        let (c3, c7) = (&constraints[0], &constraints[1]);
        assert_eq!(params_of(c7.bound_upper.as_ref().unwrap()), [EntityId(1)]);
        assert_eq!(params_of(c3.bound_lower.as_ref().unwrap()), [EntityId(2)]);
    }

    #[test]
    fn a_formula_constant_is_an_endpoint_shift() {
        let LoadedGenericConstraints {
            constraints,
            scenario,
            ..
        } = load(&constraint_json("hydro_generation(0) + 5"), &HashMap::new()).unwrap();
        assert_eq!(
            constraints[0].expression.terms,
            [LinearTerm::literal(1.0, hydro_0())]
        );
        assert_eq!(constraints[0].bound_lower, None);
        assert_eq!(constraints[0].bound_upper, None);
        assert_eq!(
            scenario,
            ScenarioLayer {
                constraints: vec![ConstraintScenario {
                    constraint_id: EntityId(0),
                    active_when: None,
                    endpoint_shift: Some(ScalarExpr::literal(-5.0)),
                }],
                ..ScenarioLayer::default()
            }
        );

        let without_constant =
            load(&constraint_json("hydro_generation(0)"), &HashMap::new()).unwrap();
        assert!(without_constant.scenario.is_empty());
        assert_eq!(without_constant.constraints[0].bound_lower, None);
        assert_eq!(without_constant.constraints[0].bound_upper, None);
    }

    #[test]
    fn a_condition_reading_an_lp_variable_is_a_schema_error() {
        let (field, message) = schema_error(
            &constraint_json("hydro_generation(0) <= if(hydro_generation(1) > 100, 800, 600)"),
            &HashMap::new(),
        );
        assert_eq!(field, "constraints[0].expression");
        assert_eq!(
            message,
            "`hydro_generation(1)`: an if() condition may read only data, not an LP variable"
        );
    }

    #[test]
    fn if_branches_over_different_variables_are_a_schema_error() {
        let conditional = "if(bus_load_value(0) > 1, hydro_generation(0), hydro_generation(1))";
        let (field, message) = schema_error(
            &constraint_json(&format!("{conditional} <= 5")),
            &HashMap::new(),
        );
        assert_eq!(field, "constraints[0].expression");
        assert_eq!(
            message,
            format!(
                "`{conditional}`: the branches of if() must read the same LP variables; they may \
                 differ only in coefficients and constants"
            )
        );
    }

    /// derived: with `k` at the largest id no data id fits; with `k` one below it the only
    /// selector takes the largest id and the next would not fit.
    #[test]
    fn ids_past_the_largest_are_a_schema_error() {
        let at = |id: i32| HashMap::from([("k".to_owned(), EntityId(id))]);
        assert_eq!(
            schema_error(
                &constraint_json("hydro_generation(0) <= bus_load_value(1)"),
                &at(i32::MAX)
            ),
            (
                "constraints".to_owned(),
                "the scalar-parameter ids leave no room for the data terms and selectors of the \
                 constraints"
                    .to_owned()
            )
        );
        assert_eq!(
            schema_error(
                &constraint_json("hydro_generation(0) <= if(@k > 1, 1, 2)"),
                &at(i32::MAX - 1)
            ),
            (
                "constraints[0].expression".to_owned(),
                "the scalar-parameter ids leave no room for the selectors of the constraints"
                    .to_owned()
            )
        );
    }

    /// derived: `D{n}` holds `2^(n+1) - 1` operations, so the left side holds
    /// 65535 + 32767 + 1023 + 511 + 127 + 31 + 5 = 99999, within the budget, and subtracting it
    /// from the remainder takes 100001.
    #[test]
    fn a_formula_fault_quotes_its_own_text() {
        let (_, unknown) = schema_error(
            &constraint_json("hydro_generation(0) + nope(1)"),
            &HashMap::new(),
        );
        assert!(
            unknown.starts_with("`nope(1)`: unknown variable"),
            "{unknown}"
        );

        let mut expressions = String::from(r#"{ "name": "D0", "expression": "@k" }"#);
        for n in 1..=15 {
            let _ = write!(
                expressions,
                r#", {{ "name": "D{n}", "expression": "@D{m} + @D{m}" }}"#,
                m = n - 1
            );
        }
        let src = "@D15 + @D14 + @D9 + @D8 + @D6 + @D4";
        let json = format!(
            r#"{{ "expressions": [ {expressions} ], "constraints": [ {{ "id": 0, "name": "c0", "expression": "{src}", "slack": {{ "enabled": false }} }} ] }}"#
        );
        let (field, message) = schema_error(&json, &HashMap::from([("k".to_owned(), EntityId(0))]));
        assert_eq!(field, "constraints[0].expression");
        assert_eq!(
            message,
            format!("`{src}`: expression expands to more than 100000 coefficient operations")
        );
    }

    fn active_when_json(expression: &str, active_when: &str) -> String {
        format!(
            r#"{{ "constraints": [ {{ "id": 0, "name": "c0", "expression": "{expression}", "active_when": "{active_when}", "slack": {{ "enabled": false }} }} ] }}"#
        )
    }

    /// derived: no parameter is declared, so the load of bus 0 the condition reads is id 0.
    #[test]
    fn active_when_lowers_to_a_condition_in_the_layer() {
        let file = |active_when: &str| {
            format!(
                r#"{{ "constraints": [ {{ "id": 703, "name": "constraint_703",
  "expression": "hydro_generation(0) <= 1000000",{active_when}
  "slack": {{ "enabled": false }} }} ] }}"#
            )
        };
        let LoadedGenericConstraints {
            constraints,
            scenario,
            ..
        } = load(
            &file(r#" "active_when": "82000 < bus_load_value(0) & bus_load_value(0) <= 95000","#),
            &HashMap::new(),
        )
        .unwrap();

        assert_eq!(
            constraints,
            load(&file(""), &HashMap::new()).unwrap().constraints
        );
        assert_eq!(
            scenario.data_atoms,
            [(
                EntityId(0),
                DataAtom::BusLoadValue {
                    bus_id: EntityId(0),
                    block_id: None,
                }
            )]
        );
        assert!(scenario.availability_atoms.is_empty() && scenario.selectors.is_empty());
        let [
            ConstraintScenario {
                constraint_id,
                active_when: Some(condition),
                endpoint_shift: None,
            },
        ] = scenario.constraints.as_slice()
        else {
            panic!(
                "expected one condition for 703, got: {:?}",
                scenario.constraints
            );
        };
        assert_eq!(*constraint_id, EntityId(703));
        let load = || ScalarExpr::param(EntityId(0));
        assert_eq!(
            *condition,
            ResolvedCondition::And(
                Box::new(ResolvedCondition::Compare {
                    lhs: ScalarExpr::literal(82_000.0),
                    op: CompareOp::Lt,
                    rhs: load(),
                }),
                Box::new(ResolvedCondition::Compare {
                    lhs: load(),
                    op: CompareOp::Le,
                    rhs: ScalarExpr::literal(95_000.0),
                }),
            )
        );
        let holds_at = |value: f64| condition.holds(&mut |side| eval(side, &[(0, value)]));
        assert!(holds_at(90_000.0));
        assert!(!holds_at(96_000.0));
    }

    /// derived: no parameter is declared, so the loads of buses 0 and 1 are ids 0 and 1, and the
    /// condition holds where their sum is in `(82000, 95000]`.
    #[test]
    fn a_condition_reads_the_sum_of_bus_loads() {
        let sum = "(bus_load_value(0) + bus_load_value(1))";
        let LoadedGenericConstraints {
            constraints,
            scenario,
            ..
        } = load(
            &active_when_json(
                "hydro_generation(0) <= 1000000",
                &format!("82000 < {sum} & {sum} <= 95000"),
            ),
            &HashMap::new(),
        )
        .unwrap();

        assert_eq!(
            constraints,
            load(
                &constraint_json("hydro_generation(0) <= 1000000"),
                &HashMap::new()
            )
            .unwrap()
            .constraints
        );
        let bus = |id: i32| DataAtom::BusLoadValue {
            bus_id: EntityId(id),
            block_id: None,
        };
        assert_eq!(
            scenario.data_atoms,
            [(EntityId(0), bus(0)), (EntityId(1), bus(1))]
        );
        assert!(scenario.availability_atoms.is_empty() && scenario.selectors.is_empty());
        let [
            ConstraintScenario {
                constraint_id: EntityId(0),
                active_when: Some(condition),
                endpoint_shift: None,
            },
        ] = scenario.constraints.as_slice()
        else {
            panic!(
                "expected one condition for 0, got: {:?}",
                scenario.constraints
            );
        };
        let load = || ScalarExpr::param(EntityId(0)) + ScalarExpr::param(EntityId(1));
        assert_eq!(
            *condition,
            ResolvedCondition::And(
                Box::new(ResolvedCondition::Compare {
                    lhs: ScalarExpr::literal(82_000.0),
                    op: CompareOp::Lt,
                    rhs: load(),
                }),
                Box::new(ResolvedCondition::Compare {
                    lhs: load(),
                    op: CompareOp::Le,
                    rhs: ScalarExpr::literal(95_000.0),
                }),
            )
        );
        let holds_at = |bus_0: f64, bus_1: f64| {
            condition.holds(&mut |side| eval(side, &[(0, bus_0), (1, bus_1)]))
        };
        assert!(holds_at(40_000.0, 50_000.0));
        assert!(!holds_at(50_000.0, 46_000.0));
    }

    #[test]
    fn a_condition_and_an_endpoint_shift_share_the_constraint_entry() {
        let LoadedGenericConstraints { scenario, .. } = load(
            &active_when_json("hydro_generation(0) + 5", "bus_load_value(0) > 1"),
            &HashMap::new(),
        )
        .unwrap();
        let [
            ConstraintScenario {
                constraint_id: EntityId(0),
                active_when: Some(_),
                endpoint_shift: Some(shift),
            },
        ] = scenario.constraints.as_slice()
        else {
            panic!(
                "expected one entry with both for 0, got: {:?}",
                scenario.constraints
            );
        };
        assert_eq!(*shift, ScalarExpr::literal(-5.0));
    }

    /// The expression differs from the condition at the fault's span, so the quoted fragment
    /// shows which text the diagnostic was rendered against.
    #[test]
    fn a_condition_reading_an_lp_variable_names_its_field() {
        assert_eq!(
            schema_error(
                &active_when_json("thermal_generation(5) <= 100", "hydro_generation(0) > 10"),
                &HashMap::new()
            ),
            (
                "constraints[0].active_when".to_owned(),
                "`hydro_generation(0)`: an LP variable cannot appear in a scalar-parameter \
                 expression"
                    .to_owned()
            )
        );
    }

    #[test]
    fn a_condition_reads_neither_availability_nor_if() {
        for (active_when, message) in [
            (
                "hydro_availability(0) > 10",
                "`hydro_availability(0)`: a condition cannot read hydro_availability: it is the \
                 capacity of a plant, resolved per stage and block, not a scenario value",
            ),
            (
                "if(bus_load_value(0) > 1, 1, 2) > 1",
                "`if(bus_load_value(0) > 1, 1, 2)`: an activation condition cannot use if()",
            ),
        ] {
            assert_eq!(
                schema_error(
                    &active_when_json("hydro_generation(0) <= 5", active_when),
                    &HashMap::new()
                ),
                ("constraints[0].active_when".to_owned(), message.to_owned()),
                "{active_when}"
            );
        }
    }

    #[test]
    fn an_empty_condition_is_a_fault_of_its_field() {
        assert_eq!(
            schema_error(
                &active_when_json("hydro_generation(0) <= 5", ""),
                &HashMap::new()
            ),
            (
                "constraints[0].active_when".to_owned(),
                "expression must not be empty (at the end of the expression)".to_owned()
            )
        );
    }

    fn hydro_0_capacity() -> (EntityId, AvailabilityAtom) {
        (
            EntityId(0),
            AvailabilityAtom {
                hydro_id: EntityId(0),
                block_id: None,
            },
        )
    }

    /// derived: no parameter is declared, so the capacity is id 0, and moving it to the left of
    /// `<=` leaves `0 - (-1 · p)`, which is `p`.
    #[test]
    fn availability_folds_into_the_bound_as_data() {
        let json = r#"{ "constraints": [
    { "id": 951, "name": "constraint_951",
      "expression": "hydro_generation(0) <= hydro_availability(0)",
      "slack": { "enabled": false } } ] }"#;
        let LoadedGenericConstraints {
            constraints,
            scenario,
            ..
        } = load(json, &HashMap::new()).unwrap();

        assert_eq!(
            constraints[0].expression.terms,
            [LinearTerm::literal(1.0, hydro_0())]
        );
        let bound = constraints[0].bound_upper.as_ref().expect("an upper bound");
        assert_eq!(eval(bound, &[(0, 750.0)]).to_bits(), 750.0_f64.to_bits());
        assert_eq!(constraints[0].bound_lower, None);
        assert_eq!(scenario.availability_atoms, [hydro_0_capacity()]);
        assert!(
            scenario.data_atoms.is_empty()
                && scenario.selectors.is_empty()
                && scenario.constraints.is_empty()
        );
    }

    /// derived: `x - 0.5 · p == 0` leaves `0 - (-0.5 · p)` on both endpoints, which is `0.5 · p`.
    #[test]
    fn availability_in_a_formula_joins_the_endpoint_shift() {
        let LoadedGenericConstraints {
            constraints,
            scenario,
            ..
        } = load(
            &constraint_json("hydro_generation(0) - 0.5 * hydro_availability(0)"),
            &HashMap::new(),
        )
        .unwrap();

        assert_eq!(
            constraints[0].expression.terms,
            [LinearTerm::literal(1.0, hydro_0())]
        );
        assert_eq!(
            (&constraints[0].bound_lower, &constraints[0].bound_upper),
            (&None, &None)
        );
        assert_eq!(scenario.availability_atoms, [hydro_0_capacity()]);
        let [
            ConstraintScenario {
                constraint_id: EntityId(0),
                active_when: None,
                endpoint_shift: Some(shift),
            },
        ] = scenario.constraints.as_slice()
        else {
            panic!("expected one shift for 0, got: {:?}", scenario.constraints);
        };
        let capacity = 750.0_f64;
        assert_eq!(
            eval(shift, &[(0, capacity)]).to_bits(),
            (0.5 * capacity).to_bits()
        );
    }

    #[test]
    fn existing_generic_files_load_to_the_same_constraints_and_an_empty_layer() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/deterministic");
        let mut cases: Vec<_> = std::fs::read_dir(&root)
            .expect("the examples are listed")
            .map(|entry| entry.expect("an example").path())
            .filter(|case| case.join("constraints/generic_constraints.json").exists())
            .collect();
        cases.sort();
        assert!(!cases.is_empty(), "no example under {}", root.display());

        for case in cases {
            let LoadedCase { system, artifacts } = load_case_with_artifacts(&case)
                .unwrap_or_else(|e| panic!("{}: {e}", case.display()));
            assert!(
                artifacts.generic_scenario.is_empty(),
                "{}: {:?}",
                case.display(),
                artifacts.generic_scenario
            );
            let raw: RawGenericConstraintsFile = serde_json::from_str(
                &std::fs::read_to_string(case.join("constraints/generic_constraints.json"))
                    .expect("the constraints file reads"),
            )
            .expect("the constraints file deserializes");
            assert_eq!(system.generic_constraints().len(), raw.constraints.len());
            for c in raw.constraints {
                let loaded = system
                    .generic_constraints()
                    .iter()
                    .find(|gc| gc.id == EntityId(c.id))
                    .expect("every constraint is loaded");
                if parse(&c.expression).expect("parses").rhs.is_none() {
                    assert_eq!(
                        (&loaded.bound_lower, &loaded.bound_upper),
                        (&None, &None),
                        "{}: constraint {}",
                        case.display(),
                        c.id
                    );
                }
            }
        }
    }

    mod lowering {
        use super::*;
        use novomodelo_core::Line;
        use proptest::prelude::{Just, ProptestConfig, Strategy, prop_assert_eq, proptest};
        use std::collections::BTreeSet;

        /// Every parameter a case reads by `@name`; the largest id is 7, so the data ids start
        /// at 8.
        fn params() -> HashMap<String, EntityId> {
            [
                ("margin_b", 0),
                ("flow_a", 1),
                ("flow_b", 2),
                ("external_load", 3),
                ("small_plants", 4),
                ("local_gen", 5),
                ("margin_a", 6),
                ("cap", 7),
            ]
            .map(|(name, id)| (name.to_owned(), EntityId(id)))
            .into()
        }

        fn record(
            id: i32,
            expression: &str,
            active_when: Option<&str>,
            penalty: Option<f64>,
        ) -> String {
            let active_when = active_when.map_or_else(String::new, |condition| {
                format!(r#""active_when": "{condition}", "#)
            });
            let slack = penalty.map_or_else(
                || r#"{ "enabled": false }"#.to_owned(),
                |penalty| {
                    format!(
                        r#"{{ "enabled": true, "penalty": {penalty:?}, "penalty_basis": "per_hour" }}"#
                    )
                },
            );
            format!(
                r#"{{ "id": {id}, "name": "constraint_{id}", "expression": "{expression}", {active_when}"slack": {slack} }}"#
            )
        }

        fn constraint(id: i32, expression: &str) -> String {
            record(id, expression, None, None)
        }

        /// A file of `constraints` after the lists of `extra`, which ends in a comma.
        fn with_constraints(extra: &str, constraints: &[String]) -> String {
            format!(
                r#"{{ {extra} "constraints": [{}] }}"#,
                constraints.join(",")
            )
        }

        fn with_expressions(expressions: &[&str], constraints: &[String]) -> String {
            with_constraints(
                &format!(r#""expressions": [{}],"#, expressions.join(",")),
                constraints,
            )
        }

        fn single(expression: &str) -> String {
            with_constraints("", &[constraint(701, expression)])
        }

        fn load_with_lines(
            json: &str,
            lines: &[Line],
        ) -> Result<LoadedGenericConstraints, LoadError> {
            let file = write_json(json);
            parse_generic_constraints_with_provenance(
                file.path(),
                &params(),
                &build_line_bus_pair_index(lines).expect("distinct line pairs"),
            )
        }

        fn loaded(json: &str) -> LoadedGenericConstraints {
            load_with_lines(json, &[]).unwrap_or_else(|e| panic!("{e:?}"))
        }

        fn fault_with_lines(json: &str, lines: &[Line]) -> (String, String) {
            match load_with_lines(json, lines).err() {
                Some(LoadError::SchemaError { field, message, .. }) => (field, message),
                other => panic!("expected a SchemaError, got: {other:?}"),
            }
        }

        fn fault(json: &str) -> (String, String) {
            fault_with_lines(json, &[])
        }

        fn literal(value: f64) -> ScalarExpr {
            ScalarExpr::literal(value)
        }

        fn param(id: i32) -> ScalarExpr {
            ScalarExpr::param(EntityId(id))
        }

        fn compare(lhs: ScalarExpr, op: CompareOp, rhs: ScalarExpr) -> ResolvedCondition {
            ResolvedCondition::Compare { lhs, op, rhs }
        }

        fn hydro_14() -> VariableRef {
            VariableRef::HydroGeneration {
                hydro_id: EntityId(14),
                block_id: None,
                bus_id: None,
            }
        }

        fn capacity_of(hydro: i32, block_id: Option<usize>) -> AvailabilityAtom {
            AvailabilityAtom {
                hydro_id: EntityId(hydro),
                block_id,
            }
        }

        fn only_shift(scenario: &ScenarioLayer, id: i32) -> &ScalarExpr {
            let [
                ConstraintScenario {
                    constraint_id,
                    active_when: None,
                    endpoint_shift: Some(shift),
                },
            ] = scenario.constraints.as_slice()
            else {
                panic!("expected one shift, got: {:?}", scenario.constraints);
            };
            assert_eq!(*constraint_id, EntityId(id));
            shift
        }

        /// derived: `2 * hydro_availability` moves to each bound the operator sets as
        /// `10.0 - 2.0 * p8`, which at a capacity of 500 is `10.0 - 2.0 * 500.0`.
        #[test]
        fn each_bound_an_operator_sets_takes_the_capacity_with_its_coefficient() {
            let bits = (10.0_f64 - 2.0 * 500.0).to_bits();
            for (operator, lower, upper) in [
                (">=", Some(bits), None),
                ("<=", None, Some(bits)),
                ("==", Some(bits), Some(bits)),
            ] {
                let LoadedGenericConstraints {
                    constraints,
                    scenario,
                    ..
                } = loaded(&single(&format!(
                    "2*hydro_availability(14) + hydro_generation(14) {operator} 10"
                )));
                assert_eq!(
                    constraints[0].expression.terms,
                    [LinearTerm::literal(1.0, hydro_14())],
                    "{operator}"
                );
                assert_eq!(
                    scenario.availability_atoms,
                    [(EntityId(8), capacity_of(14, None))],
                    "{operator}"
                );
                let at_capacity = |bound: &Option<ScalarExpr>| {
                    bound
                        .as_ref()
                        .map(|expr| eval(expr, &[(8, 500.0)]).to_bits())
                };
                assert_eq!(
                    at_capacity(&constraints[0].bound_lower),
                    lower,
                    "{operator}"
                );
                assert_eq!(
                    at_capacity(&constraints[0].bound_upper),
                    upper,
                    "{operator}"
                );
            }
        }

        /// derived: with `cap` at id 7 the capacity without a block is id 8 and the one of
        /// block 2 is id 9, so the bound is `0.0 - p8 - p9`, which at 100 and 20 is exactly
        /// -120 in either order.
        #[test]
        fn a_block_selector_gives_a_capacity_of_its_own() {
            let LoadedGenericConstraints {
                constraints,
                scenario,
                ..
            } = loaded(&single(
                "hydro_availability(14) + hydro_availability(14, 2) - hydro_generation(14) >= 0",
            ));
            assert_eq!(
                constraints[0].expression.terms,
                [LinearTerm::literal(-1.0, hydro_14())]
            );
            assert_eq!(
                scenario.availability_atoms,
                [
                    (EntityId(8), capacity_of(14, None)),
                    (EntityId(9), capacity_of(14, Some(2))),
                ]
            );
            let lower = constraints[0].bound_lower.as_ref().expect("a lower bound");
            assert_eq!(
                eval(lower, &[(8, 100.0), (9, 20.0)]).to_bits(),
                (-120.0_f64).to_bits()
            );
            assert_eq!(constraints[0].bound_upper, None);
        }

        /// derived: `cap` is id 7, the load of bus 3 the condition reads is data id 8,
        /// the capacity of plant 14 is id 9 and the selector is id 10; the bound is
        /// `(0.0 + p10) - 1.0 * p9`.
        #[test]
        fn selectors_are_numbered_after_the_capacities() {
            let LoadedGenericConstraints {
                constraints,
                scenario,
                ..
            } = loaded(&single(
                "hydro_availability(14) - hydro_generation(14) \
                 >= if(bus_load_value(3) <= 85000, 100, 200)",
            ));
            assert_eq!(
                scenario.data_atoms,
                [(
                    EntityId(8),
                    DataAtom::BusLoadValue {
                        bus_id: EntityId(3),
                        block_id: None,
                    }
                )]
            );
            assert_eq!(
                scenario.availability_atoms,
                [(EntityId(9), capacity_of(14, None))]
            );
            assert_eq!(
                constraints[0].bound_lower,
                Some((literal(0.0) + param(10)) - literal(1.0) * param(9))
            );
            let ids: Vec<i32> = scenario.selectors.iter().map(|(id, _)| id.0).collect();
            assert_eq!(ids, [10]);
        }

        #[test]
        fn a_capacity_read_through_a_named_expression_is_folded_where_it_is_referenced() {
            let extra = r#""expressions": [
                { "name": "RES", "expression": "hydro_availability(14) - hydro_generation(14)" }
            ],"#;
            let named = loaded(&with_constraints(extra, &[constraint(701, "@RES >= 0")]));
            let written = loaded(&single(
                "hydro_availability(14) - hydro_generation(14) >= 0",
            ));
            assert_eq!(named.constraints, written.constraints);
            assert_eq!(named.scenario, written.scenario);
            assert_eq!(
                named.constraints[0].expression.terms,
                [LinearTerm::literal(-1.0, hydro_14())]
            );
            assert_eq!(named.scenario.availability_atoms.len(), 1);
        }

        /// `TOTAL` reads `FIRST` and `SECOND`, which are declared after it.
        #[test]
        fn a_named_expression_is_inlined_to_the_sum_of_its_forms() {
            let extra = r#""expressions": [
                { "name": "TOTAL", "expression": "@FIRST + @SECOND" },
                { "name": "SECOND", "expression": "thermal_generation(2)" },
                { "name": "FIRST", "expression": "hydro_generation(14)" }
            ],"#;
            let named = loaded(&with_constraints(
                extra,
                &[constraint(701, "@TOTAL <= 100")],
            ));
            let written = loaded(&single(
                "hydro_generation(14) + thermal_generation(2) <= 100",
            ));
            assert_eq!(named.constraints, written.constraints);
            assert_eq!(named.scenario, written.scenario);
            assert_eq!(named.constraints[0].expression.terms.len(), 2);
        }

        #[test]
        fn a_line_flow_without_a_line_is_a_fault_of_its_field() {
            let (field, message) = fault_with_lines(
                &single("line_flow(source_bus=3, target_bus=9) >= 0"),
                &[make_line(5, 3, 7)],
            );
            assert_eq!(field, "constraints[0].expression");
            assert!(
                message.contains("no line connects buses source_bus=3 and target_bus=9"),
                "{message}"
            );
        }

        #[test]
        fn a_rejected_expression_is_a_fault_of_its_field() {
            for (src, expected) in [
                (
                    "thermal_generation(2, bus=2)",
                    "does not accept a bus selector",
                ),
                (
                    "thermal_generation(2, unit_group=1)",
                    "does not accept a unit-group selector",
                ),
                ("foo(1)", "unknown variable name"),
                (
                    "if(hydro_generation(14) <= 1, 1, 2) + hydro_generation(14)",
                    "condition may read only data",
                ),
                ("@missing + hydro_generation(14)", "unknown name"),
            ] {
                let (field, message) = fault(&single(&format!("{src} >= 0")));
                assert_eq!(field, "constraints[0].expression", "{src}");
                assert!(message.contains(expected), "{src}: {message}");
            }
        }

        /// A removed load name in an expression, an `if()` condition or `active_when` is refused
        /// in its field with the message an unknown name with the same arguments gets there.
        #[test]
        fn a_constraint_naming_a_removed_load_term_is_rejected() {
            let positions = |call: &str| {
                [
                    (
                        record(
                            701,
                            &format!("hydro_generation(14) + {call} <= 10"),
                            None,
                            None,
                        ),
                        "constraints[0].expression",
                    ),
                    (
                        record(
                            701,
                            &format!("hydro_generation(14) <= if({call} > 1, 10, 20)"),
                            None,
                            None,
                        ),
                        "constraints[0].expression",
                    ),
                    (
                        record(
                            701,
                            "hydro_generation(14) <= 10",
                            Some(&format!("{call} > 1")),
                            None,
                        ),
                        "constraints[0].active_when",
                    ),
                ]
            };
            for (removed, args) in [
                ("bus_load", "0"),
                ("system_load", ""),
                ("system_load_value", ""),
            ] {
                let named = positions(&format!("{removed}({args})"));
                let unknown = positions(&format!("no_such_term({args})"));
                for ((json, field), (twin, _)) in named.into_iter().zip(unknown) {
                    let (fault_field, message) = fault(&with_constraints("", &[json]));
                    let (_, twin_message) = fault(&with_constraints("", &[twin]));
                    assert_eq!(fault_field, field, "{removed}");
                    assert_eq!(
                        message,
                        twin_message.replace("no_such_term", removed),
                        "{removed}"
                    );
                }
            }
        }

        #[test]
        fn an_expression_that_does_not_parse_is_a_fault_of_its_field() {
            for src in ["hydro_generation(14 >= 0", "hydro_generation(14"] {
                let (field, message) = fault(&single(src));
                assert_eq!(field, "constraints[0].expression", "{src}");
                let diagnostic = parse(src).expect_err("the expression does not parse");
                assert_eq!(message, diagnostic.render(src), "{src}");
            }
        }

        /// derived: with `cap` at id 7 the load of bus 3 is data id 8 and the parameters
        /// `external_load` and `small_plants` are ids 3 and 4, so the data part is
        /// `100 + 20 - 0.5 * 200 = 20`, and the shift, its negation, is -20.
        #[test]
        fn every_kind_of_data_term_in_a_formula_joins_the_endpoint_shift() {
            let LoadedGenericConstraints {
                constraints,
                scenario,
                ..
            } = loaded(&with_constraints(
                "",
                &[constraint(
                    406,
                    "hydro_generation(14) + @external_load + @small_plants \
                     - 0.5*bus_load_value(3)",
                )],
            ));
            assert_eq!(
                constraints[0].expression.terms,
                [LinearTerm::literal(1.0, hydro_14())]
            );
            assert_eq!(
                (&constraints[0].bound_lower, &constraints[0].bound_upper),
                (&None, &None)
            );
            assert_eq!(
                scenario.data_atoms,
                [(
                    EntityId(8),
                    DataAtom::BusLoadValue {
                        bus_id: EntityId(3),
                        block_id: None,
                    }
                )]
            );
            let shift = only_shift(&scenario, 406);
            assert_eq!(
                shift.params().collect::<BTreeSet<_>>(),
                BTreeSet::from([EntityId(3), EntityId(4), EntityId(8)])
            );
            assert_eq!(
                eval(shift, &[(8, 200.0), (3, 100.0), (4, 20.0)]).to_bits(),
                (-20.0_f64).to_bits()
            );
        }

        /// derived: `cap` is id 7, the load the condition reads is data id 8 and the selector is
        /// id 9, so the shift is `-p9`, -100 where the selector is 100.
        #[test]
        fn a_selector_in_a_formula_joins_the_endpoint_shift() {
            let LoadedGenericConstraints {
                constraints,
                scenario,
                ..
            } = loaded(&with_constraints(
                "",
                &[constraint(
                    407,
                    "if(bus_load_value(3) <= 85000, 100, 200) + hydro_generation(14)",
                )],
            ));
            assert_eq!(
                constraints[0].expression.terms,
                [LinearTerm::literal(1.0, hydro_14())]
            );
            assert_eq!(
                scenario.data_atoms,
                [(
                    EntityId(8),
                    DataAtom::BusLoadValue {
                        bus_id: EntityId(3),
                        block_id: None,
                    }
                )]
            );
            let ids: Vec<i32> = scenario.selectors.iter().map(|(id, _)| id.0).collect();
            assert_eq!(ids, [9]);
            let shift = only_shift(&scenario, 407);
            assert_eq!(params_of(shift), [EntityId(9)]);
            assert_eq!(eval(shift, &[(9, 100.0)]).to_bits(), (-100.0_f64).to_bits());
        }

        /// 4021 and 4022 are two segments of one split constraint; no constraint has an `if()`,
        /// so the loaded model is equal as a whole.
        #[test]
        fn declaration_order_does_not_change_the_loaded_constraints() {
            let constraints = [
                constraint(
                    4021,
                    "hydro_availability(14) - hydro_generation(14) >= 0.04*bus_load_value(7)",
                ),
                constraint(
                    4022,
                    "hydro_availability(14) - hydro_generation(14) \
                     >= 0.05*(bus_load_value(7) + bus_load_value(3))",
                ),
                constraint(
                    701,
                    "hydro_generation(14) + bus_load_value(3) <= 10200 - 0.5*bus_load_value(7, 2)",
                ),
                constraint(
                    405,
                    "thermal_generation(2) - non_controllable_generation(3)",
                ),
            ];
            let build =
                |order: [usize; 4]| with_constraints("", &order.map(|at| constraints[at].clone()));
            let declared = build([0, 1, 2, 3]);
            let expected = loaded(&declared);
            assert_eq!(expected.constraints.len(), 4);
            assert_eq!(expected.scenario.data_atoms.len(), 3);
            assert_eq!(expected.scenario.availability_atoms.len(), 1);

            for order in [[3, 0, 2, 1], [1, 0, 3, 2], [2, 3, 1, 0]] {
                let other = build(order);
                assert_ne!(other, declared);
                let resolved = loaded(&other);
                assert_eq!(resolved.constraints, expected.constraints, "{order:?}");
                assert_eq!(resolved.scenario, expected.scenario, "{order:?}");
            }
        }

        /// derived: `cap` is id 7, so the one data atom, the load of bus 3 that the
        /// condition reads, is id 8 and the selector is id 9; the bound is
        /// `0.0 + p9` and the selector reads the load, p8, against 85000.
        #[test]
        fn a_conditional_bound_resolves_to_a_selector_over_a_data_atom() {
            let LoadedGenericConstraints {
                constraints,
                scenario,
                ..
            } = loaded(&single(
                "hydro_generation(14) <= if(bus_load_value(3) <= 85000, 100, 200)",
            ));
            assert_eq!(
                scenario.data_atoms,
                [(
                    EntityId(8),
                    DataAtom::BusLoadValue {
                        bus_id: EntityId(3),
                        block_id: None,
                    }
                )]
            );
            assert_eq!(constraints[0].bound_upper, Some(literal(0.0) + param(9)));
            assert_eq!(constraints[0].bound_lower, None);
            assert_eq!(
                scenario.selectors,
                [(
                    EntityId(9),
                    ConditionalScalar {
                        cond: compare(param(8), CompareOp::Le, literal(85000.0)),
                        then: literal(100.0),
                        else_: literal(200.0),
                    }
                )]
            );
        }

        #[test]
        fn conditional_branches_share_their_variables() {
            let LoadedGenericConstraints {
                constraints,
                scenario,
                ..
            } = loaded(&single(
                "if(bus_load_value(3) <= 1, hydro_generation(14) + 1, hydro_generation(14) + 2) >= 0",
            ));
            assert_eq!(
                constraints[0].expression.terms,
                [LinearTerm::literal(1.0, hydro_14())]
            );
            let [(_, selector)] = scenario.selectors.as_slice() else {
                panic!("expected one selector, got: {:?}", scenario.selectors);
            };
            assert_eq!(selector.then, literal(1.0));
            assert_eq!(selector.else_, literal(2.0));

            let (field, message) = fault(&single(
                "if(bus_load_value(3) <= 1, hydro_generation(14), thermal_generation(2)) >= 0",
            ));
            assert_eq!(field, "constraints[0].expression");
            assert!(
                message.contains("must read the same LP variables"),
                "{message}"
            );
        }

        /// derived: `cap` is id 7, so the load of bus 3 in block 2 is data id 8 and that of
        /// bus 7 is id 9; `local_gen` is parameter 5.
        #[test]
        fn data_atoms_of_conditions_and_coefficients_are_listed() {
            let LoadedGenericConstraints {
                constraints,
                scenario,
                ..
            } = loaded(&with_constraints(
                "",
                &[
                    constraint(
                        701,
                        "hydro_generation(14) <= if(@local_gen <= 3000 & bus_load_value(3, 2) > 5, 1, 2)",
                    ),
                    constraint(702, "bus_load_value(7) * thermal_generation(2) <= 1"),
                ],
            ));
            assert_eq!(
                scenario.data_atoms,
                [
                    (
                        EntityId(8),
                        DataAtom::BusLoadValue {
                            bus_id: EntityId(3),
                            block_id: Some(2),
                        }
                    ),
                    (
                        EntityId(9),
                        DataAtom::BusLoadValue {
                            bus_id: EntityId(7),
                            block_id: None,
                        }
                    ),
                ]
            );
            assert_eq!(
                scenario.selectors[0].1.cond,
                ResolvedCondition::And(
                    Box::new(compare(param(5), CompareOp::Le, literal(3000.0))),
                    Box::new(compare(param(8), CompareOp::Gt, literal(5.0))),
                )
            );
            assert_eq!(constraints[1].id, EntityId(702));
            assert_eq!(constraints[1].expression.terms[0].coefficient, param(9));
        }

        #[test]
        fn a_reference_cycle_through_data_only_expressions_is_one_diagnostic() {
            let extra = r#""expressions": [
                { "name": "B", "expression": "@A + bus_load_value(3)" },
                { "name": "A", "expression": "@B - 1" }
            ],"#;
            assert_eq!(
                fault(&with_constraints(
                    extra,
                    &[constraint(701, "hydro_generation(14) <= @A")]
                )),
                (
                    "expressions".to_owned(),
                    "named-expression reference cycle detected: B -> A -> B".to_owned()
                )
            );
        }

        /// derived: `FLOW` reads the data-only `NET` and the condition compares it, so the file
        /// loads to what it loads to with the body written in parentheses in both places, and
        /// the load of bus 3 stays a data atom.
        #[test]
        fn a_data_only_expression_resolves_as_its_body_written_in_place() {
            let body = "bus_load_value(3) - @margin_a - @margin_b";
            let with = |declared: &str, read: &str, condition: &str| {
                let flow = format!(
                    "@GEN + @small_plants - {read} + @local_gen \
                     - line_flow(source_bus=3, target_bus=7) + line_flow(source_bus=7, target_bus=3)"
                );
                let extra = format!(
                    r#""expressions": [{declared}
                         {{ "name": "GEN", "expression": "hydro_generation(14) + thermal_generation(2)" }},
                         {{ "name": "FLOW", "expression": "{flow}" }}],"#
                );
                let json = with_constraints(
                    &extra,
                    &[record(701, "@FLOW <= 1000", Some(condition), None)],
                );
                load_with_lines(&json, &[make_line(5, 3, 7)]).unwrap_or_else(|e| panic!("{e:?}"))
            };
            let named = with(
                &format!(r#"{{ "name": "NET", "expression": "{body}" }},"#),
                "@NET",
                "@NET > 0",
            );
            let inline = with("", &format!("({body})"), &format!("{body} > 0"));
            assert_eq!(named.constraints, inline.constraints);
            assert_eq!(named.scenario, inline.scenario);
            assert!(matches!(
                named.scenario.constraints.as_slice(),
                [ConstraintScenario {
                    active_when: Some(_),
                    ..
                }]
            ));
            assert_eq!(named.constraints[0].expression.terms.len(), 4);
            assert!(named.scenario.data_atoms.iter().any(|(_, atom)| *atom
                == DataAtom::BusLoadValue {
                    bus_id: EntityId(3),
                    block_id: None,
                }));
        }

        #[test]
        fn an_unknown_name_in_a_condition_is_a_fault_of_its_field() {
            let (field, message) = fault(&with_constraints(
                "",
                &[record(
                    701,
                    "hydro_generation(14) <= 10",
                    Some("@missing > 1"),
                    None,
                )],
            ));
            assert_eq!(field, "constraints[0].active_when");
            assert!(
                message.contains("unknown scalar parameter \"@missing\""),
                "{message}"
            );
        }

        #[derive(Debug, PartialEq)]
        struct EvaluatedConstraint {
            id: EntityId,
            name: String,
            slack: Option<SlackConfig>,
            terms: Vec<(VariableRef, u64)>,
            lower: Option<u64>,
            upper: Option<u64>,
            shift: Option<u64>,
            active_when: Option<ResolvedCondition>,
            selectors: Vec<usize>,
        }

        /// A loaded file with every value as bits, each selector resolved by its own condition,
        /// and each selector id replaced by its position of first use, so that two files whose
        /// selector ids differ, as they do when the file order differs, compare equal.
        #[derive(Debug, PartialEq)]
        struct Evaluated {
            constraints: Vec<EvaluatedConstraint>,
            data_atoms: Vec<(EntityId, DataAtom)>,
            availability_atoms: Vec<(EntityId, AvailabilityAtom)>,
            selectors: Vec<ConditionalScalar>,
            entries: usize,
        }

        fn leaf(id: EntityId, layer: &ScenarioLayer) -> f64 {
            let Some((_, selector)) = layer.selectors.iter().find(|(own, _)| *own == id) else {
                return f64::from(id.0) * 10.0 + 1.0;
            };
            let value = |side: &ScalarExpr| side.evaluate(|leaf_id| leaf(leaf_id, layer));
            if selector.cond.holds(&mut |side| value(side)) {
                value(&selector.then)
            } else {
                value(&selector.else_)
            }
        }

        fn evaluated(loaded: &LoadedGenericConstraints) -> Evaluated {
            let layer = &loaded.scenario;
            let bits = |expr: &ScalarExpr| expr.evaluate(|id| leaf(id, layer)).to_bits();
            let entry = |id: EntityId| {
                layer
                    .constraints
                    .iter()
                    .find(|entry| entry.constraint_id == id)
            };
            let selector = |id: EntityId| {
                layer
                    .selectors
                    .iter()
                    .find(|(own, _)| *own == id)
                    .map(|(_, selector)| selector)
            };
            let mut first_use: Vec<EntityId> = Vec::new();
            let mut constraints = Vec::with_capacity(loaded.constraints.len());
            for gc in &loaded.constraints {
                let shift = entry(gc.id).and_then(|entry| entry.endpoint_shift.as_ref());
                let reads = gc
                    .expression
                    .terms
                    .iter()
                    .map(|term| &term.coefficient)
                    .chain(gc.bound_lower.as_ref())
                    .chain(gc.bound_upper.as_ref())
                    .chain(shift);
                let mut selectors = Vec::new();
                for id in reads.flat_map(ScalarExpr::params) {
                    if selector(id).is_none() {
                        continue;
                    }
                    let ordinal = first_use
                        .iter()
                        .position(|&seen| seen == id)
                        .unwrap_or_else(|| {
                            first_use.push(id);
                            first_use.len() - 1
                        });
                    selectors.push(ordinal);
                }
                constraints.push(EvaluatedConstraint {
                    id: gc.id,
                    name: gc.name.clone(),
                    slack: gc.slack,
                    terms: gc
                        .expression
                        .terms
                        .iter()
                        .map(|term| (term.variable, bits(&term.coefficient)))
                        .collect(),
                    lower: gc.bound_lower.as_ref().map(bits),
                    upper: gc.bound_upper.as_ref().map(bits),
                    shift: shift.map(bits),
                    active_when: entry(gc.id).and_then(|entry| entry.active_when.clone()),
                    selectors,
                });
            }
            assert_eq!(
                first_use.len(),
                layer.selectors.len(),
                "every selector is read"
            );
            Evaluated {
                constraints,
                data_atoms: layer.data_atoms.clone(),
                availability_atoms: layer.availability_atoms.clone(),
                selectors: first_use
                    .into_iter()
                    .filter_map(|id| selector(id).cloned())
                    .collect(),
                entries: layer.constraints.len(),
            }
        }

        /// 701, 7011 and 7012 are segments of one split constraint, the last two with a selector
        /// of their own, and 703 reads a third selector; `F701` is the expression of 701, and 702
        /// and 703 read it.
        #[test]
        fn declaration_order_does_not_change_the_evaluated_result_with_names_selectors_and_references()
         {
            let expressions = [
                r#"{ "name": "ZED", "expression": "hydro_generation(14) + thermal_generation(2)" }"#,
                r#"{ "name": "ABE", "expression": "@ZED - non_controllable_generation(3)" }"#,
                r#"{ "name": "F701", "expression": "@ZED + bus_load_value(3)" }"#,
            ];
            let constraints = [
                constraint(
                    703,
                    "@ABE + @F701 <= 0.5 * @margin_a \
                     + if(bus_load_value(7) <= 9, @external_load, @small_plants)",
                ),
                constraint(
                    701,
                    "@ZED + bus_load_value(3) >= bus_load_value(7) + @local_gen",
                ),
                constraint(7011, "@ZED <= if(@margin_a > 1, 5, 6)"),
                constraint(7012, "@ZED <= if(@margin_a > 2, 7, 8)"),
                constraint(702, "@F701 - hydro_generation(14)"),
            ];
            let build = |expressions: &[&str], order: [usize; 5]| {
                with_expressions(expressions, &order.map(|at| constraints[at].clone()))
            };
            let declared = build(&expressions, [1, 2, 3, 4, 0]);
            let expected = evaluated(&loaded(&declared));
            assert_eq!(expected.constraints.len(), 5);
            assert_eq!(expected.data_atoms.len(), 2);
            assert_eq!(expected.selectors.len(), 3);

            let reversed_expressions = [expressions[2], expressions[1], expressions[0]];
            for other in [
                build(&reversed_expressions, [0, 1, 2, 3, 4]),
                build(&expressions, [4, 0, 3, 2, 1]),
                build(&reversed_expressions, [0, 4, 3, 1, 2]),
            ] {
                assert_ne!(other, declared);
                assert_eq!(evaluated(&loaded(&other)), expected);
            }
        }

        /// The condition of 702, which reads the loads of buses 7 and 3, joined by `&` and `|`.
        const GATE_702: &str = "82000 < bus_load_value(7) & bus_load_value(7) <= 95000 \
            | bus_load_value(3) > 5";

        const GATE_701: &str = "bus_load_value(7) > 100 | bus_load_value(3) <= 50 | @DAT > 3";

        const SOFT_PENALTY: f64 = 5000.0;

        /// A formula that reads a load and a capacity, split into two segments (401, 4011); an
        /// inequality split into four segments, one hard and one soft without a selector (7011,
        /// 7012) and one hard and one soft with it (7013, 701); and an inequality that reads the
        /// parameter `margin_a`, held where `GATE_702` holds (702). `gate_701` holds every
        /// segment of 701.
        fn every_kind(gate_701: Option<&str>) -> Vec<String> {
            let with_selector = "hydro_generation(14) + bus_load_value(3) \
                <= if(bus_load_value(3) > 9000 & bus_load_value(7) <= 5, 10200, 9800)";
            let without_selector =
                "hydro_generation(14) + bus_load_value(3) <= 0.5*bus_load_value(3)";
            vec![
                record(
                    401,
                    "hydro_generation(14) + bus_load_value(3) - hydro_availability(14)",
                    None,
                    Some(SOFT_PENALTY),
                ),
                record(
                    4011,
                    "2*hydro_generation(14) - hydro_availability(14)",
                    None,
                    Some(SOFT_PENALTY),
                ),
                record(7011, without_selector, gate_701, Some(50.0)),
                record(7012, without_selector, gate_701, None),
                record(7013, with_selector, gate_701, None),
                record(701, with_selector, gate_701, Some(50.0)),
                record(
                    702,
                    "thermal_generation(2) - non_controllable_generation(3) \
                     >= bus_load_value(7) + @margin_a",
                    Some(GATE_702),
                    Some(SOFT_PENALTY),
                ),
            ]
        }

        #[test]
        fn the_loaded_model_round_trips_through_postcard() {
            let LoadedGenericConstraints {
                constraints,
                scenario,
                ..
            } = loaded(&with_constraints("", &every_kind(None)));
            let by_id = |id: i32| {
                constraints
                    .iter()
                    .find(|gc| gc.id == EntityId(id))
                    .expect("the constraint is loaded")
            };
            let entry = |id: i32| {
                scenario
                    .constraints
                    .iter()
                    .find(|entry| entry.constraint_id == EntityId(id))
            };
            assert!(entry(401).is_some_and(|entry| entry.endpoint_shift.is_some()));
            assert!(by_id(401).slack.is_some() && by_id(7013).slack.is_none());
            assert!(matches!(
                entry(702).and_then(|entry| entry.active_when.as_ref()),
                Some(ResolvedCondition::Or(..))
            ));
            assert!(
                scenario
                    .selectors
                    .iter()
                    .all(|(_, selector)| matches!(selector.cond, ResolvedCondition::And(..)))
            );
            assert_eq!(scenario.selectors.len(), 2);
            assert!(!scenario.data_atoms.is_empty() && !scenario.availability_atoms.is_empty());
            let parameter_read = by_id(702).bound_lower.as_ref().expect("a lower bound");
            assert!(parameter_read.params().any(|id| id == EntityId(6)));

            let bytes = postcard::to_allocvec(&(&constraints, &scenario)).expect("serializes");
            let decoded: (Vec<GenericConstraint>, ScenarioLayer) =
                postcard::from_bytes(&bytes).expect("deserializes");
            assert_eq!(decoded, (constraints, scenario));
            // `==` does not tell `0.0` from `-0.0`; the re-encoded bytes do.
            assert_eq!(postcard::to_allocvec(&decoded).expect("serializes"), bytes);
        }

        /// The two segments that read a selector, 7013 and 701, swap their selector ids when the
        /// constraints are reversed.
        #[test]
        fn the_loaded_model_is_invariant_to_constraint_order_up_to_selector_ids() {
            let declared = every_kind(None);
            let expected = evaluated(&loaded(&with_constraints("", &declared)));
            assert_eq!(expected.selectors.len(), 2);
            let mut rotated = declared.clone();
            rotated.rotate_right(1);
            let mut reversed = declared.clone();
            reversed.reverse();
            for other in [rotated, reversed] {
                assert_ne!(other, declared);
                assert_eq!(evaluated(&loaded(&with_constraints("", &other))), expected);
            }
        }

        /// `every_kind` with every segment of 701 held where `GATE_701` holds, a third segment of
        /// 401 (4012), two more segments of 701, one hard and one soft, with a selector of their
        /// own (7014, 7015), and 703, which reads named expressions and a selector. `DAT` reads
        /// no LP variable, and `ABE` and `GATE_701` read it; `F701` is the expression of 701,
        /// which 703 reads.
        fn order_reference() -> (Vec<String>, Vec<&'static str>) {
            let mut constraints = every_kind(Some(GATE_701));
            let lower_selector =
                "hydro_generation(14) + bus_load_value(3) >= if(bus_load_value(3) > 7, 1, 2)";
            constraints.extend([
                record(
                    4012,
                    "hydro_generation(14) - 0.5*hydro_availability(14)",
                    None,
                    Some(SOFT_PENALTY),
                ),
                record(7014, lower_selector, Some(GATE_701), None),
                record(7015, lower_selector, Some(GATE_701), Some(50.0)),
                record(
                    703,
                    "@ABE + @F701 <= 0.5 * @local_gen \
                     + if(bus_load_value(7) <= 9, @external_load, @small_plants)",
                    None,
                    Some(SOFT_PENALTY),
                ),
            ]);
            let expressions = vec![
                r#"{ "name": "ZED", "expression": "hydro_generation(14) + thermal_generation(2)" }"#,
                r#"{ "name": "ABE", "expression": "@ZED - non_controllable_generation(3) - @DAT" }"#,
                r#"{ "name": "DAT", "expression": "bus_load_value(7) - @local_gen" }"#,
                r#"{ "name": "F701", "expression": "hydro_generation(14) + bus_load_value(3)" }"#,
            ];
            (constraints, expressions)
        }

        fn pick<T: Clone>(items: &[T], order: &[usize]) -> Vec<T> {
            assert_eq!(items.len(), order.len(), "an order permutes its whole list");
            order.iter().map(|&at| items[at].clone()).collect()
        }

        fn shuffled(len: usize) -> impl Strategy<Value = Vec<usize>> {
            Just((0..len).collect::<Vec<usize>>()).prop_shuffle()
        }

        #[test]
        fn the_order_reference_permutes_every_list() {
            let (constraints, expressions) = order_reference();
            let reference = loaded(&with_expressions(&expressions, &constraints));
            assert_eq!(reference.constraints.len(), constraints.len());
            assert!(reference.scenario.selectors.len() >= 2);

            let reversed = |len: usize| (0..len).rev().collect::<Vec<_>>();
            assert!(constraints.len() >= 2);
            assert_ne!(
                pick(&constraints, &reversed(constraints.len())),
                constraints
            );
            assert!(expressions.len() >= 2);
            assert_ne!(
                pick(&expressions, &reversed(expressions.len())),
                expressions
            );
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            #[test]
            fn loading_is_invariant_to_any_declaration_order_up_to_selector_ids(
                constraint_order in shuffled(order_reference().0.len()),
                expression_order in shuffled(order_reference().1.len()),
            ) {
                let (constraints, expressions) = order_reference();
                let expected = evaluated(&loaded(&with_expressions(&expressions, &constraints)));
                let permuted = with_expressions(
                    &pick(&expressions, &expression_order),
                    &pick(&constraints, &constraint_order),
                );
                prop_assert_eq!(evaluated(&loaded(&permuted)), expected);
            }
        }
    }

    mod provenance {
        use super::*;

        #[test]
        fn parse_records_the_named_expressions_each_constraint_reaches() {
            let json = r#"{
  "expressions": [
    { "name": "f", "expression": "hydro_generation(10) + hydro_generation(11)" },
    { "name": "g", "expression": "2 * @f + thermal_generation(5)" }
  ],
  "constraints": [
    { "id": 0, "name": "c0", "expression": "hydro_generation(0) >= @g", "slack": { "enabled": false } },
    { "id": 1, "name": "c1", "expression": "hydro_generation(1) <= 5", "slack": { "enabled": false } }
  ]
}"#;
            let file = write_json(json);
            let LoadedGenericConstraints {
                constraints,
                named_sources,
                ..
            } = parse_generic_constraints_with_provenance(
                file.path(),
                &HashMap::new(),
                &LineBusPairIndex::default(),
            )
            .unwrap();

            let hydro = |id: i32| VariableRef::HydroGeneration {
                hydro_id: EntityId(id),
                block_id: None,
                bus_id: None,
            };
            let expected = BTreeMap::from([(
                EntityId(0),
                vec![
                    NamedSource {
                        name: "f".to_owned(),
                        variables: vec![hydro(10), hydro(11)],
                    },
                    NamedSource {
                        name: "g".to_owned(),
                        variables: vec![VariableRef::ThermalGeneration {
                            thermal_id: EntityId(5),
                            block_id: None,
                        }],
                    },
                ],
            )]);
            assert_eq!(named_sources, expected);
            assert_eq!(
                parse_generic_constraints(
                    file.path(),
                    &HashMap::new(),
                    &LineBusPairIndex::default()
                )
                .unwrap(),
                constraints
            );
        }
    }
}
