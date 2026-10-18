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
//!       "expression": "hydro_generation(0) + hydro_generation(1) <= @demanda_sin",
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
//! Deferred validations (not performed here):
//!
//! - Entity ID existence in entity registries — Layer 3.
//! - Block ID validity for the referenced stage — Layer 3/5.

use novomodelo_core::{
    ConstraintExpression, EntityId, GenericConstraint, LinearTerm, PenaltyBasis, SlackConfig,
};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::reference_cycles::detect_cycles;
use crate::LoadError;
use crate::expression::catalog::LineBusPairIndex;
use crate::expression::lower::{NamedTable, Scope, lower_relation};

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
    /// An optional inline relational operator (`<=`, `>=`, `==`) moves every variable term to
    /// the left and assigns the remainder to the bound it targets.
    expression: String,

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

    /// Linear expression over LP variables, under the same arithmetic as a constraint's
    /// `expression`, without a relational operator.
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
/// | Named expression: body rejected (syntax error, unknown name, not linear, no LP variable) | [`LoadError::SchemaError`] |
/// | Named expression: reference cycle              | [`LoadError::SchemaError`] |
/// | Constraint expression rejected (syntax error, unknown variable or `@name`, not linear) | [`LoadError::SchemaError`] |
///
/// Faults are reported in this order, each in file order: ids and slacks, named-expression
/// names, named-expression bodies, reference cycles, constraint expressions.
///
/// # Parameters
///
/// `name_to_id` maps parameter definition names to their [`EntityId`]. Pass
/// `&HashMap::new()` when no parameters have been loaded; expressions that
/// contain `@name` tokens will then fail with a schema error. The real mapping
/// is wired in by the caller once the parameter loader output is available.
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
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawGenericConstraintsFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw(&raw, path)?;

    let table = parse_named_expressions(&raw.expressions, path, name_to_id, line_index)?;
    detect_cycles(&table.reference_graph(), "named-expression").map_err(|message| {
        LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "expressions".to_string(),
            message,
        }
    })?;

    convert(raw, path, name_to_id, &table, line_index)
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

/// Convert the validated raw data to `Vec<GenericConstraint>`, sorted by `id`.
fn convert(
    raw: RawGenericConstraintsFile,
    path: &Path,
    name_to_id: &HashMap<String, EntityId>,
    table: &NamedTable,
    line_index: &LineBusPairIndex,
) -> Result<Vec<GenericConstraint>, LoadError> {
    let scope = Scope {
        params: name_to_id,
        line_index,
        named: table,
    };
    let mut result = Vec::with_capacity(raw.constraints.len());

    for (i, c) in raw.constraints.into_iter().enumerate() {
        let lowered =
            lower_relation(&c.expression, &scope).map_err(|diagnostic| LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("constraints[{i}].expression"),
                message: diagnostic.render(&c.expression),
            })?;

        let slack = resolve_slack(&c.slack, i, path)?;

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

        result.push(GenericConstraint {
            id: EntityId::from(c.id),
            name: c.name,
            description: c.description,
            expression,
            slack,
            bound_lower: lowered.lower,
            bound_upper: lowered.upper,
        });
    }

    result.sort_by_key(|gc| gc.id.0);

    Ok(result)
}

// ── Named-expression declarations ───────────────────────────────────────────

/// Check the `expressions` declarations' names, then parse, type and lower each
/// body into a [`NamedTable`]. Reference cycles are checked over the table
/// afterwards.
///
/// # Errors
///
/// [`LoadError::SchemaError`] — `field = "expressions[i].name"` for a duplicate or
/// parameter-colliding name; `field = "expressions[i].expression"` for the first
/// body, in file order, the engine rejects.
fn parse_named_expressions(
    entries: &[RawNamedExpression],
    path: &Path,
    name_to_id: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
) -> Result<NamedTable, LoadError> {
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

    let declarations: Vec<(&str, &str)> = entries
        .iter()
        .map(|entry| (entry.name.as_str(), entry.expression.as_str()))
        .collect();
    NamedTable::declare(&declarations, name_to_id, line_index).map_err(|(i, diagnostic)| {
        LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("expressions[{i}].expression"),
            message: diagnostic.render(&entries[i].expression),
        }
    })
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
    use crate::test_support::write_json;
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

    fn one_param_table() -> std::collections::HashMap<String, EntityId> {
        let mut m = std::collections::HashMap::new();
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
    fn ac1_inline_constant_and_variable_rhs_normalizes() {
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
    fn ac2_inline_lower_bound_bare_param_reference() {
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
    fn ac3_net_zero_column_drops_and_constant_bound_survives() {
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

    /// DECOMP-shaped form 1: a decision-variable RHS mixing two variable terms
    /// and a constant (`RSE <= -0.316*FBIPS + 0.5*GPC + 12352`).
    #[test]
    fn decomp_shaped_decision_variable_rhs() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "rse", "expression": "bus_deficit(0) <= -0.316 * bus_deficit(1) + 0.5 * bus_deficit(2) + 12352", "slack": { "enabled": false } }
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

    /// DECOMP-shaped form 2: affine parameters plus a constant, no decision
    /// variable on the RHS at all (`RNE <= 11000 - 0.04*val_demanda(3) +
    /// 0.06*ger_pee(11)`, authored with novomodelo's own `@name` scalar-parameter
    /// grammar).
    #[test]
    fn decomp_shaped_affine_param_rhs() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "rne", "expression": "bus_deficit(0) <= 11000 - 0.04 * @val_demanda_3 + 0.06 * @ger_pee_11", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let mut tbl = HashMap::new();
        tbl.insert("val_demanda_3".to_string(), EntityId(50));
        tbl.insert("ger_pee_11".to_string(), EntityId(51));
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

    /// DECOMP-shaped form 3: a pure-constant RHS (`FSIOR <= 73`).
    #[test]
    fn decomp_shaped_pure_constant_rhs() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "fsior", "expression": "bus_deficit(0) <= 73", "slack": { "enabled": false } }
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

    /// The retired `bound_upper_ref` JSON field is now an unknown field under
    /// `deny_unknown_fields` — a symbolic endpoint is authored inline instead.
    #[test]
    fn test_parse_retired_bound_upper_ref_field_is_unknown_field_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "a",
      "expression": "hydro_generation(0)",
      "slack": { "enabled": false },
      "bound_upper_ref": "demanda"
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for the retired \"bound_upper_ref\" field, got: {err:?}"
        );
    }

    /// A constraint declaring neither bound reference parses with both fields `None`
    /// (existing files are unaffected).
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
    fn parse_named_expressions_empty_is_ok() {
        let table = parse_named_expressions(
            &[],
            Path::new("generic_constraints.json"),
            &HashMap::new(),
            &LineBusPairIndex::default(),
        )
        .unwrap();
        assert!(table.reference_graph().is_empty());
    }

    /// Two distinct valid definitions ⇒ a two-entry table, each lowering to its own terms in
    /// authoring order.
    #[test]
    fn parse_named_expressions_two_valid_definitions() {
        let entries = [
            raw_named("fnese", "hydro_generation(0) + hydro_generation(1)"),
            raw_named("fns", "2.5 * thermal_generation(5) - hydro_generation(3)"),
        ];
        let params = HashMap::new();
        let line_index = LineBusPairIndex::default();
        let table = parse_named_expressions(
            &entries,
            Path::new("generic_constraints.json"),
            &params,
            &line_index,
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
    fn parse_named_expressions_duplicate_name_is_error() {
        let entries = [
            raw_named("fnese", "hydro_generation(0)"),
            raw_named("fnese", "hydro_generation(1)"),
        ];
        let err = parse_named_expressions(
            &entries,
            Path::new("generic_constraints.json"),
            &HashMap::new(),
            &LineBusPairIndex::default(),
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
    fn parse_named_expressions_param_collision_is_error() {
        let entries = [raw_named("rho", "hydro_generation(0)")];
        let err = parse_named_expressions(
            &entries,
            Path::new("generic_constraints.json"),
            &one_param_table(),
            &LineBusPairIndex::default(),
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
}
