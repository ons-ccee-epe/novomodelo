//! Inlining of named-expression references into a flat term list.
//!
//! The term parser in the `generic` module produces a [`ParsedExpression`]: a
//! term list that may still carry unresolved `@name` references
//! ([`ParsedTerm::Ref`]) alongside already-resolved [`ParsedTerm::Flat`] terms.
//! [`inline`] substitutes each reference with the referenced expression's terms,
//! multiplying the reference's `scale` into each substituted term's `scale`; the
//! coefficient kind is preserved (a `Parameter` coefficient stays a `Parameter`,
//! whose runtime value cannot be folded into a literal). Composition — an
//! expression referencing another expression — resolves transitively.
//!
//! [`detect_cycles`] rejects a reference cycle before [`inline`] runs, so the
//! recursion in [`inline`] terminates. It is an iterative three-colour DFS, never
//! recursion over the chain, so a deep dependency chain cannot overflow the stack.

use cobre_core::LinearTerm;
use std::collections::{HashMap, HashSet};

/// Hard cap on terms one [`inline`] call may materialize — aborts an exponential
/// reference blowup (a doubling chain `@e_k = @e_{k-1} + @e_{k-1}`) fast. Set far
/// above any legitimate authored expression (tens of expressions × tens of terms).
const MAX_INLINED_TERMS: usize = 100_000;

/// One term of a parsed expression before reference substitution.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ParsedTerm {
    /// A term with no unresolved reference — maps 1:1 to a [`LinearTerm`].
    Flat(LinearTerm),
    /// A reference to a named expression, scaled by `scale`.
    Ref {
        /// Referenced expression name (the identifier after `@`).
        name: String,
        /// Literal multiplier distributed into every substituted term's `scale`.
        scale: f64,
    },
}

/// A parsed expression: an ordered term list that may carry [`ParsedTerm::Ref`]s.
pub(crate) type ParsedExpression = Vec<ParsedTerm>;

/// Flatten `parsed` into a [`LinearTerm`] list, substituting every reference with
/// the referenced expression's inlined terms (scale distributed, coefficient kind
/// preserved). Duplicate variables are left un-merged, in substitution order.
///
/// Call [`detect_cycles`] on `table` first; this substitution recurses over the
/// reference graph and relies on it being acyclic.
///
/// # Errors
///
/// Returns `Err(message)` naming the missing name when a reference resolves to no
/// declaration in `table`, or when expansion would exceed [`MAX_INLINED_TERMS`]
/// (an exponential reference blowup, caught before the full vector materializes).
pub(crate) fn inline(
    parsed: &ParsedExpression,
    table: &[(String, ParsedExpression)],
) -> Result<Vec<LinearTerm>, String> {
    let index: HashMap<&str, &ParsedExpression> = table
        .iter()
        .map(|(name, expr)| (name.as_str(), expr))
        .collect();
    let mut out = Vec::new();
    inline_into(parsed, &index, &mut out, None)?;
    Ok(out)
}

/// Check that every `@name` reference across `table` resolves to a declared entry,
/// without materializing any terms — the cheap counterpart to [`inline`], run at
/// declaration time so declaring an expensive (e.g. exponentially-expanding)
/// expression that no constraint references never forces its expansion.
///
/// # Errors
///
/// Returns `Err((index, message))` for the first declaration (in `table` order,
/// terms in order) that references an undeclared name; `message` is identical to
/// [`inline`]'s undeclared-reference wording.
pub(crate) fn validate_references_resolve(
    table: &[(String, ParsedExpression)],
) -> Result<(), (usize, String)> {
    let declared: HashSet<&str> = table.iter().map(|(name, _)| name.as_str()).collect();
    for (i, (_name, parsed)) in table.iter().enumerate() {
        for term in parsed {
            if let ParsedTerm::Ref { name, .. } = term
                && !declared.contains(name.as_str())
            {
                return Err((i, undeclared_reference_message(name)));
            }
        }
    }
    Ok(())
}

/// Append `parsed`'s flattened terms to `out`, resolving references against `index`.
/// `origin` names the outermost reference under expansion, used only to attribute a
/// [`MAX_INLINED_TERMS`] overflow; the entry call passes `None`.
fn inline_into(
    parsed: &ParsedExpression,
    index: &HashMap<&str, &ParsedExpression>,
    out: &mut Vec<LinearTerm>,
    origin: Option<&str>,
) -> Result<(), String> {
    for term in parsed {
        match term {
            ParsedTerm::Flat(lt) => {
                if out.len() >= MAX_INLINED_TERMS {
                    return Err(inline_budget_exceeded(origin));
                }
                out.push(lt.clone());
            }
            ParsedTerm::Ref { name, scale } => {
                let referenced = index
                    .get(name.as_str())
                    .ok_or_else(|| undeclared_reference_message(name))?;
                let start = out.len();
                inline_into(referenced, index, out, origin.or(Some(name)))?;
                for lt in &mut out[start..] {
                    lt.scale *= scale;
                }
            }
        }
    }
    Ok(())
}

/// The undeclared-reference message, shared by [`inline`] and
/// [`validate_references_resolve`] so their wording never drifts.
fn undeclared_reference_message(name: &str) -> String {
    format!(
        "undeclared named-expression reference \"@{name}\": no expression with this name was declared"
    )
}

/// The [`MAX_INLINED_TERMS`] overflow message, naming the outermost reference when known.
fn inline_budget_exceeded(origin: Option<&str>) -> String {
    match origin {
        Some(name) => format!(
            "named-expression reference \"@{name}\" expands to more than {MAX_INLINED_TERMS} \
             inlined terms; this indicates an exponential reference pattern \
             (e.g. a doubling chain \"@e = @prev + @prev\")"
        ),
        None => {
            format!("expression expands to more than {MAX_INLINED_TERMS} inlined terms")
        }
    }
}

/// Three-colour DFS marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Color {
    Unvisited,
    InProgress,
    Done,
}

/// Reject a reference cycle over the expression dependency graph.
///
/// Iterative three-colour DFS: never recurses over the chain, so a deep chain
/// cannot overflow the stack. A self-reference is a cycle of length 1.
///
/// # Errors
///
/// Returns `Err(message)` listing the full cycle path (e.g. `a -> b -> a`) when a
/// reference cycle exists.
pub(crate) fn detect_cycles(table: &[(String, ParsedExpression)]) -> Result<(), String> {
    let adjacency: HashMap<&str, Vec<&str>> = table
        .iter()
        .map(|(name, expr)| {
            let refs = expr
                .iter()
                .filter_map(|t| match t {
                    ParsedTerm::Ref { name, .. } => Some(name.as_str()),
                    ParsedTerm::Flat(_) => None,
                })
                .collect();
            (name.as_str(), refs)
        })
        .collect();

    let mut color: HashMap<&str, Color> = table
        .iter()
        .map(|(name, _)| (name.as_str(), Color::Unvisited))
        .collect();

    let mut stack: Vec<(&str, usize)> = Vec::new();
    let mut path: Vec<&str> = Vec::new();

    for (start, _) in table {
        let start = start.as_str();
        if color.get(start).copied() != Some(Color::Unvisited) {
            continue;
        }
        color.insert(start, Color::InProgress);
        path.push(start);
        stack.push((start, 0));

        while let Some((node, child_idx)) = stack.last().copied() {
            let neighbors: &[&str] = adjacency.get(node).map_or(&[], Vec::as_slice);
            if child_idx < neighbors.len() {
                if let Some(frame) = stack.last_mut() {
                    frame.1 += 1;
                }
                let child = neighbors[child_idx];
                match color.get(child).copied() {
                    Some(Color::InProgress) => return Err(cycle_message(&path, child)),
                    Some(Color::Unvisited) => {
                        color.insert(child, Color::InProgress);
                        path.push(child);
                        stack.push((child, 0));
                    }
                    // A `Done` node is fully explored; an unknown child names no
                    // declaration (left to `inline` to report as undeclared).
                    Some(Color::Done) | None => {}
                }
            } else {
                color.insert(node, Color::Done);
                path.pop();
                stack.pop();
            }
        }
    }

    Ok(())
}

/// Render the cycle path closing back to `back_to`, which is on the current path.
fn cycle_message(path: &[&str], back_to: &str) -> String {
    let pos = path.iter().position(|&n| n == back_to).unwrap_or(0);
    let mut cycle: Vec<&str> = path[pos..].to_vec();
    cycle.push(back_to);
    format!(
        "named-expression reference cycle detected: {}",
        cycle.join(" -> ")
    )
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use cobre_core::{CoefficientRef, EntityId, VariableRef};

    fn hg(id: i32) -> VariableRef {
        VariableRef::HydroGeneration {
            hydro_id: EntityId(id),
            block_id: None,
            bus_id: None,
        }
    }

    fn flat_lit(coef: f64, id: i32) -> ParsedTerm {
        ParsedTerm::Flat(LinearTerm::literal(coef, hg(id)))
    }

    fn effective(term: &LinearTerm) -> f64 {
        match term.coefficient {
            CoefficientRef::Literal(v) => v * term.scale,
            CoefficientRef::Parameter(_) => panic!("expected a literal coefficient"),
        }
    }

    #[test]
    fn inline_reference_free_passthrough() {
        let parsed = vec![flat_lit(1.0, 0), flat_lit(-1.0, 1)];
        let terms = inline(&parsed, &[]).unwrap();
        assert_eq!(terms.len(), 2);
        assert!((effective(&terms[0]) - 1.0).abs() < f64::EPSILON);
        assert!((effective(&terms[1]) - (-1.0)).abs() < f64::EPSILON);
        assert_eq!(terms[0].variable, hg(0));
        assert_eq!(terms[1].variable, hg(1));
    }

    #[test]
    fn inline_single_reference_distributes_scale() {
        let table = vec![(
            "fnese".to_string(),
            vec![flat_lit(1.0, 0), flat_lit(1.0, 1)],
        )];
        let parsed = vec![ParsedTerm::Ref {
            name: "fnese".to_string(),
            scale: 2.0,
        }];
        let terms = inline(&parsed, &table).unwrap();
        assert_eq!(terms.len(), 2);
        assert!((effective(&terms[0]) - 2.0).abs() < f64::EPSILON);
        assert!((effective(&terms[1]) - 2.0).abs() < f64::EPSILON);
        assert_eq!(terms[0].variable, hg(0));
        assert_eq!(terms[1].variable, hg(1));
    }

    #[test]
    fn inline_negated_reference() {
        let table = vec![("f".to_string(), vec![flat_lit(1.0, 0)])];
        let parsed = vec![ParsedTerm::Ref {
            name: "f".to_string(),
            scale: -1.0,
        }];
        let terms = inline(&parsed, &table).unwrap();
        assert_eq!(terms.len(), 1);
        assert!((effective(&terms[0]) - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn inline_composition_nested() {
        let table = vec![
            (
                "inner".to_string(),
                vec![ParsedTerm::Ref {
                    name: "base".to_string(),
                    scale: 3.0,
                }],
            ),
            ("base".to_string(), vec![flat_lit(1.0, 0)]),
        ];
        let parsed = vec![ParsedTerm::Ref {
            name: "inner".to_string(),
            scale: 2.0,
        }];
        let terms = inline(&parsed, &table).unwrap();
        assert_eq!(terms.len(), 1);
        assert!((effective(&terms[0]) - 6.0).abs() < f64::EPSILON);
        assert_eq!(terms[0].variable, hg(0));
    }

    #[test]
    fn inline_preserves_parameter_coefficient() {
        let table = vec![(
            "p".to_string(),
            vec![ParsedTerm::Flat(LinearTerm::parameter(
                EntityId(7),
                1.0,
                hg(0),
            ))],
        )];
        let parsed = vec![ParsedTerm::Ref {
            name: "p".to_string(),
            scale: 2.0,
        }];
        let terms = inline(&parsed, &table).unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].coefficient, CoefficientRef::Parameter(EntityId(7)));
        assert!((terms[0].scale - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn inline_does_not_merge_duplicate_variables() {
        let table = vec![("f".to_string(), vec![flat_lit(1.0, 0), flat_lit(1.0, 0)])];
        let parsed = vec![ParsedTerm::Ref {
            name: "f".to_string(),
            scale: 1.0,
        }];
        let terms = inline(&parsed, &table).unwrap();
        assert_eq!(
            terms.len(),
            2,
            "duplicate-variable terms must not be merged"
        );
        assert_eq!(terms[0].variable, hg(0));
        assert_eq!(terms[1].variable, hg(0));
    }

    #[test]
    fn inline_undeclared_reference_errors() {
        let parsed = vec![ParsedTerm::Ref {
            name: "missing".to_string(),
            scale: 1.0,
        }];
        let err = inline(&parsed, &[]).unwrap_err();
        assert!(err.contains("undeclared"), "message: {err}");
        assert!(err.contains("missing"), "message: {err}");
    }

    #[test]
    fn detect_cycles_acyclic_chain_is_ok() {
        let table = vec![
            (
                "a".to_string(),
                vec![ParsedTerm::Ref {
                    name: "b".to_string(),
                    scale: 1.0,
                }],
            ),
            (
                "b".to_string(),
                vec![ParsedTerm::Ref {
                    name: "c".to_string(),
                    scale: 1.0,
                }],
            ),
            ("c".to_string(), vec![flat_lit(1.0, 0)]),
        ];
        assert!(detect_cycles(&table).is_ok());
    }

    #[test]
    fn detect_cycles_diamond_is_ok() {
        let table = vec![
            (
                "outer".to_string(),
                vec![
                    ParsedTerm::Ref {
                        name: "a".to_string(),
                        scale: 1.0,
                    },
                    ParsedTerm::Ref {
                        name: "b".to_string(),
                        scale: 1.0,
                    },
                ],
            ),
            (
                "a".to_string(),
                vec![ParsedTerm::Ref {
                    name: "base".to_string(),
                    scale: 1.0,
                }],
            ),
            (
                "b".to_string(),
                vec![ParsedTerm::Ref {
                    name: "base".to_string(),
                    scale: 1.0,
                }],
            ),
            ("base".to_string(), vec![flat_lit(1.0, 0)]),
        ];
        assert!(detect_cycles(&table).is_ok());
    }

    #[test]
    fn detect_cycles_two_node_cycle_names_both() {
        let table = vec![
            (
                "a".to_string(),
                vec![ParsedTerm::Ref {
                    name: "b".to_string(),
                    scale: 1.0,
                }],
            ),
            (
                "b".to_string(),
                vec![ParsedTerm::Ref {
                    name: "a".to_string(),
                    scale: 1.0,
                }],
            ),
        ];
        let err = detect_cycles(&table).unwrap_err();
        assert!(err.contains('a'), "message: {err}");
        assert!(err.contains('b'), "message: {err}");
        assert!(err.contains("a -> b -> a"), "message: {err}");
    }

    #[test]
    fn detect_cycles_self_reference_is_cycle_of_one() {
        let table = vec![(
            "e".to_string(),
            vec![ParsedTerm::Ref {
                name: "e".to_string(),
                scale: 1.0,
            }],
        )];
        let err = detect_cycles(&table).unwrap_err();
        assert!(err.contains('e'), "message: {err}");
        assert!(err.contains("e -> e"), "message: {err}");
    }

    /// Build a `levels`-deep doubling chain `e0 = hydro_generation(0)`,
    /// `e_k = @e_{k-1} + @e_{k-1}`. Un-capped, `@e_levels` expands to `2^levels` terms.
    fn doubling_chain(levels: u32) -> Vec<(String, ParsedExpression)> {
        let mut table = vec![("e0".to_string(), vec![flat_lit(1.0, 0)])];
        for k in 1..=levels {
            let prev = format!("e{}", k - 1);
            table.push((
                format!("e{k}"),
                vec![
                    ParsedTerm::Ref {
                        name: prev.clone(),
                        scale: 1.0,
                    },
                    ParsedTerm::Ref {
                        name: prev,
                        scale: 1.0,
                    },
                ],
            ));
        }
        table
    }

    #[test]
    fn inline_doubling_chain_hits_budget_and_returns_fast() {
        // 60 levels ⇒ 2^60 terms un-capped: the test completing at all proves the
        // cap fires long before the full vector materializes.
        let table = doubling_chain(60);
        let parsed = vec![ParsedTerm::Ref {
            name: "e60".to_string(),
            scale: 1.0,
        }];
        let err = inline(&parsed, &table).unwrap_err();
        assert!(err.contains("more than"), "message: {err}");
        assert!(err.contains("100000"), "should name the cap, got: {err}");
        assert!(
            err.contains("e60"),
            "should name the outermost reference, got: {err}"
        );
    }

    #[test]
    fn validate_references_resolve_does_not_expand_doubling_chain() {
        // The walk visits each Ref once (O(terms)); completing on a 2^60 chain
        // proves it never inlines.
        let table = doubling_chain(60);
        assert!(validate_references_resolve(&table).is_ok());
    }

    #[test]
    fn validate_references_resolve_accepts_resolvable_chain() {
        let table = vec![
            (
                "a".to_string(),
                vec![ParsedTerm::Ref {
                    name: "b".to_string(),
                    scale: 1.0,
                }],
            ),
            ("b".to_string(), vec![flat_lit(1.0, 0)]),
        ];
        assert!(validate_references_resolve(&table).is_ok());
    }

    #[test]
    fn validate_references_resolve_flags_undeclared_at_direct_site() {
        let table = vec![
            (
                "top".to_string(),
                vec![ParsedTerm::Ref {
                    name: "a".to_string(),
                    scale: 1.0,
                }],
            ),
            (
                "a".to_string(),
                vec![ParsedTerm::Ref {
                    name: "missing".to_string(),
                    scale: 1.0,
                }],
            ),
        ];
        let (i, msg) = validate_references_resolve(&table).unwrap_err();
        assert_eq!(i, 1, "the direct reference site is declaration index 1");
        assert!(
            msg.contains("undeclared") && msg.contains("missing"),
            "message: {msg}"
        );
    }

    #[test]
    fn inline_large_flat_expression_under_cap_is_ok() {
        let parsed: ParsedExpression = (0..1000).map(|id| flat_lit(1.0, id)).collect();
        let terms = inline(&parsed, &[]).unwrap();
        assert_eq!(terms.len(), 1000);
    }
}
