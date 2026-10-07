//! Normalization of a relational generic-constraint expression onto the F3
//! interval.
//!
//! [`normalize`] takes each side of an authored `LHS op RHS` expression,
//! already resolved to [`SideTerm`]s (named-expression references inlined,
//! any bound-position `@name` resolved to a parameter — both the caller's
//! job, in the `generic` module), and folds them onto a single merged
//! variable term list plus the operator's targeted [`AffineBound`]
//! endpoint(s): variable terms collect on the left (the right side's
//! sign-flipped), a same-variable/same-coefficient-kind pair merges by
//! summing its effective contribution, a merged literal column at exactly
//! `0.0` drops, and every non-variable term collects into the affine
//! remainder `R` assigned to the endpoint(s) the operator selects.

use cobre_core::{AffineBound, CoefficientRef, EntityId, LinearTerm};

/// One term of a relational side after the caller has inlined every
/// named-expression reference and resolved every bound-position `@name` to a
/// parameter — the three leaves [`normalize`] partitions.
pub(crate) enum SideTerm {
    /// A variable-bearing term.
    Variable(LinearTerm),
    /// A bare numeric literal with no variable.
    Constant(f64),
    /// A resolved `(coefficient, parameter_id)` scalar with no variable.
    Param(f64, EntityId),
}

/// The relational operator splitting an authored `LHS op RHS` expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RelOp {
    /// `<=` — assigns the affine remainder to the upper endpoint.
    Le,
    /// `>=` — assigns the affine remainder to the lower endpoint.
    Ge,
    /// `==` — assigns the same affine remainder to both endpoints.
    Eq,
}

/// Fold `lhs op rhs` onto the F3 interval.
///
/// Returns the merged variable term list (still authoring-order; the caller
/// canonicalizes) and the operator's targeted [`AffineBound`] endpoint(s).
pub(crate) fn normalize(
    lhs: Vec<SideTerm>,
    rhs: Vec<SideTerm>,
    op: RelOp,
) -> (Vec<LinearTerm>, Option<AffineBound>, Option<AffineBound>) {
    let mut variable_terms: Vec<LinearTerm> = Vec::with_capacity(lhs.len() + rhs.len());
    let mut constant = 0.0_f64;
    let mut param_terms: Vec<(f64, EntityId)> = Vec::new();

    for term in rhs {
        match term {
            SideTerm::Variable(lt) => variable_terms.push(negate_variable_term(lt)),
            SideTerm::Constant(v) => constant += v,
            SideTerm::Param(coef, id) => param_terms.push((coef, id)),
        }
    }
    for term in lhs {
        match term {
            SideTerm::Variable(lt) => variable_terms.push(lt),
            SideTerm::Constant(v) => constant -= v,
            SideTerm::Param(coef, id) => param_terms.push((-coef, id)),
        }
    }

    let variable_terms = fold_variable_terms(variable_terms);
    let bound = AffineBound {
        constant,
        terms: param_terms,
    };
    let (lower, upper) = match op {
        RelOp::Le => (None, Some(bound)),
        RelOp::Ge => (Some(bound), None),
        RelOp::Eq => (Some(bound.clone()), Some(bound)),
    };

    (variable_terms, lower, upper)
}

/// Merge terms sharing an identical variable and coefficient kind — summing
/// their effective contribution — then drop any merged literal column at
/// exactly `0.0`. Quadratic in the term count, which is small per constraint;
/// this runs once at study load, never per solver iteration.
fn fold_variable_terms(terms: Vec<LinearTerm>) -> Vec<LinearTerm> {
    let mut merged: Vec<LinearTerm> = Vec::with_capacity(terms.len());
    for term in terms {
        let slot = merged.iter_mut().find(|existing| {
            existing.variable == term.variable
                && coefficient_kind_matches(&existing.coefficient, &term.coefficient)
        });
        match slot {
            Some(existing) => merge_into(existing, &term),
            None => merged.push(term),
        }
    }
    merged.retain(|term| !is_zero_literal(&term.coefficient));
    merged
}

/// Negate a variable term moving from the RHS to the merged LHS: flips the
/// literal value directly for a `Literal` coefficient (so an unmerged term's
/// shape matches what authoring the negated form by hand would parse to —
/// `scale` untouched), or flips `scale` for a `Parameter` coefficient (the
/// parameter id itself cannot be negated).
fn negate_variable_term(mut lt: LinearTerm) -> LinearTerm {
    match &mut lt.coefficient {
        CoefficientRef::Literal(v) => *v = -*v,
        CoefficientRef::Parameter(_) => lt.scale = -lt.scale,
    }
    lt
}

/// Two literals always share a kind; two parameters share a kind only when
/// they name the same parameter — a different `@param` on the same variable
/// has no common scalar to sum at parse time and stays a separate term.
fn coefficient_kind_matches(a: &CoefficientRef, b: &CoefficientRef) -> bool {
    match (a, b) {
        (CoefficientRef::Literal(_), CoefficientRef::Literal(_)) => true,
        (CoefficientRef::Parameter(id_a), CoefficientRef::Parameter(id_b)) => id_a == id_b,
        (CoefficientRef::Literal(_), CoefficientRef::Parameter(_))
        | (CoefficientRef::Parameter(_), CoefficientRef::Literal(_)) => false,
    }
}

/// Fold `incoming` into `existing`, which [`coefficient_kind_matches`] has
/// already confirmed shares its variable and coefficient kind: sum the two
/// literals' effective value (`coefficient * scale`) for a literal pair, or
/// sum the two scales for a same-parameter pair (the coefficient — the
/// parameter id — is already identical).
fn merge_into(existing: &mut LinearTerm, incoming: &LinearTerm) {
    match incoming.coefficient {
        CoefficientRef::Literal(b) => {
            if let CoefficientRef::Literal(a) = &mut existing.coefficient {
                *a = *a * existing.scale + b * incoming.scale;
                existing.scale = 1.0;
            }
        }
        CoefficientRef::Parameter(_) => existing.scale += incoming.scale,
    }
}

/// Bit-exact zero check on a merged literal column — the drop criterion.
fn is_zero_literal(coef: &CoefficientRef) -> bool {
    matches!(coef, CoefficientRef::Literal(v) if v.to_bits() == 0.0_f64.to_bits())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use cobre_core::VariableRef;

    fn hg(id: i32) -> VariableRef {
        VariableRef::HydroGeneration {
            hydro_id: EntityId(id),
            block_id: None,
            bus_id: None,
        }
    }

    fn tg(id: i32) -> VariableRef {
        VariableRef::ThermalGeneration {
            thermal_id: EntityId(id),
            block_id: None,
        }
    }

    fn effective(term: &LinearTerm) -> f64 {
        match term.coefficient {
            CoefficientRef::Literal(v) => v * term.scale,
            CoefficientRef::Parameter(_) => panic!("expected a literal coefficient"),
        }
    }

    /// A variable-only RHS sign-flips into the merged LHS; `<=` assigns an
    /// empty affine remainder to the upper endpoint only.
    #[test]
    fn normalize_rhs_variable_sign_flips_into_lhs() {
        let lhs = vec![SideTerm::Variable(LinearTerm::literal(1.0, tg(5)))];
        let rhs = vec![SideTerm::Variable(LinearTerm::literal(0.87, hg(140)))];
        let (terms, lower, upper) = normalize(lhs, rhs, RelOp::Le);
        assert_eq!(terms.len(), 2);
        assert_eq!(terms[0].variable, hg(140));
        assert!((effective(&terms[0]) - (-0.87)).abs() < f64::EPSILON);
        assert_eq!(terms[1].variable, tg(5));
        assert!((effective(&terms[1]) - 1.0).abs() < f64::EPSILON);
        assert_eq!(lower, None);
        assert_eq!(
            upper,
            Some(AffineBound {
                constant: 0.0,
                terms: vec![]
            })
        );
    }

    /// An unmerged RHS literal term's negation flips the coefficient value
    /// directly, leaving `scale` untouched — the exact `LinearTerm` shape a
    /// hand-authored negated literal parses to, so `convert`'s byte-equality
    /// against a hand-flattened LHS holds with no group-scale involved.
    #[test]
    fn normalize_unmerged_rhs_literal_negates_coefficient_not_scale() {
        let rhs = vec![SideTerm::Variable(LinearTerm::literal(0.87, hg(140)))];
        let (terms, _, _) = normalize(vec![], rhs, RelOp::Le);
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].coefficient, CoefficientRef::Literal(-0.87));
        assert!((terms[0].scale - 1.0).abs() < f64::EPSILON);
    }

    /// A same-variable, same-kind repeat across both sides merges into one
    /// term, dropping to zero when the effective contributions cancel.
    #[test]
    fn normalize_same_variable_repeat_merges_and_drops_at_zero() {
        let lhs = vec![
            SideTerm::Variable(LinearTerm::literal(1.0, hg(0))),
            SideTerm::Variable(LinearTerm::literal(-1.0, hg(0))),
        ];
        let rhs = vec![SideTerm::Constant(5.0)];
        let (terms, lower, upper) = normalize(lhs, rhs, RelOp::Le);
        assert!(terms.is_empty(), "net-zero column must be dropped");
        assert_eq!(lower, None);
        assert_eq!(
            upper,
            Some(AffineBound {
                constant: 5.0,
                terms: vec![]
            })
        );
    }

    /// A non-cancelling same-variable repeat merges to the summed effective
    /// coefficient instead of dropping.
    #[test]
    fn normalize_same_variable_repeat_merges_to_sum() {
        let lhs = vec![
            SideTerm::Variable(LinearTerm::literal(2.0, hg(0))),
            SideTerm::Variable(LinearTerm::literal(3.0, hg(0))),
        ];
        let (terms, _, _) = normalize(lhs, vec![], RelOp::Le);
        assert_eq!(terms.len(), 1);
        assert!((effective(&terms[0]) - 5.0).abs() < f64::EPSILON);
    }

    /// A parameter-coefficient term never merges with a literal-coefficient
    /// term on the same variable — different kinds, no common scalar.
    #[test]
    fn normalize_parameter_and_literal_on_same_variable_stay_separate() {
        let lhs = vec![
            SideTerm::Variable(LinearTerm::literal(1.0, hg(0))),
            SideTerm::Variable(LinearTerm::parameter(EntityId(9), 1.0, hg(0))),
        ];
        let (terms, _, _) = normalize(lhs, vec![], RelOp::Le);
        assert_eq!(terms.len(), 2, "distinct coefficient kinds must not merge");
    }

    /// Two parameter-coefficient terms naming the SAME parameter on the same
    /// variable merge by summing their scales.
    #[test]
    fn normalize_same_parameter_on_same_variable_merges_scale() {
        let lhs = vec![
            SideTerm::Variable(LinearTerm::parameter(EntityId(9), 2.0, hg(0))),
            SideTerm::Variable(LinearTerm::parameter(EntityId(9), 3.0, hg(0))),
        ];
        let (terms, _, _) = normalize(lhs, vec![], RelOp::Le);
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].coefficient, CoefficientRef::Parameter(EntityId(9)));
        assert!((terms[0].scale - 5.0).abs() < f64::EPSILON);
    }

    /// A bare `@target` scalar on the RHS resolves (by the caller, before
    /// `normalize` runs) to `SideTerm::Param`; `>=` assigns it — unnegated —
    /// to the lower endpoint, reproducing `AffineBound::single`.
    #[test]
    fn normalize_rhs_param_scalar_assigns_lower_as_single() {
        let lhs = vec![SideTerm::Variable(LinearTerm::literal(1.0, hg(3)))];
        let rhs = vec![SideTerm::Param(1.0, EntityId(42))];
        let (terms, lower, upper) = normalize(lhs, rhs, RelOp::Ge);
        assert_eq!(terms.len(), 1);
        assert_eq!(upper, None);
        assert_eq!(lower, Some(AffineBound::single(EntityId(42))));
    }

    /// A LHS non-variable term (constant or param) negates into `R`.
    #[test]
    fn normalize_lhs_non_variable_terms_negate_into_bound() {
        let lhs = vec![SideTerm::Constant(10.0), SideTerm::Param(2.0, EntityId(7))];
        let (terms, lower, upper) = normalize(lhs, vec![], RelOp::Le);
        assert!(terms.is_empty());
        assert_eq!(lower, None);
        assert_eq!(
            upper,
            Some(AffineBound {
                constant: -10.0,
                terms: vec![(-2.0, EntityId(7))]
            })
        );
    }

    /// `==` assigns an equal affine remainder to both endpoints.
    #[test]
    fn normalize_eq_assigns_both_endpoints_equal() {
        let rhs = vec![SideTerm::Constant(3.0)];
        let (_, lower, upper) = normalize(vec![], rhs, RelOp::Eq);
        let expected = Some(AffineBound {
            constant: 3.0,
            terms: vec![],
        });
        assert_eq!(lower, expected);
        assert_eq!(upper, expected);
    }
}
