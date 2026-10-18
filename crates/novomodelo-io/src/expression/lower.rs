//! Lowers a parsed expression to [`ScalarExpr`] programs. The program's constructors run in
//! source order, so a sub-expression without a parameter folds to one literal and nothing is
//! reassociated. A fold that divides by an exact zero or yields a non-finite value is rejected
//! here, where the source span is still known; a divisor or value that reads a parameter is
//! left to whoever supplies the parameter values.
//!
//! An expression that reads an LP variable lowers to a [`LinearForm`]: one coefficient program
//! per structurally distinct [`VariableRef`], merged in authoring order. Every node is typed
//! before anything folds, so `0 * x` still holds a term for `x`.

use std::collections::HashMap;
use std::ops::Range;

use novomodelo_core::{EntityId, ScalarExpr, VariableRef, is_exact_zero};

use super::Diagnostic;
use super::catalog::{LineBusPairIndex, LinePairOrientation, variable_from_call};
use super::parser::{AddOp, Expr, ExprKind, MulOp, parse};

/// The names an expression can read.
pub(crate) struct Scope<'a> {
    pub(crate) params: &'a HashMap<String, EntityId>,
    pub(crate) line_index: &'a LineBusPairIndex,
}

/// A coefficient of `None` is a unit coefficient, built as a program only when an operation or
/// [`LinearForm::into_terms`] needs one, so `s * x` stays `s` and not `s * 1`. A `None` constant
/// is no constant at all, which a relation tells apart from a literal `0`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LinearForm {
    pub(crate) terms: Vec<(VariableRef, Option<ScalarExpr>)>,
    pub(crate) constant: Option<ScalarExpr>,
}

impl LinearForm {
    /// Materializes unit coefficients and drops every term whose coefficient folded to an exact
    /// zero, keeping order; the constant is left to the caller.
    pub(crate) fn into_terms(self) -> Vec<(VariableRef, ScalarExpr)> {
        self.terms
            .into_iter()
            .map(|(variable, coefficient)| (variable, coefficient.unwrap_or_else(one)))
            .filter(|(_, coefficient)| !coefficient.as_literal().is_some_and(is_exact_zero))
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Type {
    Scalar,
    Linear,
}

/// Parses `src` and lowers it as the body of a scalar parameter, which has no relational
/// operator.
pub(crate) fn lower_scalar_body(
    src: &str,
    params: &HashMap<String, EntityId>,
) -> Result<ScalarExpr, Diagnostic> {
    let relation = parse(src)?;
    if relation.rhs.is_some() {
        return Err(Diagnostic {
            span: 0..src.len(),
            message: "a scalar-parameter expression cannot contain a relational operator"
                .to_owned(),
        });
    }
    lower_scalar(&relation.lhs, params)
}

/// Lowers a tree that reads no LP variable; each chain step is checked over the source of its
/// partial result, from the chain's head to the step's operand.
fn lower_scalar(expr: &Expr, params: &HashMap<String, EntityId>) -> Result<ScalarExpr, Diagnostic> {
    match &expr.kind {
        ExprKind::Number(value) => Ok(ScalarExpr::literal(*value)),
        ExprKind::Name(name) => params
            .get(name)
            .map(|id| ScalarExpr::param(*id))
            .ok_or_else(|| Diagnostic {
                span: expr.span.clone(),
                message: format!(
                    "unknown scalar parameter \"@{name}\"; a named expression cannot appear in a \
                     scalar-parameter expression"
                ),
            }),
        ExprKind::Variable(_) => Err(Diagnostic {
            span: expr.span.clone(),
            message: "an LP variable cannot appear in a scalar-parameter expression".to_owned(),
        }),
        ExprKind::Neg(operand) => checked(-lower_scalar(operand, params)?, &expr.span),
        ExprKind::Sum { head, tail } => {
            let mut acc = lower_scalar(head, params)?;
            for (op, operand) in tail {
                let rhs = lower_scalar(operand, params)?;
                let step = head.span.start..operand.span.end;
                acc = match op {
                    AddOp::Add => checked(acc + rhs, &step)?,
                    AddOp::Sub => checked(acc - rhs, &step)?,
                };
            }
            Ok(acc)
        }
        ExprKind::Product { head, tail } => {
            let mut acc = lower_scalar(head, params)?;
            for (op, operand) in tail {
                let rhs = lower_scalar(operand, params)?;
                let step = head.span.start..operand.span.end;
                acc = match op {
                    MulOp::Mul => checked(acc * rhs, &step)?,
                    MulOp::Div => checked_div(acc, rhs, &step)?,
                };
            }
            Ok(acc)
        }
    }
}

fn type_of(expr: &Expr, scope: &Scope) -> Result<Type, Diagnostic> {
    match &expr.kind {
        ExprKind::Number(_) => Ok(Type::Scalar),
        ExprKind::Name(name) if scope.params.contains_key(name) => Ok(Type::Scalar),
        ExprKind::Name(name) => Err(Diagnostic {
            span: expr.span.clone(),
            message: format!(
                "unknown name \"@{name}\": no scalar parameter or named expression with this \
                 name was declared"
            ),
        }),
        ExprKind::Variable(_) => Ok(Type::Linear),
        ExprKind::Neg(operand) => type_of(operand, scope),
        ExprKind::Sum { head, tail } => {
            let mut ty = type_of(head, scope)?;
            for (_, operand) in tail {
                if type_of(operand, scope)? == Type::Linear {
                    ty = Type::Linear;
                }
            }
            Ok(ty)
        }
        ExprKind::Product { head, tail } => {
            let mut ty = type_of(head, scope)?;
            for (op, operand) in tail {
                let rhs = type_of(operand, scope)?;
                ty = match (op, ty, rhs) {
                    (MulOp::Mul, Type::Linear, Type::Linear) | (MulOp::Div, _, Type::Linear) => {
                        return Err(not_linear(op, head.span.start..operand.span.end));
                    }
                    (_, Type::Scalar, Type::Scalar) => Type::Scalar,
                    _ => Type::Linear,
                };
            }
            Ok(ty)
        }
    }
}

fn not_linear(op: &MulOp, span: Range<usize>) -> Diagnostic {
    let message = match op {
        MulOp::Mul => "product of LP variables is not linear",
        MulOp::Div => "division by an expression containing LP variables",
    };
    Diagnostic {
        span,
        message: message.to_owned(),
    }
}

/// Lowers a tree that reads an LP variable or not; a tree that reads none becomes a form with
/// only a constant.
fn lower_form(expr: &Expr, scope: &Scope) -> Result<LinearForm, Diagnostic> {
    enum Acc {
        Scalar(ScalarExpr),
        Form(LinearForm),
    }
    let operand = |expr: &Expr| -> Result<Acc, Diagnostic> {
        Ok(match type_of(expr, scope)? {
            Type::Scalar => Acc::Scalar(lower_scalar(expr, scope.params)?),
            Type::Linear => Acc::Form(lower_form(expr, scope)?),
        })
    };

    let linear = type_of(expr, scope)? == Type::Linear;
    match &expr.kind {
        ExprKind::Variable(call) => {
            let (variable, orientation) =
                variable_from_call(call, scope.line_index).map_err(|message| Diagnostic {
                    span: expr.span.clone(),
                    message,
                })?;
            let coefficient = match orientation {
                LinePairOrientation::Forward => None,
                LinePairOrientation::Reversed => Some(ScalarExpr::literal(-1.0)),
            };
            Ok(LinearForm {
                terms: vec![(variable, coefficient)],
                constant: None,
            })
        }
        ExprKind::Neg(inner) if linear => {
            map_form(lower_form(inner, scope)?, |c| checked(-c, &expr.span), None)
        }
        ExprKind::Sum { head, tail } if linear => {
            let mut acc = lower_form(head, scope)?;
            for (op, rhs) in tail {
                let other = lower_form(rhs, scope)?;
                acc = combine(acc, other, op, &(head.span.start..rhs.span.end))?;
            }
            Ok(acc)
        }
        ExprKind::Product { head, tail } if linear => {
            let mut acc = operand(head)?;
            for (op, rhs) in tail {
                let step = head.span.start..rhs.span.end;
                acc = match (acc, op, operand(rhs)?) {
                    (Acc::Scalar(lhs), MulOp::Mul, Acc::Scalar(rhs)) => {
                        Acc::Scalar(checked(lhs * rhs, &step)?)
                    }
                    (Acc::Scalar(lhs), MulOp::Div, Acc::Scalar(rhs)) => {
                        Acc::Scalar(checked_div(lhs, rhs, &step)?)
                    }
                    (Acc::Scalar(s), MulOp::Mul, Acc::Form(form)) => {
                        Acc::Form(map_form(form, |c| checked(s.clone() * c, &step), Some(&s))?)
                    }
                    (Acc::Form(form), MulOp::Mul, Acc::Scalar(s)) => {
                        Acc::Form(map_form(form, |c| checked(c * s.clone(), &step), Some(&s))?)
                    }
                    (Acc::Form(form), MulOp::Div, Acc::Scalar(s)) => {
                        Acc::Form(map_form(form, |c| checked_div(c, s.clone(), &step), None)?)
                    }
                    (Acc::Form(_), MulOp::Mul, Acc::Form(_)) | (_, MulOp::Div, Acc::Form(_)) => {
                        return Err(not_linear(op, step));
                    }
                };
            }
            Ok(match acc {
                Acc::Scalar(value) => LinearForm {
                    terms: Vec::new(),
                    constant: Some(value),
                },
                Acc::Form(form) => form,
            })
        }
        _ => Ok(LinearForm {
            terms: Vec::new(),
            constant: Some(lower_scalar(expr, scope.params)?),
        }),
    }
}

/// Applies `f` to every coefficient and to the constant. A unit coefficient becomes
/// `unit_image` when one is given (the scale itself, for a product) and `f(literal(1.0))`
/// otherwise.
fn map_form(
    form: LinearForm,
    f: impl Fn(ScalarExpr) -> Result<ScalarExpr, Diagnostic>,
    unit_image: Option<&ScalarExpr>,
) -> Result<LinearForm, Diagnostic> {
    let terms = form
        .terms
        .into_iter()
        .map(|(variable, coefficient)| {
            let image = match (coefficient, unit_image) {
                (Some(c), _) => f(c)?,
                (None, Some(unit_image)) => unit_image.clone(),
                (None, None) => f(one())?,
            };
            Ok((variable, Some(image)))
        })
        .collect::<Result<_, Diagnostic>>()?;
    let constant = form.constant.map(&f).transpose()?;
    Ok(LinearForm { terms, constant })
}

/// Adds or subtracts `other` into `acc`: a variable already in `acc` (structural equality) has
/// its coefficient combined in place, a new one is appended. No exact-zero coefficient is
/// dropped here, so a later term can still change it.
fn combine(
    mut acc: LinearForm,
    other: LinearForm,
    op: &AddOp,
    span: &Range<usize>,
) -> Result<LinearForm, Diagnostic> {
    let apply = |lhs: ScalarExpr, rhs: ScalarExpr| match op {
        AddOp::Add => checked(lhs + rhs, span),
        AddOp::Sub => checked(lhs - rhs, span),
    };
    for (variable, c2) in other.terms {
        if let Some((_, c1)) = acc.terms.iter_mut().find(|(known, _)| *known == variable) {
            let lhs = c1.take().unwrap_or_else(one);
            *c1 = Some(apply(lhs, c2.unwrap_or_else(one))?);
        } else {
            let coefficient = match op {
                AddOp::Add => c2,
                AddOp::Sub => Some(checked(-c2.unwrap_or_else(one), span)?),
            };
            acc.terms.push((variable, coefficient));
        }
    }
    acc.constant = match (acc.constant, other.constant) {
        (Some(k1), Some(k2)) => Some(apply(k1, k2)?),
        (k1, None) => k1,
        (None, Some(k2)) => Some(match op {
            AddOp::Add => k2,
            AddOp::Sub => checked(-k2, span)?,
        }),
    };
    Ok(acc)
}

fn one() -> ScalarExpr {
    ScalarExpr::literal(1.0)
}

fn checked(value: ScalarExpr, span: &Range<usize>) -> Result<ScalarExpr, Diagnostic> {
    match value.as_literal() {
        Some(v) if !v.is_finite() => Err(Diagnostic {
            span: span.clone(),
            message: format!("evaluates to a non-finite value ({v})"),
        }),
        _ => Ok(value),
    }
}

fn checked_div(
    lhs: ScalarExpr,
    rhs: ScalarExpr,
    span: &Range<usize>,
) -> Result<ScalarExpr, Diagnostic> {
    if rhs.as_literal().is_some_and(is_exact_zero) {
        return Err(Diagnostic {
            span: span.clone(),
            message: "division by zero".to_owned(),
        });
    }
    checked(lhs / rhs, span)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use std::collections::HashMap;

    use novomodelo_core::{EntityId, Line, ScalarExpr, VariableRef};

    use super::{LinearForm, Scope, lower_form, lower_scalar_body};
    use crate::expression::Diagnostic;
    use crate::expression::catalog::build_line_bus_pair_index;
    use crate::expression::parser::{Expr, parse};

    fn params() -> HashMap<String, EntityId> {
        HashMap::from([
            ("a".to_owned(), EntityId(7)),
            ("b".to_owned(), EntityId(8)),
            ("c".to_owned(), EntityId(9)),
        ])
    }

    fn lowered(src: &str) -> ScalarExpr {
        lower_scalar_body(src, &params()).unwrap()
    }

    fn rendered_rejection(src: &str) -> String {
        lower_scalar_body(src, &params()).unwrap_err().render(src)
    }

    fn value_at(expr: &ScalarExpr, a: f64, b: f64) -> f64 {
        expr.evaluate(|id| match id {
            EntityId(7) => a,
            EntityId(8) => b,
            _ => panic!("unexpected parameter id {id:?}"),
        })
    }

    /// derived: `2 * 3 + 1` is `(2.0 * 3.0) + 1.0 = 7.0`, one literal; `-(4 / 2)` is
    /// `-(4.0 / 2.0) = -2.0`.
    #[test]
    fn scalar_body_folds_literal_only_subexpressions() {
        let expr = lowered("2 * 3 + 1");
        assert_eq!(expr.as_literal().map(f64::to_bits), Some(7.0_f64.to_bits()));
        assert_eq!(expr.op_count(), 1);

        let expr = lowered("-(4 / 2)");
        assert_eq!(
            expr.as_literal().map(f64::to_bits),
            Some((-2.0_f64).to_bits())
        );
    }

    /// derived: `2 * @a * 3` is the program `2, a, mul, 3, mul` (5 ops) and evaluates at
    /// `a = 0.1` to `(2.0 * 0.1) * 3.0`; `2 * (@a * 3)` is `2, a, 3, mul, mul` and evaluates to
    /// `2.0 * (0.1 * 3.0)`; `@a + 0.1 + 0.2` at `a = 3.0` is `(3.0 + 0.1) + 0.2 =
    /// 3.3000000000000003`, which a fold of `0.1 + 0.2` would turn into `3.3`; `@a + @b * 2` is
    /// the program `a, b, 2, mul, add` (5 ops) and at `a = 1.0`, `b = 3.0` is `1.0 + 3.0 * 2.0 =
    /// 7.0`.
    #[test]
    fn scalar_body_keeps_the_authored_association() {
        let a = || ScalarExpr::param(EntityId(7));
        let lit = ScalarExpr::literal;

        let left = lowered("2 * @a * 3");
        assert_eq!(left.op_count(), 5);
        assert_eq!(left, (lit(2.0) * a()) * lit(3.0));
        assert_eq!(
            value_at(&left, 0.1, 0.0).to_bits(),
            ((2.0_f64 * 0.1) * 3.0).to_bits()
        );

        let right = lowered("2 * (@a * 3)");
        assert_eq!(right.op_count(), 5);
        assert_eq!(right, lit(2.0) * (a() * lit(3.0)));
        assert_ne!(left, right);
        assert_eq!(
            value_at(&right, 0.1, 0.0).to_bits(),
            (2.0_f64 * (0.1 * 3.0)).to_bits()
        );

        let chain = lowered("@a + 0.1 + 0.2");
        assert_eq!(chain.op_count(), 5);
        assert_eq!(
            value_at(&chain, 3.0, 0.0).to_bits(),
            ((3.0_f64 + 0.1) + 0.2).to_bits()
        );
        assert_ne!(
            value_at(&chain, 3.0, 0.0).to_bits(),
            (3.0_f64 + (0.1 + 0.2)).to_bits()
        );

        let mixed = lowered("@a + @b * 2");
        assert_eq!(mixed, a() + ScalarExpr::param(EntityId(8)) * lit(2.0));
        assert_eq!(mixed.op_count(), 5);
        assert_eq!(
            mixed.params().collect::<Vec<_>>(),
            [EntityId(7), EntityId(8)]
        );
        assert_eq!(value_at(&mixed, 1.0, 3.0).to_bits(), 7.0_f64.to_bits());
    }

    /// derived: `@a + 0` is `a, 0, add` (3 ops), so at `a = -0.0` it is `-0.0 + 0.0 = +0.0`,
    /// not `a`'s own `-0.0`.
    #[test]
    fn scalar_body_applies_no_identity() {
        let expr = lowered("@a + 0");
        assert_eq!(expr.op_count(), 3);
        assert_eq!(expr.as_param(), None);
        assert_eq!(value_at(&expr, -0.0, 0.0).to_bits(), 0.0_f64.to_bits());
    }

    /// derived: `1 / @a` is `1, a, div`; at `a = 4.0` it is `0.25`, and at `a = 0.0` it is
    /// `+inf`, a value the lowering cannot see.
    #[test]
    fn scalar_body_with_a_parameter_divisor_is_accepted() {
        let expr = lowered("1 / @a");
        assert_eq!(expr.op_count(), 3);
        assert_eq!(value_at(&expr, 4.0, 0.0).to_bits(), 0.25_f64.to_bits());
        assert_eq!(value_at(&expr, 0.0, 0.0).to_bits(), f64::INFINITY.to_bits());
    }

    /// derived: `(2 - 2)` folds to `2.0 - 2.0 = +0.0` and `-0` folds to `-0.0`; both are exact
    /// zeros, reported over the division step from the chain head.
    #[test]
    fn scalar_body_division_by_a_folded_zero_is_rejected() {
        assert_eq!(
            rendered_rejection("@a / (2 - 2)"),
            "`@a / (2 - 2)`: division by zero"
        );
        assert_eq!(rendered_rejection("@a / -0"), "`@a / -0`: division by zero");
        assert_eq!(
            rendered_rejection("1 + @a * 3 / (1 - 1) * 2"),
            "`@a * 3 / (1 - 1)`: division by zero"
        );
    }

    /// derived: `1e308 * 1e308` overflows to `+inf`; in `1 / (1e308 * 10)` the inner step
    /// `1e308 * 10` overflows to `+inf` before the division; `-1e308 - 1e308` overflows to
    /// `-inf`.
    #[test]
    fn scalar_body_non_finite_fold_is_rejected() {
        assert_eq!(
            rendered_rejection("1e308 * 1e308"),
            "`1e308 * 1e308`: evaluates to a non-finite value (inf)"
        );
        assert_eq!(
            rendered_rejection("1 / (1e308 * 10)"),
            "`1e308 * 10`: evaluates to a non-finite value (inf)"
        );
        assert_eq!(
            rendered_rejection("-1e308 - 1e308"),
            "`-1e308 - 1e308`: evaluates to a non-finite value (-inf)"
        );
    }

    #[test]
    fn scalar_body_rejects_an_lp_variable() {
        assert_eq!(
            rendered_rejection("@a * hydro_storage(1)"),
            "`hydro_storage(1)`: an LP variable cannot appear in a scalar-parameter expression"
        );
    }

    #[test]
    fn scalar_body_rejects_an_unknown_name_with_the_hint() {
        assert_eq!(
            rendered_rejection("@a + 2 * @demand"),
            "`@demand`: unknown scalar parameter \"@demand\"; a named expression cannot appear \
             in a scalar-parameter expression"
        );
    }

    #[test]
    fn scalar_body_rejects_a_relation() {
        for src in ["@a <= 1", "2 * @a == @b"] {
            assert_eq!(
                rendered_rejection(src),
                format!(
                    "`{src}`: a scalar-parameter expression cannot contain a relational operator"
                ),
                "{src:?}"
            );
        }
    }

    /// Lowers one side under `params()` and a single line 5 running from bus 3 to bus 7.
    fn lower_side(side: &Expr) -> Result<LinearForm, Diagnostic> {
        let params = params();
        let line = Line {
            id: EntityId(5),
            name: "L5".to_owned(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            source_bus_id: EntityId(3),
            target_bus_id: EntityId(7),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 100.0,
            reverse_capacity_mw: 100.0,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        };
        let line_index = build_line_bus_pair_index(&[line]).unwrap();
        lower_form(
            side,
            &Scope {
                params: &params,
                line_index: &line_index,
            },
        )
    }

    fn form(src: &str) -> LinearForm {
        let relation = parse(src).unwrap();
        assert!(relation.rhs.is_none(), "{src:?} is a relation");
        lower_side(&relation.lhs).unwrap_or_else(|d| panic!("{}", d.render(src)))
    }

    fn sides(src: &str) -> (LinearForm, LinearForm) {
        let relation = parse(src).unwrap();
        let (_, rhs) = relation.rhs.unwrap();
        (
            lower_side(&relation.lhs).unwrap(),
            lower_side(&rhs).unwrap(),
        )
    }

    fn terms(src: &str) -> Vec<(VariableRef, ScalarExpr)> {
        form(src).into_terms()
    }

    /// Every coefficient of `src` as the bits of its folded literal.
    fn literal_terms(src: &str) -> Vec<(VariableRef, u64)> {
        terms(src)
            .into_iter()
            .map(|(variable, coefficient)| {
                let value = coefficient
                    .as_literal()
                    .unwrap_or_else(|| panic!("{src:?}: {variable:?} has a non-literal program"));
                (variable, value.to_bits())
            })
            .collect()
    }

    fn linear_rejection(src: &str) -> String {
        parse(src)
            .and_then(|relation| lower_side(&relation.lhs))
            .unwrap_err()
            .render(src)
    }

    fn a() -> ScalarExpr {
        ScalarExpr::param(EntityId(7))
    }

    fn b() -> ScalarExpr {
        ScalarExpr::param(EntityId(8))
    }

    fn lit(value: f64) -> ScalarExpr {
        ScalarExpr::literal(value)
    }

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

    fn exchange(line: i32) -> VariableRef {
        VariableRef::LineExchange {
            line_id: EntityId(line),
            block_id: None,
        }
    }

    /// derived: `2 * h0 + 3 * h0` is `2.0 + 3.0 = 5.0`, one folded literal;
    /// `t5 + 2 * h0 + 3 * t5` keeps `t5` first at `1.0 + 3.0 = 4.0`, then `h0` at `2.0`;
    /// `h10 + h11` is `[1.0, 1.0]`; `2.5 * t5 - h3` is `[2.5, -1.0]`; `t5 - h3` is
    /// `[1.0, -1.0]`; the storage ramp is `[1.0, -1.0]`; `(h0 + h1)` lowers like `h0 + h1`;
    /// `h5 - (h0 + h1)` is `[1.0, -1.0, -1.0]`.
    #[test]
    fn repeated_literal_coefficients_merge_in_authoring_order() {
        let merged = terms("2 * hydro_generation(0) + 3 * hydro_generation(0)");
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].0, hg(0));
        assert_eq!(merged[0].1.op_count(), 1);
        assert_eq!(
            merged[0].1.as_literal().map(f64::to_bits),
            Some(5.0_f64.to_bits())
        );

        let storage = |final_side: bool| {
            if final_side {
                VariableRef::HydroStorageFinal {
                    hydro_id: EntityId(5),
                    block_id: Some(1),
                }
            } else {
                VariableRef::HydroStorageInitial {
                    hydro_id: EntityId(5),
                    block_id: Some(1),
                }
            }
        };
        for (src, expected) in [
            (
                "thermal_generation(5) + 2 * hydro_generation(0) + 3 * thermal_generation(5)",
                vec![(tg(5), 4.0), (hg(0), 2.0)],
            ),
            (
                "hydro_generation(10) + hydro_generation(11)",
                vec![(hg(10), 1.0), (hg(11), 1.0)],
            ),
            (
                "2.5 * thermal_generation(5) - hydro_generation(3)",
                vec![(tg(5), 2.5), (hg(3), -1.0)],
            ),
            (
                "thermal_generation(5) - hydro_generation(3)",
                vec![(tg(5), 1.0), (hg(3), -1.0)],
            ),
            (
                "hydro_storage_final(5, 1) - hydro_storage_initial(5, 1)",
                vec![(storage(true), 1.0), (storage(false), -1.0)],
            ),
            (
                "(hydro_generation(0) + hydro_generation(1))",
                vec![(hg(0), 1.0), (hg(1), 1.0)],
            ),
            (
                "hydro_generation(5) - (hydro_generation(0) + hydro_generation(1))",
                vec![(hg(5), 1.0), (hg(0), -1.0), (hg(1), -1.0)],
            ),
        ] {
            let expected: Vec<(VariableRef, u64)> = expected
                .into_iter()
                .map(|(variable, value)| (variable, f64::to_bits(value)))
                .collect();
            assert_eq!(literal_terms(src), expected, "{src:?}");
        }
        assert_eq!(
            form("(hydro_generation(0) + hydro_generation(1))"),
            form("hydro_generation(0) + hydro_generation(1)")
        );
    }

    /// derived: `2 * h0 + @a * h0` is the program `2, a, add`, at `a = 0.1` equal to
    /// `2.0 + 0.1`; `@a * h0` is the bare parameter; `2.5 * @a * h0` is `2.5, a, mul`;
    /// `2.0 * @a * h0` is `2, a, mul`; in `t5 - @a * h3`, `h3` is `a, neg`, at `a = 0.5` equal
    /// to `-0.5`.
    #[test]
    fn parameter_and_literal_on_one_variable_merge_into_one_program() {
        let merged = terms("2 * hydro_generation(0) + @a * hydro_generation(0)");
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].1, lit(2.0) + a());
        assert_ne!(merged[0].1, a() + lit(2.0));
        assert_eq!(merged[0].1.op_count(), 3);
        assert_eq!(
            merged[0].1.evaluate(|_| 0.1).to_bits(),
            (2.0_f64 + 0.1).to_bits()
        );

        let implicit = terms("@a * hydro_generation(0)");
        assert_eq!(implicit.len(), 1);
        assert_eq!(implicit[0].0, hg(0));
        assert_eq!(implicit[0].1.as_param(), Some(EntityId(7)));
        assert_eq!(implicit[0].1.op_count(), 1);

        assert_eq!(
            terms("2.5 * @a * hydro_generation(0)"),
            [(hg(0), lit(2.5) * a())]
        );
        assert_eq!(
            terms("2.0 * @a * hydro_generation(0)"),
            [(hg(0), lit(2.0) * a())]
        );

        let negated = terms("thermal_generation(5) - @a * hydro_generation(3)");
        assert_eq!(negated, [(tg(5), lit(1.0)), (hg(3), -a())]);
        assert_eq!(negated[1].1.op_count(), 2);
        assert_eq!(
            negated[1].1.evaluate(|_| 0.5).to_bits(),
            (-0.5_f64).to_bits()
        );
    }

    /// derived: `h0 - h0` is `1.0 - 1.0 = +0.0`, an exact zero, kept in the form and dropped by
    /// `into_terms`; `h0 - h0 + h0` is `(1.0 - 1.0) + 1.0 = 1.0`; `h0 - h0 + @a * h0` is the
    /// program `0, a, add`, which is not a literal and stays.
    #[test]
    fn a_variable_cancelled_to_exact_zero_is_dropped() {
        let cancelled = form("hydro_generation(0) - hydro_generation(0)");
        assert_eq!(cancelled.terms.len(), 1);
        assert_eq!(
            cancelled.terms[0]
                .1
                .as_ref()
                .and_then(ScalarExpr::as_literal)
                .map(f64::to_bits),
            Some(0.0_f64.to_bits())
        );
        assert!(cancelled.into_terms().is_empty());

        assert_eq!(
            literal_terms("2 * hydro_generation(1) + hydro_generation(0) - hydro_generation(0)"),
            [(hg(1), 2.0_f64.to_bits())]
        );
        assert_eq!(
            literal_terms("hydro_generation(0) - hydro_generation(0) + hydro_generation(0)"),
            [(hg(0), 1.0_f64.to_bits())]
        );

        let restored =
            terms("hydro_generation(0) - hydro_generation(0) + @a * hydro_generation(0)");
        assert_eq!(restored, [(hg(0), lit(0.0) + a())]);
        assert_eq!(restored[0].1.op_count(), 3);
    }

    /// derived: `0 * h0` types linear and holds `h0` at `+0.0`; `-0 * h0` holds it at `-0.0`;
    /// both are exact zeros. `0 * h0 + 3` keeps the constant `3.0`. `@a * 0 * h0` is the
    /// program `a, 0, mul`, which is not a literal and stays.
    #[test]
    fn zero_times_a_variable_types_linear_and_drops_its_term() {
        let zero = form("0 * hydro_generation(0)");
        assert_eq!(
            zero,
            LinearForm {
                terms: vec![(hg(0), Some(lit(0.0)))],
                constant: None,
            }
        );
        assert!(zero.into_terms().is_empty());

        let negative_zero = form("-0 * hydro_generation(0)");
        assert_eq!(
            negative_zero.terms[0]
                .1
                .as_ref()
                .and_then(ScalarExpr::as_literal)
                .map(f64::to_bits),
            Some((-0.0_f64).to_bits())
        );
        assert!(negative_zero.into_terms().is_empty());

        let with_constant = form("0 * hydro_generation(0) + 3");
        assert_eq!(
            with_constant
                .constant
                .as_ref()
                .and_then(ScalarExpr::as_literal)
                .map(f64::to_bits),
            Some(3.0_f64.to_bits())
        );
        assert!(with_constant.into_terms().is_empty());

        assert_eq!(
            terms("@a * 0 * hydro_generation(0)"),
            [(hg(0), a() * lit(0.0))]
        );
    }

    /// derived: `@a * (h0 + 2 * h1)` is `h0: a`, `h1: a, 2, mul`; `(h0 + 2 * h1) * @a` is
    /// `h0: a`, `h1: 2, a, mul`; the two `h1` programs differ only in operand order.
    #[test]
    fn parameter_scaling_distributes_over_a_group() {
        let left = terms("@a * (hydro_generation(0) + 2 * hydro_generation(1))");
        assert_eq!(left, [(hg(0), a()), (hg(1), a() * lit(2.0))]);
        assert_eq!(left[0].1.as_param(), Some(EntityId(7)));

        let right = terms("(hydro_generation(0) + 2 * hydro_generation(1)) * @a");
        assert_eq!(right, [(hg(0), a()), (hg(1), lit(2.0) * a())]);
        assert_eq!(right[0].1.as_param(), Some(EntityId(7)));

        assert_ne!(left[1].1, right[1].1);
    }

    /// derived: `2 * (3 * (@a * h0))` is `2, 3, a, mul, mul` (5 ops), not the `6, a, mul` of
    /// `2 * 3 * @a * h0`; `2.0 * (@a * h0)` is `2, a, mul`; `2 * (h0 - h1)` and
    /// `2 * h0 - 2 * h1` are both `[2.0, -2.0]`; `3 * (h0 + h1)` is `[3.0, 3.0]`;
    /// `2 * (h0 - 3 * (h1 + h2))` is `[2.0, 2.0 * -3.0, 2.0 * -3.0] = [2.0, -6.0, -6.0]`.
    #[test]
    fn nested_literal_scales_keep_the_authored_order() {
        let nested = terms("2 * (3 * (@a * hydro_generation(0)))");
        assert_eq!(nested, [(hg(0), lit(2.0) * (lit(3.0) * a()))]);
        assert_eq!(nested[0].1.op_count(), 5);
        let flat = terms("2 * 3 * @a * hydro_generation(0)");
        assert_eq!(flat, [(hg(0), lit(6.0) * a())]);
        assert_ne!(nested[0].1, flat[0].1);

        assert_eq!(
            terms("2.0 * (@a * hydro_generation(0))"),
            [(hg(0), lit(2.0) * a())]
        );

        let distributed = [(hg(0), 2.0_f64.to_bits()), (hg(1), (-2.0_f64).to_bits())];
        assert_eq!(
            literal_terms("2 * (hydro_generation(0) - hydro_generation(1))"),
            distributed
        );
        assert_eq!(
            literal_terms("2 * hydro_generation(0) - 2 * hydro_generation(1)"),
            distributed
        );
        assert_eq!(
            literal_terms("3 * (hydro_generation(0) + hydro_generation(1))"),
            [(hg(0), 3.0_f64.to_bits()), (hg(1), 3.0_f64.to_bits())]
        );
        assert_eq!(
            literal_terms(
                "2 * (hydro_generation(0) - 3 * (hydro_generation(1) + hydro_generation(2)))"
            ),
            [
                (hg(0), 2.0_f64.to_bits()),
                (hg(1), (-6.0_f64).to_bits()),
                (hg(2), (-6.0_f64).to_bits())
            ]
        );
    }

    /// derived: in `(h0 + @a * h1) / 2`, `h0` is `1.0 / 2.0 = 0.5` and `h1` is `a, 2, div`;
    /// `(h0 + 3) / 2` has the constant `3.0 / 2.0 = 1.5`; `h0 / @a` is `1, a, div`, at
    /// `a = 4.0` equal to `0.25`.
    #[test]
    fn division_by_a_scalar_divides_every_coefficient() {
        let divided = terms("(hydro_generation(0) + @a * hydro_generation(1)) / 2");
        assert_eq!(divided, [(hg(0), lit(0.5)), (hg(1), a() / lit(2.0))]);
        assert_eq!(
            divided[0].1.as_literal().map(f64::to_bits),
            Some(0.5_f64.to_bits())
        );
        assert_eq!(divided[1].1.op_count(), 3);

        let with_constant = form("(hydro_generation(0) + 3) / 2");
        assert_eq!(
            with_constant
                .constant
                .as_ref()
                .and_then(ScalarExpr::as_literal)
                .map(f64::to_bits),
            Some(1.5_f64.to_bits())
        );

        let by_parameter = terms("hydro_generation(0) / @a");
        assert_eq!(by_parameter, [(hg(0), lit(1.0) / a())]);
        assert_eq!(
            by_parameter[0].1.evaluate(|_| 4.0).to_bits(),
            0.25_f64.to_bits()
        );
    }

    /// derived: the pair (7, 3) runs against line 5's 3 -> 7 direction, so its coefficient is
    /// `-1.0`; `2 * pair` is `2.0 * -1.0 = -2.0`; `line_exchange(5) + pair` is
    /// `1.0 + -1.0 = 0.0`, dropped.
    #[test]
    fn reversed_line_pair_contributes_minus_one() {
        let reversed = form("line_exchange(source_bus=7, target_bus=3)");
        assert_eq!(
            reversed,
            LinearForm {
                terms: vec![(exchange(5), Some(lit(-1.0)))],
                constant: None,
            }
        );
        assert_eq!(
            literal_terms("line_exchange(source_bus=7, target_bus=3)"),
            [(exchange(5), (-1.0_f64).to_bits())]
        );
        assert_eq!(
            literal_terms("2 * line_exchange(source_bus=7, target_bus=3)"),
            [(exchange(5), (-2.0_f64).to_bits())]
        );

        let cancelled = form("line_exchange(5) + line_exchange(source_bus=7, target_bus=3)");
        assert_eq!(cancelled.terms.len(), 1);
        assert!(cancelled.into_terms().is_empty());
    }

    /// derived: the pair (3, 7) follows line 5's direction, so it is a unit coefficient, `1.0`
    /// once materialized, exactly as `line_exchange(5)`.
    #[test]
    fn forward_line_pair_contributes_plus_one() {
        let forward = form("line_exchange(source_bus=3, target_bus=7)");
        assert_eq!(
            forward,
            LinearForm {
                terms: vec![(exchange(5), None)],
                constant: None,
            }
        );
        assert_eq!(forward, form("line_exchange(5)"));
        assert_eq!(
            literal_terms("line_exchange(source_bus=3, target_bus=7)"),
            [(exchange(5), 1.0_f64.to_bits())]
        );
    }

    #[test]
    fn products_and_divisions_of_lp_variables_are_rejected() {
        let product = "product of LP variables is not linear";
        let division = "division by an expression containing LP variables";
        let unknown = |name: &str| {
            format!(
                "unknown name \"@{name}\": no scalar parameter or named expression with this \
                 name was declared"
            )
        };
        for (src, fragment, message) in [
            (
                "hydro_generation(0) * hydro_generation(1)",
                "hydro_generation(0) * hydro_generation(1)",
                product.to_owned(),
            ),
            (
                "@a / hydro_generation(0)",
                "@a / hydro_generation(0)",
                division.to_owned(),
            ),
            (
                "hydro_generation(0) / hydro_generation(1)",
                "hydro_generation(0) / hydro_generation(1)",
                division.to_owned(),
            ),
            (
                "1 + 2 * hydro_generation(0) * (hydro_generation(1) + 1)",
                "2 * hydro_generation(0) * (hydro_generation(1) + 1)",
                product.to_owned(),
            ),
            (
                "@nobody * hydro_generation(0)",
                "@nobody",
                unknown("nobody"),
            ),
            (
                "@nobody + hydro_generation(0)",
                "@nobody",
                unknown("nobody"),
            ),
            ("3 * ()", ")", "unexpected ')'".to_owned()),
        ] {
            assert_eq!(
                linear_rejection(src),
                format!("`{fragment}`: {message}"),
                "{src:?}"
            );
        }
    }

    /// derived: the inner `1e308 * h0` holds `1e308`, and the outer `1e308 * 1e308` overflows
    /// to `+inf`; `2 - 2` folds to `+0.0` and `-0` to `-0.0`, both exact zeros;
    /// `1e308 + 1e308` overflows to `+inf` when two `h0` coefficients merge; the constant
    /// `-1e308 - 1e308` overflows to `-inf`.
    #[test]
    fn folded_faults_in_linear_lowering_are_rejected() {
        let overflow = "evaluates to a non-finite value";
        for (src, message) in [
            (
                "1e308 * (1e308 * hydro_generation(0))",
                format!("{overflow} (inf)"),
            ),
            (
                "1e308 * hydro_generation(0) + 1e308 * hydro_generation(0)",
                format!("{overflow} (inf)"),
            ),
            (
                "hydro_generation(0) - 1e308 - 1e308",
                format!("{overflow} (-inf)"),
            ),
            (
                "hydro_generation(0) / (2 - 2)",
                "division by zero".to_owned(),
            ),
            ("hydro_generation(0) / -0", "division by zero".to_owned()),
            (
                "(hydro_generation(0) + 1) / (1 - 1)",
                "division by zero".to_owned(),
            ),
        ] {
            assert_eq!(
                linear_rejection(src),
                format!("`{src}`: {message}"),
                "{src:?}"
            );
        }
    }

    /// derived: each accepted lone reference is a unit coefficient, `1.0` once materialized;
    /// `2.5 * anticipated_decision(5)` is `2.5`; two distinct references keep `1.0` each; two
    /// equal `hydro_turbined(5, 0)` references merge into `1.0 + 1.0 = 2.0`.
    #[test]
    // Rationale: one table per rule set (every rejection message, every accepted form) reads
    // as the argument catalogue; splitting it would scatter the cases across tests.
    #[allow(clippy::too_many_lines)]
    fn variable_argument_rules_keep_their_messages() {
        for (src, fragment, message) in [
            (
                "hydro_turbined(5.5)",
                "hydro_turbined(5.5)",
                "entity_id must be a non-negative integer, got 5.5 in variable \"hydro_turbined\"",
            ),
            (
                "hydro_turbined(bus=2)",
                "hydro_turbined(bus=2)",
                "expected integer entity_id in variable \"hydro_turbined\"",
            ),
            (
                "hydro_turbined(5, bus=2, 0)",
                "hydro_turbined(5, bus=2, 0)",
                "positional block argument must precede the named \"bus=\" argument in variable \
                 \"hydro_turbined\"",
            ),
            (
                "2 * (hydro_turbined(5, 0, 1))",
                "(hydro_turbined(5, 0, 1))",
                "repeated block argument in variable \"hydro_turbined\"",
            ),
            (
                "hydro_turbined(5, 0.5)",
                "hydro_turbined(5, 0.5)",
                "block_id must be a non-negative integer, got 0.5 in variable \"hydro_turbined\"",
            ),
            (
                "hydro_turbined(5, bus=2, bus=3)",
                "hydro_turbined(5, bus=2, bus=3)",
                "repeated \"bus=\" argument in variable \"hydro_turbined\"",
            ),
            (
                "hydro_turbined(5, bus=2.5)",
                "hydro_turbined(5, bus=2.5)",
                "bus_id must be a non-negative integer, got 2.5 in variable \"hydro_turbined\"",
            ),
            (
                "hydro_generation(0, source_bus=3, target_bus=7)",
                "hydro_generation(0, source_bus=3, target_bus=7)",
                "the (source_bus=, target_bus=) pair form is only accepted on \"line_exchange\" \
                 as its sole argument, e.g. \"line_exchange(source_bus=3, target_bus=7)\"; it \
                 cannot follow a positional argument in \"hydro_generation\"",
            ),
            (
                "hydro_turbined(5, foo=2)",
                "hydro_turbined(5, foo=2)",
                "unknown named argument \"foo\" in variable \"hydro_turbined\": only \"bus\" is \
                 supported",
            ),
            (
                "hydro_generation(source_bus=3, target_bus=7)",
                "hydro_generation(source_bus=3, target_bus=7)",
                "the (source_bus=, target_bus=) pair form addresses a line by its endpoint buses \
                 and is only accepted on \"line_exchange\", not \"hydro_generation\"",
            ),
            (
                "line_exchange(source_bus=3, 7)",
                "line_exchange(source_bus=3, 7)",
                "expected \"source_bus\" or \"target_bus\" in \"line_exchange\" pair form",
            ),
            (
                "line_exchange(source_bus=3, bus=7)",
                "line_exchange(source_bus=3, bus=7)",
                "expected \"source_bus\" or \"target_bus\" in \"line_exchange\" pair form",
            ),
            (
                "line_exchange(source_bus=3.5, target_bus=7)",
                "line_exchange(source_bus=3.5, target_bus=7)",
                "source_bus must be a non-negative integer, got 3.5 in \"line_exchange\"",
            ),
            (
                "line_exchange(source_bus=3, source_bus=7)",
                "line_exchange(source_bus=3, source_bus=7)",
                "repeated \"source_bus=\" argument in \"line_exchange\"",
            ),
            (
                "line_exchange(target_bus=3, target_bus=7)",
                "line_exchange(target_bus=3, target_bus=7)",
                "repeated \"target_bus=\" argument in \"line_exchange\"",
            ),
            (
                "line_exchange(source_bus=3)",
                "line_exchange(source_bus=3)",
                "the \"line_exchange\" pair form requires both \"source_bus=\" and \"target_bus=\"; \
                 write \"line_exchange(source_bus=X, target_bus=Y)\"",
            ),
            (
                "line_exchange(target_bus=7)",
                "line_exchange(target_bus=7)",
                "the \"line_exchange\" pair form requires both \"source_bus=\" and \"target_bus=\"; \
                 write \"line_exchange(source_bus=X, target_bus=Y)\"",
            ),
            (
                "line_exchange(source_bus=1, target_bus=2)",
                "line_exchange(source_bus=1, target_bus=2)",
                "no line connects buses source_bus=1 and target_bus=2",
            ),
            (
                "hydro_storage(7, 0)",
                "hydro_storage(7, 0)",
                "variable \"hydro_storage\" does not accept a block argument",
            ),
            (
                "invalid_var(0)",
                "invalid_var(0)",
                "unknown variable name \"invalid_var\": not a recognised LP variable name",
            ),
            (
                "not_a_variable(5, bus=2)",
                "not_a_variable(5, bus=2)",
                "unknown variable name \"not_a_variable\": not a recognised LP variable name",
            ),
            (
                "hydro_storage(5, bus=2)",
                "hydro_storage(5, bus=2)",
                "variable \"hydro_storage\" does not accept a bus selector; only \
                 \"hydro_turbined\" and \"hydro_generation\" accept a \"bus=\" argument",
            ),
            (
                "anticipated_decision(5, 0)",
                "anticipated_decision(5, 0)",
                "variable \"anticipated_decision\" is a stage-level scalar and does not accept a \
                 block_id — write \"anticipated_decision(5)\", not \"anticipated_decision(5, ...)\"",
            ),
        ] {
            assert_eq!(
                linear_rejection(src),
                format!("`{fragment}`: {message}"),
                "{src:?}"
            );
        }
        for (src, name, message) in [
            (
                "hydro_spillage(5, bus=2)",
                "hydro_spillage",
                "does not accept a bus selector; only \"hydro_turbined\" and \"hydro_generation\" \
                 accept a \"bus=\" argument",
            ),
            (
                "hydro_outflow(5, bus=2)",
                "hydro_outflow",
                "does not accept a bus selector; only \"hydro_turbined\" and \"hydro_generation\" \
                 accept a \"bus=\" argument",
            ),
            ("hydro_turbined(5, 0, 0)", "hydro_turbined", ""),
            ("hydro_generation(5, 1, 2, bus=3)", "hydro_generation", ""),
            ("thermal_generation(0, 0, 0)", "thermal_generation", ""),
        ] {
            let expected = if message.is_empty() {
                format!("`{src}`: repeated block argument in variable \"{name}\"")
            } else {
                format!("`{src}`: variable \"{name}\" {message}")
            };
            assert_eq!(linear_rejection(src), expected, "{src:?}");
        }

        let turbined = |block_id, bus_id: Option<i32>| VariableRef::HydroTurbined {
            hydro_id: EntityId(5),
            block_id,
            bus_id: bus_id.map(EntityId),
        };
        let accepted: Vec<(&str, VariableRef)> = vec![
            (
                "hydro_storage(0)",
                VariableRef::HydroStorage {
                    hydro_id: EntityId(0),
                },
            ),
            (
                "hydro_turbined(0)",
                VariableRef::HydroTurbined {
                    hydro_id: EntityId(0),
                    block_id: None,
                    bus_id: None,
                },
            ),
            (
                "hydro_spillage(0)",
                VariableRef::HydroSpillage {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "hydro_diversion(0)",
                VariableRef::HydroDiversion {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "hydro_outflow(0)",
                VariableRef::HydroOutflow {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            ("hydro_generation(0)", hg(0)),
            (
                "hydro_evaporation(0)",
                VariableRef::HydroEvaporation {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "hydro_withdrawal(0)",
                VariableRef::HydroWithdrawal {
                    hydro_id: EntityId(0),
                },
            ),
            (
                "hydro_inflow(0)",
                VariableRef::HydroInflow {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            ("thermal_generation(0)", tg(0)),
            (
                "line_direct(0)",
                VariableRef::LineDirect {
                    line_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "line_reverse(0)",
                VariableRef::LineReverse {
                    line_id: EntityId(0),
                    block_id: None,
                },
            ),
            ("line_exchange(0)", exchange(0)),
            (
                "bus_deficit(0)",
                VariableRef::BusDeficit {
                    bus_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "bus_excess(0)",
                VariableRef::BusExcess {
                    bus_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "pumping_flow(0)",
                VariableRef::PumpingFlow {
                    station_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "pumping_power(0)",
                VariableRef::PumpingPower {
                    station_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "contract_import(0)",
                VariableRef::ContractImport {
                    contract_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "contract_export(0)",
                VariableRef::ContractExport {
                    contract_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "non_controllable_generation(0)",
                VariableRef::NonControllableGeneration {
                    source_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "non_controllable_curtailment(0)",
                VariableRef::NonControllableCurtailment {
                    source_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "anticipated_decision(5)",
                VariableRef::AnticipatedDecision {
                    thermal_id: EntityId(5),
                },
            ),
            (
                "anticipated_decision( 5 )",
                VariableRef::AnticipatedDecision {
                    thermal_id: EntityId(5),
                },
            ),
            ("hydro_turbined(5)", turbined(None, None)),
            ("hydro_turbined(5, 0)", turbined(Some(0), None)),
            ("hydro_turbined(5, bus=2)", turbined(None, Some(2))),
            ("hydro_turbined(5, 0, bus=2)", turbined(Some(0), Some(2))),
            ("hydro_turbined(5, 1, bus=2)", turbined(Some(1), Some(2))),
            (
                "line_exchange(0, 1)",
                VariableRef::LineExchange {
                    line_id: EntityId(0),
                    block_id: Some(1),
                },
            ),
            (
                "hydro_useful_volume_final(3, 0)",
                VariableRef::HydroUsefulVolumeFinal {
                    hydro_id: EntityId(3),
                    block_id: Some(0),
                },
            ),
        ];
        for (src, variable) in accepted {
            assert_eq!(
                literal_terms(src),
                [(variable, 1.0_f64.to_bits())],
                "{src:?}"
            );
        }

        let anticipated = |thermal_id| VariableRef::AnticipatedDecision {
            thermal_id: EntityId(thermal_id),
        };
        for (src, expected) in [
            ("2.5 * anticipated_decision(5)", vec![(anticipated(5), 2.5)]),
            (
                "anticipated_decision(5) + anticipated_decision(6)",
                vec![(anticipated(5), 1.0), (anticipated(6), 1.0)],
            ),
            (
                "hydro_turbined(5, 0) + hydro_turbined(5, 1)",
                vec![
                    (turbined(Some(0), None), 1.0),
                    (turbined(Some(1), None), 1.0),
                ],
            ),
            (
                "hydro_turbined(5, 0) + hydro_turbined(5, 0)",
                vec![(turbined(Some(0), None), 2.0)],
            ),
        ] {
            let expected: Vec<(VariableRef, u64)> = expected
                .into_iter()
                .map(|(variable, value)| (variable, f64::to_bits(value)))
                .collect();
            assert_eq!(literal_terms(src), expected, "{src:?}");
        }
    }

    /// derived: `@a * (h0 + h1)` gives each unit coefficient the scale itself, `a`.
    #[test]
    fn parameter_scaled_group_distributes_the_parameter() {
        assert_eq!(
            terms("@a * (hydro_generation(0) + hydro_generation(1))"),
            [(hg(0), a()), (hg(1), a())]
        );
    }

    /// derived: `2.0 * @a` folds first to the program `2, a, mul`, which each unit coefficient
    /// of the group takes as is.
    #[test]
    fn literal_times_parameter_scaled_group_distributes_the_product() {
        let scale = lit(2.0) * a();
        assert_eq!(
            terms("2.0 * @a * (hydro_generation(0) + hydro_generation(1))"),
            [(hg(0), scale.clone()), (hg(1), scale)]
        );
    }

    /// derived: `@a * @b * h0` is `a, b, mul`, at `a = 3.0`, `b = 0.5` equal to `1.5`.
    #[test]
    fn two_parameters_in_one_term_multiply() {
        let product = terms("@a * @b * hydro_generation(0)");
        assert_eq!(product, [(hg(0), a() * b())]);
        assert_eq!(
            product[0]
                .1
                .evaluate(|id| if id == EntityId(7) { 3.0 } else { 0.5 })
                .to_bits(),
            1.5_f64.to_bits()
        );
    }

    /// derived: the left side `@a * @b * h0` gives `h0` the program `a, b, mul`; the right side
    /// is the constant `0.0` with no term.
    #[test]
    fn parameter_product_times_a_variable_is_its_coefficient() {
        let (lhs, rhs) = sides("@a * @b * hydro_generation(0) >= 0");
        assert_eq!(
            lhs,
            LinearForm {
                terms: vec![(hg(0), Some(a() * b()))],
                constant: None,
            }
        );
        assert_eq!(
            rhs.constant
                .as_ref()
                .and_then(ScalarExpr::as_literal)
                .map(f64::to_bits),
            Some(0.0_f64.to_bits())
        );
        assert!(rhs.terms.is_empty());
    }

    /// derived: the right side `@a * @b * @c` is the constant program `a, b, mul, c, mul`.
    #[test]
    fn chained_parameter_product_lowers_to_one_constant() {
        let (lhs, rhs) = sides("hydro_generation(0) >= @a * @b * @c");
        assert_eq!(
            lhs,
            LinearForm {
                terms: vec![(hg(0), None)],
                constant: None,
            }
        );
        assert_eq!(
            rhs,
            LinearForm {
                terms: Vec::new(),
                constant: Some(a() * b() * ScalarExpr::param(EntityId(9))),
            }
        );
    }

    /// derived: the right side `2 * (@a * @b)` is the constant program `2, a, b, mul, mul`.
    #[test]
    fn parameter_product_inside_a_group_lowers_to_a_constant() {
        let (_, rhs) = sides("hydro_generation(0) >= 2 * (@a * @b)");
        assert_eq!(
            rhs,
            LinearForm {
                terms: Vec::new(),
                constant: Some(lit(2.0) * (a() * b())),
            }
        );
        assert_eq!(rhs.constant.as_ref().map(ScalarExpr::op_count), Some(5));
    }
}
