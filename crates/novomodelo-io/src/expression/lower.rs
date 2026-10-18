//! Lowers a parsed expression to a [`ScalarExpr`] program. The program's constructors run in
//! source order, so a sub-expression without a parameter folds to one literal and nothing is
//! reassociated. A fold that divides by an exact zero or yields a non-finite value is rejected
//! here, where the source span is still known; a divisor or value that reads a parameter is
//! left to whoever supplies the parameter values.

use std::collections::HashMap;
use std::ops::Range;

use novomodelo_core::{EntityId, ScalarExpr, is_exact_zero};

use super::Diagnostic;
use super::parser::{AddOp, Expr, ExprKind, MulOp, parse};

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

    use novomodelo_core::{EntityId, ScalarExpr};

    use super::lower_scalar_body;

    fn params() -> HashMap<String, EntityId> {
        HashMap::from([("a".to_owned(), EntityId(7)), ("b".to_owned(), EntityId(8))])
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
}
