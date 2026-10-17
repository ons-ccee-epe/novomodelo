//! Scalar coefficient programs and the exact-zero rule.
//!
//! A [`ScalarExpr`] is a flat postfix program over literals and scalar-parameter
//! ids. A value can only be built through [`ScalarExpr::literal`],
//! [`ScalarExpr::param`] and the `std::ops` operators, and only
//! [`ScalarExpr::evaluate`] runs one, so a program's bits never depend on who
//! evaluates it.
//!
//! The program runs left to right on a value stack: a leaf pushes a value, a
//! unary op replaces the top, and a binary op pops the right operand, then the
//! left, and pushes `left op right`. An operator whose operands are all single
//! literals is folded into one literal with the same `f64` operation the
//! evaluator applies; no algebraic identity is ever applied, because `x + 0.0`
//! and `0.0 - x` differ from `x` and `-x` on signed zeros.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::EntityId;

/// Whether `v` is an exact zero: `+0.0` and `-0.0` both are, `NaN` never is.
#[must_use]
pub fn is_exact_zero(v: f64) -> bool {
    v == 0.0
}

/// A postfix program over `f64` literals and scalar-parameter ids.
///
/// Built by [`ScalarExpr::literal`], [`ScalarExpr::param`] and the `Neg`, `Add`,
/// `Sub`, `Mul` and `Div` operators, which append to the left operand's program
/// and fold literal-only operands. [`ScalarExpr::evaluate`] is the single
/// evaluator.
///
/// # Examples
///
/// ```
/// use novomodelo_core::{EntityId, ScalarExpr};
///
/// let expr = ScalarExpr::literal(2.0) * ScalarExpr::param(EntityId(7));
/// assert_eq!(expr.evaluate(|_| 3.0), 6.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarExpr {
    ops: Vec<ScalarOp>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ScalarOp {
    Literal(f64),
    Param(EntityId),
    Neg,
    Add,
    Sub,
    Mul,
    Div,
}

impl ScalarExpr {
    /// A program holding the single literal `value`.
    #[must_use]
    pub fn literal(value: f64) -> Self {
        Self {
            ops: vec![ScalarOp::Literal(value)],
        }
    }

    /// A program holding the single scalar-parameter reference `id`.
    #[must_use]
    pub fn param(id: EntityId) -> Self {
        Self {
            ops: vec![ScalarOp::Param(id)],
        }
    }

    /// Runs the program in order and returns its value, `leaf` mapping each
    /// parameter id to its value.
    ///
    /// Returns `NaN` for a malformed stack, which no constructed program produces.
    #[must_use]
    pub fn evaluate(&self, leaf: impl Fn(EntityId) -> f64) -> f64 {
        match self.ops.as_slice() {
            [ScalarOp::Literal(v)] => return *v,
            [ScalarOp::Param(id)] => return leaf(*id),
            _ => {}
        }
        let mut stack: Vec<f64> = Vec::with_capacity(self.op_count());
        for op in &self.ops {
            let value = match *op {
                ScalarOp::Literal(v) => v,
                ScalarOp::Param(id) => leaf(id),
                ScalarOp::Neg => {
                    let Some(a) = stack.pop() else {
                        return f64::NAN;
                    };
                    -a
                }
                ScalarOp::Add => {
                    let (Some(b), Some(a)) = (stack.pop(), stack.pop()) else {
                        return f64::NAN;
                    };
                    a + b
                }
                ScalarOp::Sub => {
                    let (Some(b), Some(a)) = (stack.pop(), stack.pop()) else {
                        return f64::NAN;
                    };
                    a - b
                }
                ScalarOp::Mul => {
                    let (Some(b), Some(a)) = (stack.pop(), stack.pop()) else {
                        return f64::NAN;
                    };
                    a * b
                }
                ScalarOp::Div => {
                    let (Some(b), Some(a)) = (stack.pop(), stack.pop()) else {
                        return f64::NAN;
                    };
                    a / b
                }
            };
            stack.push(value);
        }
        match stack.as_slice() {
            [v] => *v,
            _ => f64::NAN,
        }
    }

    /// Parameter ids in program order, duplicates kept.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn params(&self) -> impl Iterator<Item = EntityId> + '_ {
        self.ops.iter().filter_map(|op| match op {
            ScalarOp::Param(id) => Some(*id),
            _ => None,
        })
    }

    /// Whether the program references no parameter.
    #[must_use]
    pub fn is_constant(&self) -> bool {
        !self.ops.iter().any(|op| matches!(op, ScalarOp::Param(_)))
    }

    /// The value when the program is exactly one literal.
    #[must_use]
    pub fn as_literal(&self) -> Option<f64> {
        match self.ops.as_slice() {
            [ScalarOp::Literal(v)] => Some(*v),
            _ => None,
        }
    }

    /// The id when the program is exactly one parameter reference.
    #[must_use]
    pub fn as_param(&self) -> Option<EntityId> {
        match self.ops.as_slice() {
            [ScalarOp::Param(id)] => Some(*id),
            _ => None,
        }
    }

    /// Number of ops in the program.
    #[must_use]
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }
}

fn binary(
    mut lhs: ScalarExpr,
    rhs: ScalarExpr,
    op: ScalarOp,
    apply: fn(f64, f64) -> f64,
) -> ScalarExpr {
    if let (Some(a), Some(b)) = (lhs.as_literal(), rhs.as_literal()) {
        return ScalarExpr::literal(apply(a, b));
    }
    lhs.ops.extend(rhs.ops);
    lhs.ops.push(op);
    lhs
}

impl Neg for ScalarExpr {
    type Output = ScalarExpr;

    fn neg(mut self) -> ScalarExpr {
        if let Some(a) = self.as_literal() {
            return ScalarExpr::literal(-a);
        }
        self.ops.push(ScalarOp::Neg);
        self
    }
}

impl Add for ScalarExpr {
    type Output = ScalarExpr;

    fn add(self, rhs: ScalarExpr) -> ScalarExpr {
        binary(self, rhs, ScalarOp::Add, |a, b| a + b)
    }
}

impl Sub for ScalarExpr {
    type Output = ScalarExpr;

    fn sub(self, rhs: ScalarExpr) -> ScalarExpr {
        binary(self, rhs, ScalarOp::Sub, |a, b| a - b)
    }
}

impl Mul for ScalarExpr {
    type Output = ScalarExpr;

    fn mul(self, rhs: ScalarExpr) -> ScalarExpr {
        binary(self, rhs, ScalarOp::Mul, |a, b| a * b)
    }
}

impl Div for ScalarExpr {
    type Output = ScalarExpr;

    fn div(self, rhs: ScalarExpr) -> ScalarExpr {
        binary(self, rhs, ScalarOp::Div, |a, b| a / b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: EntityId = EntityId(1);
    const B: EntityId = EntityId(2);
    const C: EntityId = EntityId(3);

    fn lit(v: f64) -> ScalarExpr {
        ScalarExpr::literal(v)
    }

    fn par(id: EntityId) -> ScalarExpr {
        ScalarExpr::param(id)
    }

    fn values(id: EntityId) -> f64 {
        match id {
            A => 5.0,
            B => 2.0,
            C => 4.0,
            _ => f64::NAN,
        }
    }

    #[test]
    fn is_exact_zero_matches_both_signed_zeros_only() {
        assert!(is_exact_zero(0.0));
        assert!(is_exact_zero(-0.0));
        assert!(!is_exact_zero(f64::NAN));
        assert!(!is_exact_zero(f64::MIN_POSITIVE));
        assert!(!is_exact_zero(5e-324));
        assert!(!is_exact_zero(1.0));
    }

    /// derived: a single-leaf program returns its leaf unchanged; a negation
    /// flips the leaf's sign.
    #[test]
    fn literal_and_param_evaluate_to_their_values() {
        assert_eq!(lit(2.5).evaluate(|_| f64::NAN), 2.5);
        assert_eq!(par(A).evaluate(values), 5.0);
        assert_eq!(par(EntityId(99)).evaluate(|id| f64::from(id.0)), 99.0);
        assert_eq!((-par(A)).evaluate(values), -5.0);
    }

    /// derived: with a = 5, b = 2, c = 4, `(a - b) / c` is 0.75 while
    /// `a - (b / c)` is 4.5 and `b - a` is -3; `(2 * a) * 3` is 30 evaluated as
    /// written.
    #[test]
    fn binary_operators_evaluate_left_operand_first() {
        assert_eq!((par(A) - par(B)).evaluate(values), 3.0);
        assert_eq!((par(B) - par(A)).evaluate(values), -3.0);
        assert_eq!((par(A) / par(B)).evaluate(values), 2.5);
        assert_eq!(((par(A) - par(B)) / par(C)).evaluate(values), 0.75);

        let scaled = (lit(2.0) * par(A)) * lit(3.0);
        assert_eq!(scaled.op_count(), 5);
        assert_eq!(
            scaled.evaluate(values).to_bits(),
            ((2.0 * 5.0) * 3.0_f64).to_bits()
        );
    }

    /// derived: `2 * 3` folds to the literal 6, `5 - 2` to 3; `1 / 0` folds to infinity; a
    /// folded operand next to a parameter leaves one literal op, and a folded
    /// negation keeps the sign of zero.
    #[test]
    fn constructors_fold_literal_only_operands() {
        let product = lit(2.0) * lit(3.0);
        assert_eq!(product.as_literal(), Some(6.0));
        assert_eq!(product.op_count(), 1);

        assert_eq!((lit(5.0) - lit(2.0)).as_literal(), Some(3.0));
        assert_eq!((lit(1.0) / lit(0.0)).as_literal(), Some(f64::INFINITY));
        assert_eq!((-lit(2.0)).as_literal(), Some(-2.0));
        assert_eq!(
            (-lit(0.0)).as_literal().map(f64::to_bits),
            Some((-0.0_f64).to_bits())
        );
        assert_eq!((-(lit(1.0) + lit(2.0))).as_literal(), Some(-3.0));

        let left_folded = (lit(2.0) * lit(3.0)) * par(A);
        assert_eq!(left_folded.op_count(), 3);
        assert_eq!(left_folded.evaluate(values), 30.0);

        let right_folded = par(A) * (lit(2.0) + lit(3.0));
        assert_eq!(right_folded.op_count(), 3);
        assert_eq!(right_folded.evaluate(values), 25.0);
    }

    /// derived: `0 - x` at `x = +0.0` is `+0.0` (a rewrite to `-x` would give
    /// `-0.0`); `x + 0`, `1 * x` and `x / 1` keep their literal operand.
    #[test]
    fn constructors_apply_no_algebraic_identity() {
        let negated_zero = lit(0.0) - par(A);
        assert_eq!(negated_zero.op_count(), 3);
        assert_eq!(
            negated_zero.evaluate(|_| 0.0).to_bits(),
            (0.0_f64).to_bits()
        );

        assert_eq!((par(A) + lit(0.0)).op_count(), 3);
        assert_eq!((lit(1.0) * par(A)).op_count(), 3);
        assert_eq!((par(A) / lit(1.0)).op_count(), 3);
        assert_eq!((-(-par(A))).op_count(), 3);
    }

    /// derived: `a + a * b` lists a, a, b — leaves in program order, the repeated
    /// id kept, operators skipped.
    #[test]
    fn params_lists_leaves_in_program_order_with_duplicates() {
        let expr = par(A) + par(A) * par(B);
        assert_eq!(expr.params().collect::<Vec<_>>(), vec![A, A, B]);
        assert_eq!((lit(2.0) * lit(3.0)).params().count(), 0);
    }

    /// derived: only a one-op program is a literal or a parameter; a constant is
    /// a program without a `Param` op.
    #[test]
    fn queries_classify_single_leaf_programs() {
        let literal = lit(4.0);
        assert!(literal.is_constant());
        assert_eq!(literal.as_literal(), Some(4.0));
        assert_eq!(literal.as_param(), None);
        assert_eq!(literal.op_count(), 1);

        let param = par(A);
        assert!(!param.is_constant());
        assert_eq!(param.as_literal(), None);
        assert_eq!(param.as_param(), Some(A));
        assert_eq!(param.op_count(), 1);

        let compound = lit(1.0) + par(A);
        assert!(!compound.is_constant());
        assert_eq!(compound.as_literal(), None);
        assert_eq!(compound.as_param(), None);
        assert_eq!(compound.op_count(), 3);
    }
}
