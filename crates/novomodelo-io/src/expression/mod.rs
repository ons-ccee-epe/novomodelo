//! The generic-constraint expression language. [`catalog`] maps a variable name and its
//! arguments to a [`VariableRef`](novomodelo_core::VariableRef); [`lexer`] splits a source
//! string into spanned tokens; [`parser`] builds their syntax tree; [`lower`] turns a tree into
//! a [`ScalarExpr`](novomodelo_core::ScalarExpr) program.
//!
//! # Syntax
//!
//! An operand is a number, a `@name` reference to a scalar parameter or a named expression, a
//! catalog variable call such as `hydro_generation(0)`, `hydro_turbined(5, bus=2)`,
//! `hydro_generation(5, unit_group=2)` or `line_flow(source_bus=3, target_bus=7)`, a conditional
//! `if(cond, X, Y)`, or a parenthesized expression. A number is a finite, non-negative literal; a
//! sign is a unary operator. The operators are `+`, `-`, `*`, `/` and the unary signs `+` and `-`:
//! a unary sign binds tightest, `*` and `/` bind tighter than `+` and `-`, and every binary
//! operator is left-associative. An expression carries at most one relational operator, `<=`, `>=`
//! or `==`, outside any parentheses.
//!
//! # Conditions
//!
//! A condition is a disjunction of conjunctions of comparisons. A comparison relates two
//! expressions with `<`, `<=`, `>`, `>=` or `==` and is neither parenthesized nor chained. `&`
//! joins comparisons and binds tighter than `|`, and both fold to the left, so
//! `@a <= 1 & @b <= 2 | @c <= 3` reads as `(@a <= 1 & @b <= 2) | @c <= 3`. `&`, `|`, `<` and
//! `>` are not part of the linear-expression grammar and are rejected there, except in the
//! condition of `if()`.
//!
//! `if(cond, X, Y)` takes the value of `X` when the condition holds and of `Y` otherwise. It
//! takes exactly three arguments: a condition and two expressions. Parsing only builds the node.
//! A constraint lowers it to a selector; the body of a scalar parameter rejects it.
//!
//! # Typing
//!
//! Every node types as a constant or a linear form over LP variables before anything folds. A
//! constant times a form in either order, a form divided by a constant, and a sum of forms type
//! as linear. A product of two forms and a division by a form are rejected. A scalar-parameter
//! body reads neither an LP variable nor a named expression.
//!
//! # Relations
//!
//! A relation moves every variable term to the left and the constant to the right: the
//! remainder is the right side's constant minus the left side's. `<=` assigns it to the upper
//! bound, `>=` to the lower bound and `==` to both. An expression without a relational operator
//! is the left-hand side alone and cannot carry a constant.
//!
//! # Named expressions
//!
//! A file's `expressions` array binds names to expressions. A `@name` reference lowers to its
//! body's form, each body at most once per constraint, and a reference cycle is rejected at
//! load. A body that reads no LP variable is data: a reference to it lowers as its body written
//! in place, so its value moves into the bounds.
//!
//! # Scenario data
//!
//! A constraint scope resolves data terms. A data term such as `bus_load_value(1)` is a scalar
//! parameter standing for a scenario value. The body of a scalar parameter reads no scenario
//! data.
//!
//! `if(cond, X, Y)` lowers to a selector, a scalar parameter numbered by the caller that stands
//! for the branch the condition picks. The condition reads only data. A branch that reads LP
//! variables must read the same ones as the other, so a selector replaces only a coefficient or
//! a constant and the structure of the constraint does not depend on the condition.
//!
//! # Budgets
//!
//! Nesting of parentheses, unary operators, `if()` calls and chained `&`/`|` connectives is
//! capped at [`parser::MAX_NESTING_DEPTH`]. A form whose programs would hold more than
//! [`lower::MAX_INLINED_OPS`] operations is refused, which bounds named-expression expansion.

use std::ops::Range;

pub(crate) mod catalog;
pub(crate) mod lexer;
pub(crate) mod lower;
pub(crate) mod parser;

/// A rejection of an expression, located by a byte range of its source.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Diagnostic {
    pub(crate) span: Range<usize>,
    pub(crate) message: String,
}

impl Diagnostic {
    /// Quotes the offending fragment of `src` before the message, or says the problem is at
    /// the end of the expression when the span selects nothing.
    pub(crate) fn render(&self, src: &str) -> String {
        match src.get(self.span.clone()) {
            Some(fragment) if !fragment.is_empty() => format!("`{fragment}`: {}", self.message),
            _ => format!("{} (at the end of the expression)", self.message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Diagnostic;

    #[test]
    fn diagnostic_renders_the_quoted_fragment() {
        let diagnostic = Diagnostic {
            span: 2..6,
            message: "unknown parameter".to_owned(),
        };
        assert_eq!(diagnostic.render("1+@vaz + 2"), "`@vaz`: unknown parameter");
        let diagnostic = Diagnostic {
            span: 0..7,
            message: "unknown parameter".to_owned(),
        };
        assert_eq!(diagnostic.render("@vazão"), "`@vazão`: unknown parameter");
    }

    #[test]
    fn diagnostic_at_the_end_renders_without_a_fragment() {
        let message = "expected an operand".to_owned();
        for span in [3..3, 3..9, 9..12] {
            let diagnostic = Diagnostic {
                span,
                message: message.clone(),
            };
            assert_eq!(
                diagnostic.render("x +"),
                "expected an operand (at the end of the expression)"
            );
        }
    }
}
