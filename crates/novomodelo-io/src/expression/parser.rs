//! Builds the syntax tree of an expression source; names, types and folding are left to the
//! caller.

use std::iter::Peekable;
use std::ops::Range;
use std::vec;

use super::Diagnostic;
use super::lexer::{Token, TokenKind, tokenize};

/// The deepest nesting of parentheses and unary operators [`parse`] accepts; the bound keeps
/// the recursive descent within the stack.
pub(crate) const MAX_NESTING_DEPTH: usize = 256;

/// A parsed source: a plain expression, or two sides of one relational operator.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Relation {
    pub(crate) lhs: Expr,
    pub(crate) rhs: Option<(RelOp, Expr)>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RelOp {
    Le,
    Ge,
    Eq,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Expr {
    pub(crate) kind: ExprKind,
    /// Byte range of the node's full source text, parentheses included.
    pub(crate) span: Range<usize>,
}

/// A `Sum` or `Product` holds a whole same-precedence chain in source order and always has a
/// non-empty tail.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ExprKind {
    Number(f64),
    /// A `@name` parameter reference, without the `@`.
    Name(String),
    Variable(VariableCall),
    Neg(Box<Expr>),
    Sum {
        head: Box<Expr>,
        tail: Vec<(AddOp, Expr)>,
    },
    Product {
        head: Box<Expr>,
        tail: Vec<(MulOp, Expr)>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AddOp {
    Add,
    Sub,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum MulOp {
    Mul,
    Div,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VariableCall {
    pub(crate) name: String,
    pub(crate) args: Vec<Arg>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Arg {
    Positional {
        value: f64,
        span: Range<usize>,
    },
    Named {
        key: String,
        value: f64,
        span: Range<usize>,
    },
}

/// Parses `src` into a [`Relation`], rejecting the first token that does not fit the grammar.
pub(crate) fn parse(src: &str) -> Result<Relation, Diagnostic> {
    let tokens = tokenize(src)?;
    if tokens.is_empty() {
        return Err(Diagnostic {
            span: 0..src.len(),
            message: "expression must not be empty".to_owned(),
        });
    }
    Parser {
        src,
        tokens: tokens.into_iter().peekable(),
        last_end: 0,
        depth: 0,
        groups: 0,
    }
    .relation()
}

struct Parser<'a> {
    src: &'a str,
    tokens: Peekable<vec::IntoIter<Token>>,
    last_end: usize,
    /// Open parentheses plus unary operators on the path to the current node.
    depth: usize,
    /// Open parentheses only; a relational operator is rejected while any is open.
    groups: usize,
}

impl Parser<'_> {
    fn relation(mut self) -> Result<Relation, Diagnostic> {
        let lhs = self.expr()?;
        let rhs = match self.bump_op(rel_op) {
            Some(op) => Some((op, self.expr()?)),
            None => None,
        };
        let Some(token) = self.bump() else {
            return Ok(Relation { lhs, rhs });
        };
        let message = if rel_op(&token.kind).is_some() {
            "a generic constraint expression carries at most one relational operator (<=, >=, \
             ==); an inline double-relational range is not supported — split into two \
             constraints"
                .to_owned()
        } else {
            format!(
                "unexpected {} after a complete expression",
                self.describe(&token)
            )
        };
        Err(Diagnostic {
            span: token.span,
            message,
        })
    }

    fn expr(&mut self) -> Result<Expr, Diagnostic> {
        let head = self.product()?;
        let mut tail = Vec::new();
        while let Some(op) = self.bump_op(add_op) {
            tail.push((op, self.product()?));
        }
        Ok(chain(head, tail, |head, tail| ExprKind::Sum { head, tail }))
    }

    fn product(&mut self) -> Result<Expr, Diagnostic> {
        let head = self.unary()?;
        let mut tail = Vec::new();
        while let Some(op) = self.bump_op(mul_op) {
            tail.push((op, self.unary()?));
        }
        Ok(chain(head, tail, |head, tail| ExprKind::Product {
            head,
            tail,
        }))
    }

    fn unary(&mut self) -> Result<Expr, Diagnostic> {
        let Some(sign) = self.bump_if(|kind| matches!(kind, TokenKind::Plus | TokenKind::Minus))
        else {
            return self.primary();
        };
        self.enter(&sign.span)?;
        let operand = self.unary()?;
        self.depth -= 1;
        let span = sign.span.start..operand.span.end;
        Ok(if sign.kind == TokenKind::Minus {
            Expr {
                kind: ExprKind::Neg(Box::new(operand)),
                span,
            }
        } else {
            Expr { span, ..operand }
        })
    }

    fn primary(&mut self) -> Result<Expr, Diagnostic> {
        let Some(token) = self.bump() else {
            return Err(self.at_end("expected an operand"));
        };
        match token.kind {
            TokenKind::Number(value) => Ok(Expr {
                kind: ExprKind::Number(value),
                span: token.span,
            }),
            TokenKind::ParamRef(name) => Ok(Expr {
                kind: ExprKind::Name(name),
                span: token.span,
            }),
            TokenKind::Ident(name) => self.variable(name, token.span),
            TokenKind::LParen => self.group(token.span),
            _ => Err(self.unexpected(&token, self.groups > 0)),
        }
    }

    fn group(&mut self, open: Range<usize>) -> Result<Expr, Diagnostic> {
        self.enter(&open)?;
        self.groups += 1;
        let inner = self.expr()?;
        let close = self.expect_close(&open)?;
        self.groups -= 1;
        self.depth -= 1;
        Ok(Expr {
            kind: inner.kind,
            span: open.start..close.end,
        })
    }

    fn variable(&mut self, name: String, name_span: Range<usize>) -> Result<Expr, Diagnostic> {
        let Some(open) = self.bump_if(|kind| *kind == TokenKind::LParen) else {
            return Err(Diagnostic {
                span: name_span,
                message: format!("expected '(' after variable name \"{name}\""),
            });
        };
        let mut args = Vec::new();
        let close = loop {
            let token = self.next_in_group(&open.span)?;
            let arg = match token.kind {
                TokenKind::RParen if args.is_empty() => {
                    return Err(Diagnostic {
                        span: name_span.start..token.span.end,
                        message: format!("variable \"{name}\" has an empty argument list"),
                    });
                }
                TokenKind::Number(value) => Arg::Positional {
                    value,
                    span: token.span,
                },
                TokenKind::Ident(key) => {
                    let equals = self.next_in_group(&open.span)?;
                    if equals.kind != TokenKind::Equals {
                        return Err(self.unexpected(&equals, true));
                    }
                    let number = self.next_in_group(&open.span)?;
                    let TokenKind::Number(value) = number.kind else {
                        return Err(self.unexpected(&number, true));
                    };
                    Arg::Named {
                        key,
                        value,
                        span: token.span.start..number.span.end,
                    }
                }
                _ => return Err(self.unexpected(&token, true)),
            };
            args.push(arg);
            let separator = self.next_in_group(&open.span)?;
            match separator.kind {
                TokenKind::Comma => {}
                TokenKind::RParen => break separator.span,
                _ => return Err(self.unexpected(&separator, true)),
            }
        };
        Ok(Expr {
            kind: ExprKind::Variable(VariableCall { name, args }),
            span: name_span.start..close.end,
        })
    }

    fn expect_close(&mut self, open: &Range<usize>) -> Result<Range<usize>, Diagnostic> {
        let token = self.next_in_group(open)?;
        if token.kind == TokenKind::RParen {
            Ok(token.span)
        } else {
            Err(self.unexpected(&token, true))
        }
    }

    /// The next token inside the parentheses opened at `open`; the end of input there is a
    /// missing `)`.
    fn next_in_group(&mut self, open: &Range<usize>) -> Result<Token, Diagnostic> {
        self.bump().ok_or_else(|| Diagnostic {
            span: open.start..self.last_end,
            message: "missing closing parenthesis".to_owned(),
        })
    }

    fn enter(&mut self, span: &Range<usize>) -> Result<(), Diagnostic> {
        self.depth += 1;
        if self.depth > MAX_NESTING_DEPTH {
            return Err(Diagnostic {
                span: span.clone(),
                message: format!("expression nests deeper than {MAX_NESTING_DEPTH} levels"),
            });
        }
        Ok(())
    }

    fn bump(&mut self) -> Option<Token> {
        let token = self.tokens.next()?;
        self.last_end = token.span.end;
        Some(token)
    }

    fn bump_if(&mut self, accepts: impl Fn(&TokenKind) -> bool) -> Option<Token> {
        let token = self.tokens.next_if(|token| accepts(&token.kind))?;
        self.last_end = token.span.end;
        Some(token)
    }

    fn bump_op<Op>(&mut self, op: impl Fn(&TokenKind) -> Option<Op>) -> Option<Op> {
        let op = op(&self.tokens.peek()?.kind)?;
        self.bump();
        Some(op)
    }

    fn unexpected(&self, token: &Token, in_parentheses: bool) -> Diagnostic {
        let message = if in_parentheses && rel_op(&token.kind).is_some() {
            "a relational operator is only allowed at the top level".to_owned()
        } else {
            format!("unexpected {}", self.describe(token))
        };
        Diagnostic {
            span: token.span.clone(),
            message,
        }
    }

    fn at_end(&self, message: &str) -> Diagnostic {
        Diagnostic {
            span: self.src.len()..self.src.len(),
            message: message.to_owned(),
        }
    }

    fn describe(&self, token: &Token) -> String {
        let text = self.src.get(token.span.clone()).unwrap_or_default();
        match &token.kind {
            TokenKind::Number(_) => format!("number {text}"),
            TokenKind::Ident(name) => format!("name \"{name}\""),
            TokenKind::ParamRef(name) => format!("parameter \"@{name}\""),
            _ => format!("'{text}'"),
        }
    }
}

fn add_op(kind: &TokenKind) -> Option<AddOp> {
    match kind {
        TokenKind::Plus => Some(AddOp::Add),
        TokenKind::Minus => Some(AddOp::Sub),
        _ => None,
    }
}

fn mul_op(kind: &TokenKind) -> Option<MulOp> {
    match kind {
        TokenKind::Star => Some(MulOp::Mul),
        TokenKind::Slash => Some(MulOp::Div),
        _ => None,
    }
}

fn rel_op(kind: &TokenKind) -> Option<RelOp> {
    match kind {
        TokenKind::Le => Some(RelOp::Le),
        TokenKind::Ge => Some(RelOp::Ge),
        TokenKind::EqEq => Some(RelOp::Eq),
        _ => None,
    }
}

/// Returns `head` alone for an empty `tail`, else one node spanning the whole chain.
fn chain<Op>(
    head: Expr,
    tail: Vec<(Op, Expr)>,
    node: impl FnOnce(Box<Expr>, Vec<(Op, Expr)>) -> ExprKind,
) -> Expr {
    let Some(end) = tail.last().map(|(_, last)| last.span.end) else {
        return head;
    };
    let span = head.span.start..end;
    Expr {
        kind: node(Box::new(head), tail),
        span,
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use std::ops::Range;

    use super::{AddOp, Arg, Expr, ExprKind, MAX_NESTING_DEPTH, MulOp, RelOp, parse};

    /// Renders a tree without spans: `(sum h +x -y)`, `(prod h *x /y)`, `(neg x)`, `@name`,
    /// `name(1,key=2)`.
    fn shape(expr: &Expr) -> String {
        match &expr.kind {
            ExprKind::Number(value) => format!("{value}"),
            ExprKind::Name(name) => format!("@{name}"),
            ExprKind::Variable(call) => {
                let args: Vec<String> = call
                    .args
                    .iter()
                    .map(|arg| match arg {
                        Arg::Positional { value, .. } => format!("{value}"),
                        Arg::Named { key, value, .. } => format!("{key}={value}"),
                    })
                    .collect();
                format!("{}({})", call.name, args.join(","))
            }
            ExprKind::Neg(inner) => format!("(neg {})", shape(inner)),
            ExprKind::Sum { head, tail } => {
                let mut text = format!("(sum {}", shape(head));
                for (op, operand) in tail {
                    let sign = match op {
                        AddOp::Add => '+',
                        AddOp::Sub => '-',
                    };
                    text.push(' ');
                    text.push(sign);
                    text.push_str(&shape(operand));
                }
                text + ")"
            }
            ExprKind::Product { head, tail } => {
                let mut text = format!("(prod {}", shape(head));
                for (op, operand) in tail {
                    let sign = match op {
                        MulOp::Mul => '*',
                        MulOp::Div => '/',
                    };
                    text.push(' ');
                    text.push(sign);
                    text.push_str(&shape(operand));
                }
                text + ")"
            }
        }
    }

    fn plain(src: &str) -> Expr {
        let relation = parse(src).unwrap();
        assert_eq!(relation.rhs, None, "{src:?} has no relational operator");
        relation.lhs
    }

    fn fragment<'a>(src: &'a str, span: &Range<usize>) -> &'a str {
        src.get(span.clone()).unwrap()
    }

    fn rejection(src: &str) -> (String, String) {
        let err = parse(src).unwrap_err();
        (fragment(src, &err.span).to_owned(), err.message)
    }

    fn depth_message() -> String {
        let message = format!("expression nests deeper than {MAX_NESTING_DEPTH} levels");
        assert_eq!(message, "expression nests deeper than 256 levels");
        message
    }

    #[test]
    fn products_bind_tighter_than_sums() {
        let src = "1 + 2 * f(1) - @a / 3";
        let expr = plain(src);
        assert_eq!(shape(&expr), "(sum 1 +(prod 2 *f(1)) -(prod @a /3))");
        assert_eq!(fragment(src, &expr.span), src);
        let ExprKind::Sum { tail, .. } = &expr.kind else {
            panic!("expected a sum for {src:?}");
        };
        let spans: Vec<&str> = tail
            .iter()
            .map(|(_, operand)| fragment(src, &operand.span))
            .collect();
        assert_eq!(spans, ["2 * f(1)", "@a / 3"]);

        let src = "2 * 3 + 4";
        let expr = plain(src);
        assert_eq!(shape(&expr), "(sum (prod 2 *3) +4)");
        let ExprKind::Sum { head, .. } = &expr.kind else {
            panic!("expected a sum for {src:?}");
        };
        assert_eq!(fragment(src, &head.span), "2 * 3");
    }

    #[test]
    fn chains_are_flat_and_left_to_right() {
        for (src, expected) in [
            ("@a - @b - @c", "(sum @a -@b -@c)"),
            ("@a + @b - @c + 1", "(sum @a +@b -@c +1)"),
            ("2 * @a * 3", "(prod 2 *@a *3)"),
            ("@x / 2 / 3", "(prod @x /2 /3)"),
            ("@x / 2 * 3", "(prod @x /2 *3)"),
        ] {
            let expr = plain(src);
            assert_eq!(shape(&expr), expected, "shape of {src:?}");
            assert_eq!(fragment(src, &expr.span), src, "span of {src:?}");
        }
    }

    #[test]
    fn unary_minus_binds_tighter_than_products() {
        for (src, expected, head_fragment) in [
            ("-2 * @x", "(prod (neg 2) *@x)", "-2"),
            ("+2 * @x", "(prod 2 *@x)", "+2"),
            ("- -@x * 3", "(prod (neg (neg @x)) *3)", "- -@x"),
        ] {
            let expr = plain(src);
            assert_eq!(shape(&expr), expected, "shape of {src:?}");
            let ExprKind::Product { head, .. } = &expr.kind else {
                panic!("expected a product for {src:?}");
            };
            assert_eq!(fragment(src, &head.span), head_fragment, "head of {src:?}");
        }
        assert_eq!(shape(&plain("2 * -@x")), "(prod 2 *(neg @x))");
        assert_eq!(shape(&plain("@a - -1")), "(sum @a -(neg 1))");
    }

    #[test]
    fn parentheses_widen_the_span() {
        let src = "( @a + 1 ) * 2";
        let expr = plain(src);
        assert_eq!(shape(&expr), "(prod (sum @a +1) *2)");
        assert_eq!(fragment(src, &expr.span), src);
        let ExprKind::Product { head, .. } = &expr.kind else {
            panic!("expected a product for {src:?}");
        };
        assert_eq!(fragment(src, &head.span), "( @a + 1 )");

        let src = "((@a))";
        let expr = plain(src);
        assert_eq!(shape(&expr), "@a");
        assert_eq!(fragment(src, &expr.span), src);

        let src = "-(f(1))";
        let expr = plain(src);
        let ExprKind::Neg(inner) = &expr.kind else {
            panic!("expected a negation for {src:?}");
        };
        assert_eq!(fragment(src, &inner.span), "(f(1))");
    }

    #[test]
    fn variable_arguments_parse_in_every_form() {
        for (src, expected, arg_fragments) in [
            ("f(1)", "f(1)", vec!["1"]),
            ("f(1, 2)", "f(1,2)", vec!["1", "2"]),
            ("f(1, bus=3)", "f(1,bus=3)", vec!["1", "bus=3"]),
            (
                "f(1, 2, bus = 3)",
                "f(1,2,bus=3)",
                vec!["1", "2", "bus = 3"],
            ),
            (
                "line_exchange(source_bus=1, target_bus=2)",
                "line_exchange(source_bus=1,target_bus=2)",
                vec!["source_bus=1", "target_bus=2"],
            ),
        ] {
            let expr = plain(src);
            assert_eq!(shape(&expr), expected, "shape of {src:?}");
            assert_eq!(fragment(src, &expr.span), src, "span of {src:?}");
            let ExprKind::Variable(call) = &expr.kind else {
                panic!("expected a variable for {src:?}");
            };
            let spans: Vec<&str> = call
                .args
                .iter()
                .map(|arg| match arg {
                    Arg::Positional { span, .. } | Arg::Named { span, .. } => fragment(src, span),
                })
                .collect();
            assert_eq!(spans, arg_fragments, "argument spans of {src:?}");
        }

        let src = "hydro_turbined(5, bus=2) <= 10";
        let relation = parse(src).unwrap();
        assert_eq!(shape(&relation.lhs), "hydro_turbined(5,bus=2)");
        let Some((op, rhs)) = &relation.rhs else {
            panic!("expected a relation for {src:?}");
        };
        assert_eq!(*op, RelOp::Le);
        assert_eq!(shape(rhs), "10");
    }

    #[test]
    fn one_relational_operator_splits_the_relation() {
        let src = "hydro_generation(10) + hydro_generation(11)";
        assert_eq!(
            shape(&plain(src)),
            "(sum hydro_generation(10) +hydro_generation(11))"
        );

        for (src, expected_op) in [
            ("hydro_generation(0) <= 5", RelOp::Le),
            ("hydro_generation(0) >= 5", RelOp::Ge),
            ("hydro_generation(0) == 5", RelOp::Eq),
        ] {
            let relation = parse(src).unwrap();
            assert_eq!(fragment(src, &relation.lhs.span), "hydro_generation(0)");
            let Some((op, rhs)) = &relation.rhs else {
                panic!("expected a relation for {src:?}");
            };
            assert_eq!(*op, expected_op, "operator of {src:?}");
            assert_eq!(fragment(src, &rhs.span), "5", "right side of {src:?}");
        }

        let src = "-@a <= 2 * @b + 1";
        let relation = parse(src).unwrap();
        assert_eq!(shape(&relation.lhs), "(neg @a)");
        let Some((_, rhs)) = &relation.rhs else {
            panic!("expected a relation for {src:?}");
        };
        assert_eq!(shape(rhs), "(sum (prod 2 *@b) +1)");
        assert_eq!(fragment(src, &rhs.span), "2 * @b + 1");
    }

    #[test]
    fn a_second_relational_operator_is_rejected() {
        let message = "a generic constraint expression carries at most one relational operator \
                       (<=, >=, ==); an inline double-relational range is not supported — split \
                       into two constraints";
        for src in [
            "line_exchange(3) <= 10 >= 2",
            "@a == @b == @c",
            "1 <= @a <= 2",
        ] {
            let (fragment, err) = rejection(src);
            assert_eq!(err, message, "message for {src:?}");
            let second = src.rfind(['<', '>', '=']).unwrap() - 1;
            assert_eq!(fragment, src.get(second..second + 2).unwrap(), "{src:?}");
        }
    }

    #[test]
    fn a_relational_operator_inside_parentheses_is_rejected() {
        let message = "a relational operator is only allowed at the top level";
        for (src, operator) in [
            ("(@a <= 1)", "<="),
            ("@a <= (@b >= 1)", ">="),
            ("2 * (== 1)", "=="),
            ("f(1 <= 2)", "<="),
        ] {
            assert_eq!(
                rejection(src),
                (operator.to_owned(), message.to_owned()),
                "{src:?}"
            );
        }
        assert_eq!(
            rejection("<= 1"),
            ("<=".to_owned(), "unexpected '<='".to_owned())
        );
    }

    #[test]
    fn nesting_at_the_cap_parses_and_beyond_it_is_rejected() {
        let nested = |open: &str, levels: usize, close: &str| {
            format!("{}@x{}", open.repeat(levels), close.repeat(levels))
        };
        for (open, close, levels_per_unit) in
            [("(", ")", 1), ("-", "", 1), ("+", "", 1), ("(-", ")", 2)]
        {
            let units = MAX_NESTING_DEPTH / levels_per_unit;
            let src = nested(open, units, close);
            let expr = plain(&src);
            assert_eq!(
                fragment(&src, &expr.span),
                src,
                "span at the cap for {open:?}"
            );

            let src = format!("{open}{}", nested(open, units, close)) + close;
            let err = parse(&src).unwrap_err();
            assert_eq!(
                err.message,
                depth_message(),
                "message beyond the cap for {open:?}"
            );
            assert_eq!(
                err.span,
                MAX_NESTING_DEPTH..MAX_NESTING_DEPTH + 1,
                "{open:?}"
            );
        }
    }

    #[test]
    fn deep_nesting_is_rejected_without_overflow() {
        for src in [
            "(".repeat(100_000),
            format!("{}x", "-".repeat(100_000)),
            format!("{}x", "+".repeat(100_000)),
        ] {
            let err = parse(&src).unwrap_err();
            assert_eq!(err.message, depth_message());
            assert_eq!(err.span, MAX_NESTING_DEPTH..MAX_NESTING_DEPTH + 1);
        }
    }

    #[test]
    fn malformed_inputs_report_their_fragment() {
        for (src, expected_fragment, message) in [
            ("", "", "expression must not be empty"),
            ("  ", "  ", "expression must not be empty"),
            ("(x", "x", "expected '(' after variable name \"x\""),
            ("x +", "x", "expected '(' after variable name \"x\""),
            ("(@x", "(@x", "missing closing parenthesis"),
            ("hydro_generation(10", "(10", "missing closing parenthesis"),
            (
                "hydro_generation(0) + (hydro_generation(1) + hydro_generation(2)",
                "(hydro_generation(1) + hydro_generation(2)",
                "missing closing parenthesis",
            ),
            (
                "2 * (hydro_generation(0) hydro_generation(1))",
                "hydro_generation",
                "unexpected name \"hydro_generation\"",
            ),
            ("@x +", "", "expected an operand"),
            (
                "2 x",
                "x",
                "unexpected name \"x\" after a complete expression",
            ),
            ("@x )", ")", "unexpected ')' after a complete expression"),
            ("f()", "f()", "variable \"f\" has an empty argument list"),
            ("f(1,)", ")", "unexpected ')'"),
            ("f(bus 3)", "3", "unexpected number 3"),
            ("f(bus=@a)", "@a", "unexpected parameter \"@a\""),
            ("* 2", "*", "unexpected '*'"),
            (
                "@x < 3",
                "<",
                "bare '<' at position 3 is not a valid operator; use '<=' for a relational bound",
            ),
        ] {
            assert_eq!(
                rejection(src),
                (expected_fragment.to_owned(), message.to_owned()),
                "{src:?}"
            );
        }
    }
}
