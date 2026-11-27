//! Builds the syntax tree of an expression or condition source; names, types and folding are
//! left to the caller.

use std::iter::Peekable;
use std::ops::Range;
use std::vec;

use serde::{Deserialize, Serialize};

use super::Diagnostic;
use super::lexer::{Token, TokenKind, tokenize};

/// The deepest nesting of parentheses, unary operators, `if()` calls and chained `&`/`|`
/// connectives [`parse`] and [`parse_condition`] accept; the bound keeps the recursive descent
/// and the drop of the tree within the stack.
pub(crate) const MAX_NESTING_DEPTH: usize = 256;

const CHAINED_COMPARISON: &str =
    "a comparison relates exactly two expressions; join comparisons with '&' or '|'";

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

/// Comparisons joined by `&` and `|`; a chain of one connective folds to the left.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Condition {
    Compare { lhs: Expr, op: CompareOp, rhs: Expr },
    And(Box<Condition>, Box<Condition>),
    Or(Box<Condition>, Box<Condition>),
}

/// The relation a comparison tests between its two sides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompareOp {
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `==`
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
    /// `if(cond, then_br, else_br)`.
    Conditional {
        cond: Box<Condition>,
        then_br: Box<Expr>,
        else_br: Box<Expr>,
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
    /// Empty for a call without arguments, such as `f()`; every catalog term rejects it.
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
    Parser::new(src)?.relation()
}

/// Parses `src` into a [`Condition`]; `&` binds tighter than `|`, and a comparison relates two
/// expressions without parentheses around it.
pub(crate) fn parse_condition(src: &str) -> Result<Condition, Diagnostic> {
    Parser::new(src)?.condition()
}

struct Parser<'a> {
    src: &'a str,
    tokens: Peekable<vec::IntoIter<Token>>,
    last_end: usize,
    /// Open parentheses plus unary operators and chained connectives on the path to the
    /// current node.
    depth: usize,
    /// Open parentheses only; a relational operator is rejected while any is open.
    groups: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Result<Self, Diagnostic> {
        let tokens = tokenize(src)?;
        if tokens.is_empty() {
            return Err(Diagnostic {
                span: 0..src.len(),
                message: "expression must not be empty".to_owned(),
            });
        }
        Ok(Self {
            src,
            tokens: tokens.into_iter().peekable(),
            last_end: 0,
            depth: 0,
            groups: 0,
        })
    }

    fn condition(mut self) -> Result<Condition, Diagnostic> {
        let condition = self.or()?;
        let Some(token) = self.bump() else {
            return Ok(condition);
        };
        let message = if compare_op(&token.kind).is_some() {
            CHAINED_COMPARISON.to_owned()
        } else {
            format!(
                "unexpected {} after a complete condition",
                self.describe(&token)
            )
        };
        Err(Diagnostic {
            span: token.span,
            message,
        })
    }

    fn or(&mut self) -> Result<Condition, Diagnostic> {
        let mut joins = 0;
        let mut left = self.and()?;
        while let Some(connective) = self.bump_if(|kind| *kind == TokenKind::Pipe) {
            self.enter(&connective.span)?;
            joins += 1;
            let right = self.and()?;
            left = Condition::Or(Box::new(left), Box::new(right));
        }
        self.depth -= joins;
        Ok(left)
    }

    fn and(&mut self) -> Result<Condition, Diagnostic> {
        let mut joins = 0;
        let mut left = self.compare()?;
        while let Some(connective) = self.bump_if(|kind| *kind == TokenKind::Amp) {
            self.enter(&connective.span)?;
            joins += 1;
            let right = self.compare()?;
            left = Condition::And(Box::new(left), Box::new(right));
        }
        self.depth -= joins;
        Ok(left)
    }

    fn compare(&mut self) -> Result<Condition, Diagnostic> {
        let lhs = self.expr()?;
        let Some(op) = self.bump_op(compare_op) else {
            return Err(self.expected_comparison());
        };
        let rhs = self.expr()?;
        Ok(Condition::Compare { lhs, op, rhs })
    }

    fn expected_comparison(&mut self) -> Diagnostic {
        const EXPECTED: &str = "expected a comparison operator (<, <=, >, >=, ==)";
        match self.bump() {
            Some(token) => Diagnostic {
                message: format!("unexpected {}; {EXPECTED}", self.describe(&token)),
                span: token.span,
            },
            None => self.at_end(EXPECTED),
        }
    }

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
        } else if rhs.is_none() && matches!(token.kind, TokenKind::Lt | TokenKind::Gt) {
            let text = self.src.get(token.span.clone()).unwrap_or_default();
            format!(
                "bare '{text}' is not a valid operator in a constraint expression; use \
                 '{text}=' for a relational bound"
            )
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
        if name == "if" {
            return self.conditional(name_span, open.span);
        }
        let mut args = Vec::new();
        let close = loop {
            let token = self.next_in_group(&open.span)?;
            let arg = match token.kind {
                TokenKind::RParen if args.is_empty() => break token.span,
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

    fn conditional(
        &mut self,
        name_span: Range<usize>,
        open: Range<usize>,
    ) -> Result<Expr, Diagnostic> {
        if let Some(close) = self.bump_if(|kind| *kind == TokenKind::RParen) {
            return Err(arity(&name_span, close.span.end));
        }
        self.enter(&open)?;
        self.groups += 1;
        let cond = self.or()?;
        if let Some(token) = self.bump_if(|kind| compare_op(kind).is_some()) {
            return Err(Diagnostic {
                span: token.span,
                message: CHAINED_COMPARISON.to_owned(),
            });
        }
        self.expect_argument_separator(&name_span, &open)?;
        let then_br = self.expr()?;
        self.expect_argument_separator(&name_span, &open)?;
        let else_br = self.expr()?;
        let token = self.next_in_group(&open)?;
        let close = match token.kind {
            TokenKind::RParen => token.span,
            TokenKind::Comma => return Err(arity(&name_span, self.skip_to_close(&open)?.end)),
            _ => return Err(self.unexpected(&token, true)),
        };
        self.groups -= 1;
        self.depth -= 1;
        Ok(Expr {
            kind: ExprKind::Conditional {
                cond: Box::new(cond),
                then_br: Box::new(then_br),
                else_br: Box::new(else_br),
            },
            span: name_span.start..close.end,
        })
    }

    /// Consumes the `,` after an argument of the `if` call named at `name_span`; a `)` there
    /// means the call ends one argument short.
    fn expect_argument_separator(
        &mut self,
        name_span: &Range<usize>,
        open: &Range<usize>,
    ) -> Result<(), Diagnostic> {
        let token = self.next_in_group(open)?;
        match token.kind {
            TokenKind::Comma => Ok(()),
            TokenKind::RParen => Err(arity(name_span, token.span.end)),
            _ => Err(self.unexpected(&token, true)),
        }
    }

    /// Consumes tokens through the `)` that closes the parentheses opened at `open`.
    fn skip_to_close(&mut self, open: &Range<usize>) -> Result<Range<usize>, Diagnostic> {
        let mut nested = 0;
        loop {
            let token = self.next_in_group(open)?;
            match token.kind {
                TokenKind::LParen => nested += 1,
                TokenKind::RParen if nested == 0 => return Ok(token.span),
                TokenKind::RParen => nested -= 1,
                _ => {}
            }
        }
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

fn arity(name_span: &Range<usize>, end: usize) -> Diagnostic {
    Diagnostic {
        span: name_span.start..end,
        message: "if() takes exactly three arguments: a condition and two expressions".to_owned(),
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

fn compare_op(kind: &TokenKind) -> Option<CompareOp> {
    match kind {
        TokenKind::Lt => Some(CompareOp::Lt),
        TokenKind::Le => Some(CompareOp::Le),
        TokenKind::Gt => Some(CompareOp::Gt),
        TokenKind::Ge => Some(CompareOp::Ge),
        TokenKind::EqEq => Some(CompareOp::Eq),
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

    use super::{
        AddOp, Arg, CompareOp, Condition, Expr, ExprKind, MAX_NESTING_DEPTH, MulOp, RelOp,
        VariableCall, parse, parse_condition,
    };

    /// Renders a tree without spans: `(sum h +x -y)`, `(prod h *x /y)`, `(neg x)`, `@name`,
    /// `name(1,key=2)`, `name` for `name()`, and `(if cond then else)`.
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
                if args.is_empty() {
                    call.name.clone()
                } else {
                    format!("{}({})", call.name, args.join(","))
                }
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
            ExprKind::Conditional {
                cond,
                then_br,
                else_br,
            } => format!(
                "(if {} {} {})",
                condition_shape(cond),
                shape(then_br),
                shape(else_br)
            ),
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

    /// Renders a condition without spans: `(and c c)`, `(or c c)`, `(lhs <= rhs)`.
    fn condition_shape(condition: &Condition) -> String {
        match condition {
            Condition::Compare { lhs, op, rhs } => {
                let op = match op {
                    CompareOp::Lt => "<",
                    CompareOp::Le => "<=",
                    CompareOp::Gt => ">",
                    CompareOp::Ge => ">=",
                    CompareOp::Eq => "==",
                };
                format!("({} {op} {})", shape(lhs), shape(rhs))
            }
            Condition::And(left, right) => {
                format!("(and {} {})", condition_shape(left), condition_shape(right))
            }
            Condition::Or(left, right) => {
                format!("(or {} {})", condition_shape(left), condition_shape(right))
            }
        }
    }

    fn rejected_condition(src: &str) -> (String, String) {
        let err = parse_condition(src).unwrap_err();
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
    fn a_call_may_have_no_arguments() {
        let no_arguments = |name: &str| {
            ExprKind::Variable(VariableCall {
                name: name.to_owned(),
                args: Vec::new(),
            })
        };

        let src = "f() <= 1";
        let lhs = parse(src).unwrap().lhs;
        assert_eq!(lhs.kind, no_arguments("f"));
        assert_eq!(fragment(src, &lhs.span), "f()");

        let Condition::Compare { lhs, .. } = parse_condition(src).unwrap() else {
            panic!("expected a comparison for {src:?}");
        };
        assert_eq!(lhs.kind, no_arguments("f"));
        assert_eq!(fragment(src, &lhs.span), "f()");
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
            ("f <= 1", "f", "expected '(' after variable name \"f\""),
            ("f(,)", ",", "unexpected ','"),
            ("f(1,)", ")", "unexpected ')'"),
            ("f(bus 3)", "3", "unexpected number 3"),
            ("f(bus=@a)", "@a", "unexpected parameter \"@a\""),
            ("* 2", "*", "unexpected '*'"),
            (
                "@x < 3",
                "<",
                "bare '<' is not a valid operator in a constraint expression; use '<=' for a \
                 relational bound",
            ),
            (
                "@x > 3",
                ">",
                "bare '>' is not a valid operator in a constraint expression; use '>=' for a \
                 relational bound",
            ),
            (
                "@x <= 1 > 2",
                ">",
                "unexpected '>' after a complete expression",
            ),
            ("< 1", "<", "unexpected '<'"),
        ] {
            assert_eq!(
                rejection(src),
                (expected_fragment.to_owned(), message.to_owned()),
                "{src:?}"
            );
        }
    }

    #[test]
    fn a_conjunction_of_comparisons_parses() {
        let src = "@a <= 1 & @b <= 2";
        let condition = parse_condition(src).unwrap();
        assert_eq!(condition_shape(&condition), "(and (@a <= 1) (@b <= 2))");
        let Condition::And(left, right) = &condition else {
            panic!("expected a conjunction for {src:?}");
        };
        let (
            Condition::Compare {
                lhs: a, rhs: one, ..
            },
            Condition::Compare {
                lhs: b, rhs: two, ..
            },
        ) = (&**left, &**right)
        else {
            panic!("expected two comparisons for {src:?}");
        };
        let fragments = [a, one, b, two].map(|expr| fragment(src, &expr.span));
        assert_eq!(fragments, ["@a", "1", "@b", "2"]);
    }

    #[test]
    fn and_binds_tighter_than_or_and_chains_fold_left() {
        let condition = parse_condition("@a<=1 & @b<=2 | @c<=3").unwrap();
        let Condition::Or(left, right) = &condition else {
            panic!("expected a disjunction, got {condition:?}");
        };
        assert!(matches!(**left, Condition::And(..)));
        assert!(matches!(
            **right,
            Condition::Compare {
                op: CompareOp::Le,
                ..
            }
        ));

        for (src, expected) in [
            (
                "@a<=1 & @b<=2 | @c<=3",
                "(or (and (@a <= 1) (@b <= 2)) (@c <= 3))",
            ),
            (
                "@a<=1 | @b<=2 & @c<=3",
                "(or (@a <= 1) (and (@b <= 2) (@c <= 3)))",
            ),
            (
                "@a<=1 & @b<=2 & @c<=3",
                "(and (and (@a <= 1) (@b <= 2)) (@c <= 3))",
            ),
            (
                "@a<=1 | @b<=2 | @c<=3",
                "(or (or (@a <= 1) (@b <= 2)) (@c <= 3))",
            ),
            (
                "@a<=1 & @b<=2 | @c<=3 & @d<=4",
                "(or (and (@a <= 1) (@b <= 2)) (and (@c <= 3) (@d <= 4)))",
            ),
        ] {
            let condition = parse_condition(src).unwrap();
            assert_eq!(condition_shape(&condition), expected, "shape of {src:?}");
        }
    }

    #[test]
    fn every_comparison_operator_parses_with_or_without_whitespace() {
        for (src, expected) in [
            ("@a < 1", CompareOp::Lt),
            ("@a <= 1", CompareOp::Le),
            ("@a > 1", CompareOp::Gt),
            ("@a >= 1", CompareOp::Ge),
            ("@a == 1", CompareOp::Eq),
            ("@a<1", CompareOp::Lt),
            ("@a<=1", CompareOp::Le),
            ("@a>1", CompareOp::Gt),
            ("@a>=1", CompareOp::Ge),
            ("@a==1", CompareOp::Eq),
        ] {
            let Condition::Compare { op, .. } = parse_condition(src).unwrap() else {
                panic!("expected a comparison for {src:?}");
            };
            assert_eq!(op, expected, "operator of {src:?}");
        }
    }

    #[test]
    fn comparison_operands_are_full_expressions() {
        let src = "bus_load_value(3)+bus_load_value(4) <= 17000";
        let Condition::Compare { lhs, op, rhs } = parse_condition(src).unwrap() else {
            panic!("expected a comparison for {src:?}");
        };
        assert_eq!(op, CompareOp::Le);
        assert_eq!(
            fragment(src, &lhs.span),
            "bus_load_value(3)+bus_load_value(4)"
        );
        assert_eq!(fragment(src, &rhs.span), "17000");
        let ExprKind::Sum { head, tail } = &lhs.kind else {
            panic!("expected a sum on the left of {src:?}");
        };
        let calls: Vec<&str> = std::iter::once(&**head)
            .chain(tail.iter().map(|(_, operand)| operand))
            .map(|operand| match &operand.kind {
                ExprKind::Variable(call) if call.name == "bus_load_value" => {
                    fragment(src, &operand.span)
                }
                other => panic!("expected a bus_load_value call, got {other:?}"),
            })
            .collect();
        assert_eq!(calls, ["bus_load_value(3)", "bus_load_value(4)"]);

        for (src, expected) in [
            (
                "2 * (@a + 1) > @b - 3",
                "((prod 2 *(sum @a +1)) > (sum @b -3))",
            ),
            (
                "-@a >= f(1, bus=2) / 4",
                "((neg @a) >= (prod f(1,bus=2) /4))",
            ),
            ("1 == 2", "(1 == 2)"),
        ] {
            let condition = parse_condition(src).unwrap();
            assert_eq!(condition_shape(&condition), expected, "shape of {src:?}");
        }
    }

    #[test]
    fn malformed_conditions_report_their_fragment() {
        let expected_comparison = "expected a comparison operator (<, <=, >, >=, ==)";
        for (src, expected_fragment, message) in [
            ("", "", "expression must not be empty"),
            (
                "@a & ",
                "&",
                "unexpected '&'; expected a comparison operator (<, <=, >, >=, ==)",
            ),
            (
                "@a = 1",
                "=",
                "unexpected '='; expected a comparison operator (<, <=, >, >=, ==)",
            ),
            ("@a", "", expected_comparison),
            ("@a <= 1 & @b", "", expected_comparison),
            ("@a <=", "", "expected an operand"),
            ("@a <= 1 &", "", "expected an operand"),
            ("& @a <= 1", "&", "unexpected '&'"),
            ("@a <= 1 & & @b <= 2", "&", "unexpected '&'"),
            (
                "@a <= 1 b <= 2",
                "b",
                "unexpected name \"b\" after a complete condition",
            ),
            (
                "@a <= 1 <= 2",
                "<=",
                "a comparison relates exactly two expressions; join comparisons with '&' or '|'",
            ),
            (
                "@a <= 1 )",
                ")",
                "unexpected ')' after a complete condition",
            ),
            ("(@a + 1", "(@a + 1", "missing closing parenthesis"),
            ("@a <= f(1", "(1", "missing closing parenthesis"),
            (
                "(@a <= 1) & @b <= 2",
                "<=",
                "a relational operator is only allowed at the top level",
            ),
        ] {
            assert_eq!(
                rejected_condition(src),
                (expected_fragment.to_owned(), message.to_owned()),
                "{src:?}"
            );
        }
    }

    #[test]
    fn a_name_without_parentheses_is_rejected_in_a_condition() {
        let message = "expected '(' after variable name \"load\"".to_owned();
        assert_eq!(
            rejected_condition("82000 < load"),
            ("load".to_owned(), message.clone())
        );
        assert_eq!(
            rejection("if(load <= 1, 1, 2)"),
            ("load".to_owned(), message)
        );
    }

    #[test]
    fn a_dangling_connective_points_at_the_end() {
        for src in ["@a <= 1 &", "@a <= 1 | ", "@a <= 1 & @b <= 2 |", "@a <="] {
            let err = parse_condition(src).unwrap_err();
            assert_eq!(err.span, src.len()..src.len(), "{src:?}");
        }
    }

    #[test]
    fn boolean_connectives_stay_out_of_linear_expressions() {
        for (src, expected_fragment, message) in [
            ("a & b", "a", "expected '(' after variable name \"a\""),
            ("@a & @b", "&", "unexpected '&' after a complete expression"),
            (
                "@a <= 1 | @b <= 2",
                "|",
                "unexpected '|' after a complete expression",
            ),
            (
                "f(1) <= 2 & f(2) <= 3",
                "&",
                "unexpected '&' after a complete expression",
            ),
            ("(@a & @b)", "&", "unexpected '&'"),
        ] {
            assert_eq!(
                rejection(src),
                (expected_fragment.to_owned(), message.to_owned()),
                "{src:?}"
            );
        }
    }

    #[test]
    fn connective_chains_at_the_cap_parse_and_beyond_it_are_rejected() {
        for connective in ['&', '|'] {
            let chain = |terms: usize| vec!["@a<=1"; terms].join(&format!(" {connective} "));
            parse_condition(&chain(MAX_NESTING_DEPTH + 1)).unwrap();

            let src = chain(MAX_NESTING_DEPTH + 2);
            let err = parse_condition(&src).unwrap_err();
            assert_eq!(err.message, depth_message(), "{connective}");
            let (offending, _) = src
                .match_indices(connective)
                .nth(MAX_NESTING_DEPTH)
                .unwrap();
            assert_eq!(err.span, offending..offending + 1, "{connective}");
        }
    }

    #[test]
    fn deep_conditions_are_rejected_without_overflow() {
        for connective in ['&', '|'] {
            let src = format!("@a<=1 {connective} ").repeat(100_000) + "@a<=1";
            let err = parse_condition(&src).unwrap_err();
            assert_eq!(err.message, depth_message(), "{connective}");
            let (offending, _) = src
                .match_indices(connective)
                .nth(MAX_NESTING_DEPTH)
                .unwrap();
            assert_eq!(err.span, offending..offending + 1, "{connective}");
        }

        let src = format!("{}@a <= 1", "(".repeat(100_000));
        let err = parse_condition(&src).unwrap_err();
        assert_eq!(err.message, depth_message());
        assert_eq!(err.span, MAX_NESTING_DEPTH..MAX_NESTING_DEPTH + 1);
    }

    const ARITY: &str = "if() takes exactly three arguments: a condition and two expressions";

    /// Unoptimized frames of `if()` nested in a condition outgrow the 2 MiB stack of a test
    /// thread before the nesting cap.
    fn on_large_stack(test: impl FnOnce() + Send + 'static) {
        let thread = std::thread::Builder::new()
            .stack_size(16 << 20)
            .spawn(test)
            .unwrap();
        if let Err(panic) = thread.join() {
            std::panic::resume_unwind(panic);
        }
    }

    #[test]
    fn a_conditional_parses_into_its_condition_and_branches() {
        let src = "if(@a<=1, 10, 20)";
        let expr = plain(src);
        assert_eq!(shape(&expr), "(if (@a <= 1) 10 20)");
        assert_eq!(fragment(src, &expr.span), src);
        let ExprKind::Conditional {
            cond,
            then_br,
            else_br,
        } = &expr.kind
        else {
            panic!("expected a conditional for {src:?}");
        };
        let Condition::Compare { lhs, op, rhs } = &**cond else {
            panic!("expected a comparison for {src:?}");
        };
        assert_eq!(*op, CompareOp::Le);
        let fragments = [lhs, rhs, then_br, else_br].map(|expr| fragment(src, &expr.span));
        assert_eq!(fragments, ["@a", "1", "10", "20"]);
    }

    #[test]
    fn conditional_branches_are_full_expressions() {
        for (src, expected) in [
            (
                "if(@a<=1, 2 * @x + 1, -@y)",
                "(if (@a <= 1) (sum (prod 2 *@x) +1) (neg @y))",
            ),
            (
                "if(@a<=1, f(1, bus=2) / 4, (@x))",
                "(if (@a <= 1) (prod f(1,bus=2) /4) @x)",
            ),
            (
                "1 + if(@a<=1, 2, 3) * 2",
                "(sum 1 +(prod (if (@a <= 1) 2 3) *2))",
            ),
            ("-if(@a<=1, 2, 3)", "(neg (if (@a <= 1) 2 3))"),
            ("(if(@a<=1, 2, 3))", "(if (@a <= 1) 2 3)"),
        ] {
            assert_eq!(shape(&plain(src)), expected, "shape of {src:?}");
        }

        let src = "hydro_generation(1) + if(@a<=1, 2, 3) <= if(@a>1, 4, 5)";
        let relation = parse(src).unwrap();
        assert_eq!(
            shape(&relation.lhs),
            "(sum hydro_generation(1) +(if (@a <= 1) 2 3))"
        );
        let Some((RelOp::Le, rhs)) = &relation.rhs else {
            panic!("expected a relation for {src:?}");
        };
        assert_eq!(shape(rhs), "(if (@a > 1) 4 5)");
        assert_eq!(fragment(src, &rhs.span), "if(@a>1, 4, 5)");
    }

    #[test]
    fn conditionals_nest_in_every_position() {
        for (src, expected) in [
            (
                "if(@a<=1, if(@b<=2,1,2), 3)",
                "(if (@a <= 1) (if (@b <= 2) 1 2) 3)",
            ),
            (
                "if(@a<=1, 3, if(@b<=2,1,2))",
                "(if (@a <= 1) 3 (if (@b <= 2) 1 2))",
            ),
            (
                "if(if(@a<=1,1,2) <= 3, 4, 5)",
                "(if ((if (@a <= 1) 1 2) <= 3) 4 5)",
            ),
        ] {
            assert_eq!(shape(&plain(src)), expected, "shape of {src:?}");
        }
    }

    #[test]
    fn a_conditional_in_a_condition_operand_parses() {
        let src = "if(@a<=1, 2, 3) > @x";
        let Condition::Compare { lhs, .. } = parse_condition(src).unwrap() else {
            panic!("expected a comparison for {src:?}");
        };
        assert_eq!(shape(&lhs), "(if (@a <= 1) 2 3)");
        assert_eq!(fragment(src, &lhs.span), "if(@a<=1, 2, 3)");
    }

    #[test]
    fn malformed_parameter_expressions_report_their_fragment() {
        for (src, expected_fragment, message) in [
            ("@a +", "", "expected an operand"),
            (
                "@a @b",
                "@b",
                "unexpected parameter \"@b\" after a complete expression",
            ),
            (
                "@a <= 1 <= 2",
                "<=",
                "a generic constraint expression carries at most one relational operator (<=, >=, ==); an inline double-relational range is not supported — split into two constraints",
            ),
            ("@a & 1", "&", "unexpected '&' after a complete expression"),
        ] {
            let diagnostic = parse(src).unwrap_err();
            let fragment = src.get(diagnostic.span.clone()).unwrap_or_default();
            assert_eq!(
                (fragment, diagnostic.message.as_str()),
                (expected_fragment, message),
                "{src:?}"
            );
        }
    }

    #[test]
    fn a_conditional_with_the_wrong_argument_count_is_rejected_over_the_call() {
        for (src, call) in [
            ("if()", "if()"),
            ("if(@a<=1)", "if(@a<=1)"),
            ("if(@a<=1, 10)", "if(@a<=1, 10)"),
            ("if(@a<=1, 10, 20, 30)", "if(@a<=1, 10, 20, 30)"),
            (
                "if(@a<=1, 10, 20, f(3), g(4, 5))",
                "if(@a<=1, 10, 20, f(3), g(4, 5))",
            ),
            ("1 + if(@a<=1, 10) * 2", "if(@a<=1, 10)"),
            ("if(@a<=1, 10, 20, 30) + 1", "if(@a<=1, 10, 20, 30)"),
        ] {
            let (fragment, message) = rejection(src);
            assert_eq!(fragment, call, "{src:?}");
            assert!(message.ends_with(ARITY), "{src:?}: {message}");
        }
    }

    #[test]
    fn a_malformed_conditional_reports_its_fragment() {
        for (src, expected_fragment, message) in [
            (
                "if(@a, 1, 2)",
                ",",
                "unexpected ','; expected a comparison operator (<, <=, >, >=, ==)",
            ),
            (
                "if(@a<=1<=2, 1, 2)",
                "<=",
                "a comparison relates exactly two expressions; join comparisons with '&' or '|'",
            ),
            ("if(@a<=1 1, 2)", "1", "unexpected number 1"),
            (
                "if(@a<=1, @b <= 2, 3)",
                "<=",
                "a relational operator is only allowed at the top level",
            ),
            (
                "if(@a<=1, 1, @b == 2)",
                "==",
                "a relational operator is only allowed at the top level",
            ),
            (
                "if(@a<=1, 1, 2",
                "(@a<=1, 1, 2",
                "missing closing parenthesis",
            ),
            ("if(@a<=1", "(@a<=1", "missing closing parenthesis"),
            (
                "if(@a<=1, 1, 2, 3",
                "(@a<=1, 1, 2, 3",
                "missing closing parenthesis",
            ),
            ("if(@a<=1, 1,", "", "expected an operand"),
            ("if(@a<=1, , 2)", ",", "unexpected ','"),
            ("if(@a<=1 & , 1, 2)", ",", "unexpected ','"),
            (
                "if(<=1, 1, 2)",
                "<=",
                "a relational operator is only allowed at the top level",
            ),
        ] {
            assert_eq!(
                rejection(src),
                (expected_fragment.to_owned(), message.to_owned()),
                "{src:?}"
            );
        }
    }

    #[test]
    fn conditional_nesting_at_the_cap_parses_and_beyond_it_is_rejected() {
        on_large_stack(conditional_nesting_cap);
    }

    fn conditional_nesting_cap() {
        let nested = |open: &str, close: &str, levels: usize| {
            format!("{}1{}", open.repeat(levels), close.repeat(levels))
        };
        for (open, close, offset) in [("if(@a<=1,", ",2)", 2), ("if(", "<=1,2,3)", 2)] {
            let src = nested(open, close, MAX_NESTING_DEPTH);
            let expr = plain(&src);
            assert_eq!(fragment(&src, &expr.span), src, "span at the cap");

            let src = nested(open, close, MAX_NESTING_DEPTH + 1);
            let err = parse(&src).unwrap_err();
            assert_eq!(err.message, depth_message(), "message beyond the cap");
            let at = MAX_NESTING_DEPTH * open.len() + offset;
            assert_eq!(err.span, at..at + 1, "span beyond the cap");
        }

        let chain = |terms: usize| format!("if({}, 1, 2)", vec!["@a<=1"; terms].join(" & "));
        plain(&chain(MAX_NESTING_DEPTH));
        let src = chain(MAX_NESTING_DEPTH + 1);
        let err = parse(&src).unwrap_err();
        assert_eq!(err.message, depth_message());
        let (offending, _) = src.match_indices('&').nth(MAX_NESTING_DEPTH - 1).unwrap();
        assert_eq!(err.span, offending..offending + 1);
    }

    #[test]
    fn deep_conditionals_are_rejected_without_overflow() {
        on_large_stack(deep_conditionals);
    }

    fn deep_conditionals() {
        for src in [
            "if(@a<=1,".repeat(100_000),
            "if(".repeat(100_000),
            "if(@a<=1, 1,".repeat(100_000),
        ] {
            let err = parse(&src).unwrap_err();
            assert_eq!(err.message, depth_message());
        }
    }
}
