//! The generic-constraint expression language. [`catalog`] maps a variable name and its
//! arguments to a [`VariableRef`](novomodelo_core::VariableRef); [`lexer`] splits a source
//! string into spanned tokens; [`parser`] builds their syntax tree; [`lower`] turns a tree into
//! a [`ScalarExpr`](novomodelo_core::ScalarExpr) program.

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
