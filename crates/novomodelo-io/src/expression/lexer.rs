//! Splits an expression source into tokens, each carrying the byte range it was read from.

use std::iter::Peekable;
use std::ops::Range;
use std::str::CharIndices;

use super::Diagnostic;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Token {
    pub(crate) kind: TokenKind,
    /// Byte range of the token in the source.
    pub(crate) span: Range<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TokenKind {
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Comma,
    /// The `=` of a named argument such as `bus=`.
    Equals,
    Lt,
    Gt,
    Le,
    Ge,
    EqEq,
    Amp,
    Pipe,
    /// A finite, non-negative literal; the lexer never emits a sign.
    Number(f64),
    Ident(String),
    /// A `@name` parameter reference; holds the name without the `@`.
    ParamRef(String),
}

type Chars<'a> = Peekable<CharIndices<'a>>;

/// Tokenizes `src`, rejecting the first character or literal that is not part of the
/// expression language.
pub(crate) fn tokenize(src: &str) -> Result<Vec<Token>, Diagnostic> {
    let mut chars = src.char_indices().peekable();
    let mut tokens = Vec::new();
    while let Some((start, c)) = chars.next() {
        if c.is_ascii_whitespace() {
            continue;
        }
        let kind = match c {
            '+' => TokenKind::Plus,
            '-' => TokenKind::Minus,
            '*' => TokenKind::Star,
            '/' => TokenKind::Slash,
            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            ',' => TokenKind::Comma,
            '=' => eq_variant(&mut chars, TokenKind::EqEq, TokenKind::Equals),
            '<' => eq_variant(&mut chars, TokenKind::Le, TokenKind::Lt),
            '>' => eq_variant(&mut chars, TokenKind::Ge, TokenKind::Gt),
            '&' => TokenKind::Amp,
            '|' => TokenKind::Pipe,
            '@' => {
                let Some((_, first)) = chars.next_if(|&(_, next)| is_ident_start(next)) else {
                    return Err(Diagnostic {
                        span: start..start + c.len_utf8(),
                        message: format!(
                            "@ must be followed by an identifier at position {}",
                            char_position(src, start)
                        ),
                    });
                };
                TokenKind::ParamRef(take_while(&mut chars, first, |_, next| {
                    is_ident_continue(next)
                }))
            }
            c if c.is_ascii_digit() || c == '.' => {
                let text = take_while(&mut chars, c, |text, next| {
                    next.is_ascii_digit()
                        || matches!(next, '.' | 'e' | 'E')
                        || (matches!(next, '+' | '-') && text.ends_with(['e', 'E']))
                });
                let span = start..next_offset(&mut chars, src);
                let Ok(value) = text.parse::<f64>() else {
                    return Err(Diagnostic {
                        span,
                        message: format!(
                            "invalid number literal \"{text}\" at position {}",
                            char_position(src, start)
                        ),
                    });
                };
                if !value.is_finite() {
                    return Err(Diagnostic {
                        span,
                        message: format!("numeric literal {text} is not finite"),
                    });
                }
                TokenKind::Number(value)
            }
            c if is_ident_start(c) => {
                TokenKind::Ident(take_while(&mut chars, c, |_, next| is_ident_continue(next)))
            }
            other => {
                return Err(Diagnostic {
                    span: start..start + other.len_utf8(),
                    message: format!(
                        "unexpected character '{other}' at position {}",
                        char_position(src, start)
                    ),
                });
            }
        };
        tokens.push(Token {
            kind,
            span: start..next_offset(&mut chars, src),
        });
    }
    Ok(tokens)
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn eq_variant(chars: &mut Chars<'_>, with_eq: TokenKind, without_eq: TokenKind) -> TokenKind {
    if chars.next_if(|&(_, next)| next == '=').is_some() {
        with_eq
    } else {
        without_eq
    }
}

/// Collects `first` and every following character `continues` accepts given the text so far.
fn take_while(
    chars: &mut Chars<'_>,
    first: char,
    continues: impl Fn(&str, char) -> bool,
) -> String {
    let mut text = String::from(first);
    while let Some((_, next)) = chars.next_if(|&(_, next)| continues(&text, next)) {
        text.push(next);
    }
    text
}

fn next_offset(chars: &mut Chars<'_>, src: &str) -> usize {
    chars.peek().map_or(src.len(), |&(offset, _)| offset)
}

/// The character index of byte `offset`, the position diagnostics print.
fn char_position(src: &str, offset: usize) -> usize {
    src.get(..offset).map_or(0, |prefix| prefix.chars().count())
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::{Token, TokenKind, tokenize};

    fn kinds(src: &str) -> Vec<TokenKind> {
        tokenize(src)
            .unwrap()
            .into_iter()
            .map(|token| token.kind)
            .collect()
    }

    #[test]
    fn tokenize_emits_byte_spans_for_every_token() {
        let tok = |kind, span| Token { kind, span };
        assert_eq!(
            tokenize("2*x").unwrap(),
            vec![
                tok(TokenKind::Number(2.0), 0..1),
                tok(TokenKind::Star, 1..2),
                tok(TokenKind::Ident("x".to_owned()), 2..3),
            ]
        );

        let src = "@vazão + 1";
        let tokens = tokenize(src).unwrap();
        assert_eq!(
            tokens,
            vec![
                tok(TokenKind::ParamRef("vazão".to_owned()), 0..7),
                tok(TokenKind::Plus, 8..9),
                tok(TokenKind::Number(1.0), 10..11),
            ]
        );
        assert_eq!(
            src.get(tokens.first().unwrap().span.clone()),
            Some("@vazão")
        );

        let src = " geração_hidro(3, bus=7) <= 1.5e2 ";
        let tokens = tokenize(src).unwrap();
        let fragments: Vec<&str> = tokens
            .iter()
            .map(|token| src.get(token.span.clone()).unwrap())
            .collect();
        assert_eq!(
            fragments,
            [
                "geração_hidro",
                "(",
                "3",
                ",",
                "bus",
                "=",
                "7",
                ")",
                "<=",
                "1.5e2"
            ]
        );
    }

    #[test]
    fn tokenize_recognises_division_and_relational_operators() {
        assert_eq!(
            kinds("x / 2 <= y >= z == w"),
            vec![
                TokenKind::Ident("x".to_owned()),
                TokenKind::Slash,
                TokenKind::Number(2.0),
                TokenKind::Le,
                TokenKind::Ident("y".to_owned()),
                TokenKind::Ge,
                TokenKind::Ident("z".to_owned()),
                TokenKind::EqEq,
                TokenKind::Ident("w".to_owned()),
            ]
        );
        assert_eq!(kinds("="), vec![TokenKind::Equals]);
        assert_eq!(
            tokenize("a / b").unwrap().get(1),
            Some(&Token {
                kind: TokenKind::Slash,
                span: 2..3
            })
        );
    }

    #[test]
    fn tokenize_recognises_strict_comparisons_and_boolean_connectives() {
        let ident = |name: &str| TokenKind::Ident(name.to_owned());
        assert_eq!(kinds("a < b"), vec![ident("a"), TokenKind::Lt, ident("b")]);
        assert_eq!(kinds("a > b"), vec![ident("a"), TokenKind::Gt, ident("b")]);
        assert_eq!(
            kinds("a & b | c"),
            vec![
                ident("a"),
                TokenKind::Amp,
                ident("b"),
                TokenKind::Pipe,
                ident("c")
            ]
        );
        assert_eq!(
            kinds("82000 < load & load <= 95000"),
            vec![
                TokenKind::Number(82000.0),
                TokenKind::Lt,
                ident("load"),
                TokenKind::Amp,
                ident("load"),
                TokenKind::Le,
                TokenKind::Number(95000.0),
            ]
        );
    }

    #[test]
    fn tokenize_disambiguates_strict_from_non_strict_without_whitespace() {
        let ident = |name: &str| TokenKind::Ident(name.to_owned());
        for (src, op) in [
            ("x<1", TokenKind::Lt),
            ("x<=1", TokenKind::Le),
            ("x>1", TokenKind::Gt),
            ("x>=1", TokenKind::Ge),
        ] {
            assert_eq!(
                kinds(src),
                vec![ident("x"), op, TokenKind::Number(1.0)],
                "{src:?}"
            );
        }
    }

    #[test]
    fn tokenize_emits_byte_spans_for_strict_comparisons_and_connectives() {
        let src = "ç<=1 < 2 > 3 & 4|5";
        let fragments: Vec<&str> = tokenize(src)
            .unwrap()
            .iter()
            .map(|token| src.get(token.span.clone()).unwrap())
            .collect();
        assert_eq!(
            fragments,
            ["ç", "<=", "1", "<", "2", ">", "3", "&", "4", "|", "5"]
        );
    }

    #[test]
    fn tokenize_number_forms() {
        for (src, expected) in [
            ("2", 2.0_f64),
            (".5", 0.5),
            ("1.5e-3", 1.5e-3),
            ("2E+4", 2.0e4),
        ] {
            let tokens = kinds(src);
            let [TokenKind::Number(value)] = tokens.as_slice() else {
                panic!("expected one number token for {src:?}");
            };
            assert_eq!(value.to_bits(), expected.to_bits(), "value of {src:?}");
        }
        assert_eq!(
            kinds("2-3"),
            vec![
                TokenKind::Number(2.0),
                TokenKind::Minus,
                TokenKind::Number(3.0)
            ]
        );
    }

    #[test]
    fn tokenize_rejects_non_finite_literal() {
        let err = tokenize("1e999").unwrap_err();
        assert_eq!(err.span, 0..5);
        assert_eq!(err.message, "numeric literal 1e999 is not finite");
    }

    #[test]
    fn tokenize_keeps_the_lexical_messages() {
        for (src, fragment, message) in [
            (
                "ação + @",
                "@",
                "@ must be followed by an identifier at position 7",
            ),
            ("é # 1", "#", "unexpected character '#' at position 2"),
            (
                "ã + 1.2.3",
                "1.2.3",
                "invalid number literal \"1.2.3\" at position 4",
            ),
        ] {
            let err = tokenize(src).unwrap_err();
            assert_eq!(err.message, message, "message for {src:?}");
            assert_eq!(
                src.get(err.span.clone()),
                Some(fragment),
                "span for {src:?}"
            );
        }
    }
}
