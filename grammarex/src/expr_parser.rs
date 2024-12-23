use thiserror::Error;

use crate::types::GrammarEx;

#[derive(Error, Debug)]
pub enum GrammarexParseError {
    #[error("Missing closing backet")]
    MissingClosingBracket,
    #[error("Unexpected end of input")]
    UnexpectedEnd,
    #[error("Invalid backslash escape")]
    InvalidEscape,
    #[error("Mismatched parenthesis")]
    MismatchedParenthesis,
    #[error("The charachter`{0}` must follow an expression")]
    DidntFollowExpression(char),
    #[error("Mismatched quotes")]
    MismatchedQuotes,
}

//Takes the first char of the string. This mutates the input.
fn take_first(input: &mut &str) -> Option<char> {
    let first: char = input.chars().next()?;
    *input = &input[first.len_utf8()..];
    return Some(first);
}
//Peek at the next k chars. Doesn't advance the input.
fn peek_k<const N: usize>(input: &&str) -> Option<[char; N]> {
    let mut result: [char; N] = ['_'; N];
    let mut chars = input.chars();
    for i in 0..N {
        let char = chars.next()?;
        result[i] = char;
    }
    return Some(result);
}
//Peek at the next char. Doesn't advance the input.
fn peek(input: &&str) -> Option<char> {
    let item = input.chars().next()?;
    return Some(item);
}
//Skips some number of input chars. This method panics if there isn't enough input
fn skip(input: &mut &str, amount: usize) {
    for _ in 0..amount {
        let c = input.chars().next().expect("Uexpectedly ran out of input");
        *input = &input[c.len_utf8()..];
    }
}
// This parses the inner part of a regex bracket match.
// The '[' token must have already been parsed.
fn parse_backet_inner(input: &mut &str) -> Result<GrammarEx, GrammarexParseError> {
    let mut classes = Vec::new();
    let mut first = true;
    loop {
        let c = take_first(input).ok_or(GrammarexParseError::MissingClosingBracket)?;
        //A regex backet can match the ']' char if it is the first char in the [] expression.
        if c == ']' && !first {
            return Ok(GrammarEx::Alt(classes));
        }
        //[a-c] represents [abc] but if the dash is first or last, it is considered a literal like in [ax-]
        if let Some(cs) = peek_k::<2>(input) {
            if cs[0] == '-' && cs[1] != ']' {
                first = false;
                classes.push(GrammarEx::CharRangeInclusive(c,cs[1]));
                skip(input, 2);
                continue;
            }
        }
        classes.push(GrammarEx::Char(c));
        first = false;
    }
}
// This method trims all ascii whitespace from the input.
fn trim_whitespace(input: &mut &str) {
    loop {
        let checkpoint = *input;
        let Some(first) = take_first(input) else {
            return;
        };
        if !first.is_ascii_whitespace() {
            *input = checkpoint;
            return;
        }
    }
}

static ESCAPED_CHARS: &'static str = "\\\"'+*?()[]=";

// Parses a terminal expression. These expressions always have the highest binding precendence.
fn parse_terminal(input: &mut &str) -> Result<GrammarEx, GrammarexParseError> {
    trim_whitespace(input);
    let checkpoint = *input;
    let first = take_first(input).ok_or(GrammarexParseError::UnexpectedEnd)?;
    if first == '\\' {
        return handle_escaped_char(input);
    }
    if first == '[' {
        return parse_backet_inner(input);
    }
    if first == '(' {
        return parse_parentheses(input);
    }
    if first == '"' {
        return parse_string(input);
    }
    if first.is_ascii_alphabetic() {
        *input = checkpoint;
        return parse_varname(input);
    }
    *input = checkpoint;
    Err(GrammarexParseError::DidntFollowExpression(first))
}

fn parse_varname(input: &mut &str) -> Result<GrammarEx, GrammarexParseError> {
    for (i, c) in input.char_indices() {
        if !c.is_ascii_alphabetic() {
            let res = GrammarEx::Var(input[..i].to_string());
            *input = &input[i..];
            return Ok(res);
        }
    }
    *input = "";
    return Ok(GrammarEx::Var(input.to_string()));
}
fn parse_string(input: &mut &str) -> Result<GrammarEx, GrammarexParseError> {
    let mut res = Vec::new();
    while let Some(c) = take_first(input) {
        if c == '"' {
            return Ok(GrammarEx::Seq(res));
        } else {
            res.push(GrammarEx::Char(c));
        }
    }
    return Err(GrammarexParseError::MismatchedQuotes);
}

//Parses a terminal expression and any tightly bound modifiers.
fn parse_single_item(input: &mut &str) -> Result<GrammarEx, GrammarexParseError> {
    let mut top = parse_terminal(input)?;
    trim_whitespace(input);
    loop {
        if let Some(first) = peek(input) {
            let sym_match = match first {
                '?' => Ok(GrammarEx::Optional(Box::new(top))),
                '+' => Ok(GrammarEx::Plus(Box::new(top))),
                '*' => Ok(GrammarEx::Star(Box::new(top))),
                _ => Err(top),
            };
            match sym_match {
                Ok(expr) => {
                    //The peek matched a symbol, so use it.
                    take_first(input);
                    //There might be more modifiers, so loop.
                    top = expr;
                }
                Err(expr) => {
                    return Ok(expr);
                }
            }
        } else {
            return Ok(top);
        }
    }
}

//Parses an assignment or tighter binding.
fn parse_assignment(input: &mut &str) -> Result<GrammarEx, GrammarexParseError> {
    let current = parse_single_item(input)?;
    trim_whitespace(input);
    if let Some('=') = peek(input) {
        take_first(input);
        Ok(GrammarEx::Assign(
            Box::new(current),
            Box::new(parse_single_item(input)?),
        ))
    } else {
        Ok(current)
    }
}

//Parses a sequence of items. If the sequence is empty it returns an error.
fn parse_seq(input: &mut &str) -> Result<GrammarEx, GrammarexParseError> {
    let mut res = vec![parse_assignment(input)?];
    while let Ok(next) = parse_assignment(input) {
        res.push(next);
    }
    Ok(match res.len() {
        1 => res.remove(0),
        _ => GrammarEx::Seq(res),
    })
}

fn parse_or_group(input: &mut &str) -> Result<GrammarEx, GrammarexParseError> {
    let mut res = vec![parse_seq(input)?];
    loop {
        trim_whitespace(input);
        if let Some('|') = peek(input) {
            take_first(input);
            if let Ok(next) = parse_seq(input) {
                res.push(next);
            } else {
                break;
            }
        } else {
            break;
        }
    }
    Ok(match res.len() {
        1 => res.remove(0),
        _ => GrammarEx::Alt(res),
    })
}

pub fn parse_grammarex(input: &mut &str) -> Result<GrammarEx, GrammarexParseError> {
    trim_whitespace(input);
    if input.is_empty() {
        Ok(GrammarEx::Epsilon)
    } else {
        parse_or_group(input)
    }
}

fn parse_parentheses(input: &mut &str) -> Result<GrammarEx, GrammarexParseError> {
    let res = parse_grammarex(input)?;
    if let Some(')') = peek(input) {
        Ok(res)
    } else {
        Err(GrammarexParseError::MismatchedParenthesis)
    }
}
fn handle_escaped_char(input: &mut &str) -> Result<GrammarEx, GrammarexParseError> {
    let escaped = take_first(input).ok_or(GrammarexParseError::UnexpectedEnd)?;
    if ESCAPED_CHARS.contains(escaped) {
        Ok(GrammarEx::Char(escaped))
    } else {
        return Err(GrammarexParseError::InvalidEscape);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_char_class() {
        let result = parse_grammarex(&mut "[abc]").unwrap();
        assert_eq!(
            GrammarEx::Alt(vec![
                GrammarEx::Char('a'),
                GrammarEx::Char('b'),
                GrammarEx::Char('c')
            ]),
            result
        );
    }

    #[test]
    fn test_single_char_class() {
        let result = parse_grammarex(&mut "[a]").unwrap();
        assert_eq!(GrammarEx::Alt(vec![GrammarEx::Char('a'),]), result);
    }

    #[test]
    fn test_char_class_range() {
        let result = parse_grammarex(&mut "[a-c]").unwrap();
        assert_eq!(
            GrammarEx::Alt(vec![GrammarEx::CharRangeInclusive('a','c')],),
            result
        );
    }
    #[test]
    fn test_dash_at_end() {
        let result = parse_grammarex(&mut "[abc-]").unwrap();
        assert_eq!(
            GrammarEx::Alt(vec![
                GrammarEx::Char('a'),
                GrammarEx::Char('b'),
                GrammarEx::Char('c'),
                GrammarEx::Char('-')
            ]),
            result
        );
    }
    #[test]
    fn test_dash_at_start() {
        let result = parse_grammarex(&mut "[-abc]").unwrap();
        assert_eq!(
            GrammarEx::Alt(vec![
                GrammarEx::Char('-'),
                GrammarEx::Char('a'),
                GrammarEx::Char('b'),
                GrammarEx::Char('c'),
            ]),
            result
        );
    }

    #[test]
    fn test_star() {
        let result = parse_grammarex(&mut "[a]*").unwrap();
        assert_eq!(
            GrammarEx::Star(Box::new(GrammarEx::Alt(vec![GrammarEx::Char('a'),]))),
            result
        );
    }

    #[test]
    fn test_question_star() {
        let result = parse_grammarex(&mut "[a]?*").unwrap();
        assert_eq!(
            GrammarEx::Star(Box::new(GrammarEx::Optional(Box::new(GrammarEx::Alt(
                vec![GrammarEx::Char('a'),]
            ))))),
            result
        );
    }

    #[test]
    fn test_string() {
        let result = parse_grammarex(&mut "\"abc\"").unwrap();
        assert_eq!(
            GrammarEx::Seq(vec![
                GrammarEx::Char('a'),
                GrammarEx::Char('b'),
                GrammarEx::Char('c')
            ]),
            result
        );
    }

    #[test]
    fn test_raw_string() {
        let result = parse_grammarex(&mut r#""abc""#).unwrap();
        assert_eq!(
            GrammarEx::Seq(vec![
                GrammarEx::Char('a'),
                GrammarEx::Char('b'),
                GrammarEx::Char('c')
            ]),
            result
        );
    }

    #[test]
    fn test_trimming() {
        let result = parse_grammarex(&mut r#" "abc" "#).unwrap();
        assert_eq!(
            GrammarEx::Seq(vec![
                GrammarEx::Char('a'),
                GrammarEx::Char('b'),
                GrammarEx::Char('c')
            ]),
            result
        );
    }

    #[test]
    fn test_simple_seq() {
        let result = parse_grammarex(&mut r#" "a" "b" "c" "#).unwrap();
        assert_eq!(
            GrammarEx::Seq(vec![
                GrammarEx::Seq(vec![GrammarEx::Char('a')]),
                GrammarEx::Seq(vec![GrammarEx::Char('b')]),
                GrammarEx::Seq(vec![GrammarEx::Char('c')]),
            ]),
            result
        );
    }
    #[test]
    fn test_var_assignment() {
        let result = parse_grammarex(&mut r#" abc = def "#).unwrap();
        assert_eq!(
            GrammarEx::Assign(
                Box::new(GrammarEx::Var("abc".to_owned())),
                Box::new(GrammarEx::Var("def".to_owned()))
            ),
            result
        );
    }

    #[test]
    fn test_alt() {
        let result = parse_grammarex(&mut r#" "a" | "b" "#).unwrap();
        assert_eq!(
            GrammarEx::Alt(vec![
                GrammarEx::Seq(vec![GrammarEx::Char('a')]),
                GrammarEx::Seq(vec![GrammarEx::Char('b')])
            ]),
            result
        );
    }
}
