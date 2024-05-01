use std::ops::RangeInclusive;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GrammarexParseError {
    #[error("Missing closing backet")]
    MissingClosingBracket,
    #[error("Unexpected end of input")]
    UnexpectedEnd,
    #[error("Invalid backslash escape")]
    InvalidEscape,
    #[error("You must close all inner parenthesis before closing quotes")]
    MismatchedQuotesAndParenthesis,
    #[error("The charachter`{0}` must follow an expression")]
    DidntFollowExpression(char),
}
pub enum GrammarEx {
    Char(char),
    CharRange(RangeInclusive<char>),
    Seq(Vec<GrammarEx>),
    Star(Box<GrammarEx>),
    Alt(Vec<GrammarEx>),
    Plus(Box<GrammarEx>),
    Optional(Box<GrammarEx>),
    Assign(String, Box<GrammarEx>),
    Call(String),
}

fn take_first(input: &mut &str) -> Option<char> {
    let first: char = input.chars().next()?;
    *input = &input[first.len_utf8()..];
    return Some(first);
}

fn peek_k<const N: usize>(input: &&str) -> Option<[char; N]> {
    let mut result: [char; N] = ['_'; N];
    let mut chars = input.chars();
    for i in 0..N {
        let char = chars.next()?;
        result[i] = char;
    }
    return Some(result);
}
fn peek(input: &&str, amount: usize) -> Option<char> {
    let item = input.chars().skip(amount).next()?;
    return Some(item);
}
fn skip(input: &mut &str, amount: usize) {
    for _ in 0..amount {
        let c = input.chars().next().expect("Uexpectedly ran out of input");
        *input = &input[c.len_utf8()..];
    }
}
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
                classes.push(GrammarEx::CharRange(c..=cs[2]));
                skip(input, 2);
            }
        }
        classes.push(GrammarEx::Char(c));
        first = false;
    }
}

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
enum Part {
    GrammarEx(GrammarEx),
    OrBar,
    Equals,
    VarName(String),
    Semicolon,
}
static ESCAPED_CHARS: &'static str = "\\\"'+*?";
static SINGLE_ITEMS_MODS: &'static str = "*+?";
pub fn parse_grammarex(input: &mut &str, mut in_string: bool) -> Result<GrammarEx, GrammarexParseError> {
    let mut parts:Vec<Part> = Vec::new();
    let mut started_string = false;
    loop {
        if !in_string {
            trim_whitespace(input);
        }
        let checkpoint = *input;
        let first = take_first(input).ok_or(GrammarexParseError::UnexpectedEnd)?;
        if first == '\\' {
            let escaped = take_first(input).ok_or(GrammarexParseError::UnexpectedEnd)?;
            if ESCAPED_CHARS.contains(escaped) {
                parts.push(Part::GrammarEx(GrammarEx::Char(escaped)));
            } else {
                return Err(GrammarexParseError::InvalidEscape);
            }
        }
        if first == '[' {
            parts.push(Part::GrammarEx(parse_backet_inner(input)?));
        }
        if first == '(' {
            parts.push(Part::GrammarEx(parse_grammarex(input, in_string)?));
        }
        if first == '"' {
            if in_string && started_string {
                in_string = false;
            } else if !in_string {
                in_string = true;
                started_string = true;
                parts.push(Part::GrammarEx(parse_grammarex(input, true)?));
            } else {
                return Err(GrammarexParseError::MismatchedQuotesAndParenthesis);
            }
        }
        if SINGLE_ITEMS_MODS.contains(first) {
            if let Some(top) = parts.pop() {
                if let Part::GrammarEx(ex) = top {
                    let ex = match first {
                        '?' => GrammarEx::Optional(Box::new(ex)),
                        '+' => GrammarEx::Plus(Box::new(ex)),
                        '*' => GrammarEx::Star(Box::new(ex)),
                        _ => panic!("Operator {} isn't a modifier.",first)
                    };
                    parts.push(Part::GrammarEx(ex));
                } else {
                    return Err(GrammarexParseError::DidntFollowExpression(first));
                }
            } else {
                return Err(GrammarexParseError::DidntFollowExpression(first));
            }
        }
    }

}

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
