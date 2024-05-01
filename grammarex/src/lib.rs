use std::ops::{Range, RangeInclusive};

use thiserror::Error;

#[derive(Error, Debug)]
enum GrammarexParseError {
    #[error("Missing closing backet")]
    MissingClosingBracket,
}
enum CharClass {
    Char(char),
    CharRange(RangeInclusive<char>),
    Union(Vec<CharClass>),
}

impl CharClass {
    fn contains(&self, c: char) -> bool {
        match self {
            CharClass::Char(x) => c == *x,
            CharClass::CharRange(r) => r.contains(&c),
            CharClass::Union(classes) => classes.into_iter().any(|class| class.contains(c)),
        }
    }
}

fn take_first(input: &mut &str) -> Option<char> {
    let first: char = input.chars().next()?;
    *input = &input[first.len_utf8()..];
    return Some(first);
}

fn peek_k<const N: usize>(input: &&str) -> Option<[char; N]>{
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
fn parse_backet_inner(input: &mut &str) -> Result<CharClass, GrammarexParseError> {
    let mut classes = Vec::new();
    let mut first = true;
    loop {
        let c = take_first(input).ok_or(GrammarexParseError::MissingClosingBracket)?;
        //A regex backet can match the ']' char if it is the first char in the [] expression.
        if c == ']' && !first {
            return Ok(CharClass::Union(classes));
        }
        //[a-c] represents [abc] but if the dash is first or last, it is considered a literal like in [ax-]
        if let Some(cs) = peek_k::<2>(input) {
            if cs[0] == '-' && cs[1] != ']' {
                classes.push(CharClass::CharRange(c..=cs[2]));
                skip(input, 2);
            }
        }
        classes.push(CharClass::Char(c));
        first = false;
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
