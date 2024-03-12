use std::ops::Range;

enum CharClass {
    Char(char),
    CharRange(Range<char>),
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
