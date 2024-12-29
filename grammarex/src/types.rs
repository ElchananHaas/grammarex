#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarEx {
    Epsilon,
    Char(char),
    CharRangeInclusive(char, char),
    Seq(Vec<GrammarEx>),
    Star(Box<GrammarEx>),
    Alt(Vec<GrammarEx>),
    Plus(Box<GrammarEx>),
    Optional(Box<GrammarEx>),
    Assign(Box<GrammarEx>, Box<GrammarEx>),
    Var(String),
}
