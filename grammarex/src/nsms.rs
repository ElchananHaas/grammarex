use std::{
    collections::{HashMap, VecDeque},
    rc::Rc,
};

use thiserror::Error;

#[derive(PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Hash)]
pub enum CharMatch {
    Epsilon,
    Char(char),
    RangeInclusive(char, char),
}

impl CharMatch {
    pub fn matches(self, c: char) -> bool {
        match self {
            CharMatch::Epsilon => panic!("Epsilon transitions cannot be matched!"),
            CharMatch::Char(c2) => c2 == c,
            CharMatch::RangeInclusive(c_start, c_end) => c_start <= c && c <= c_end,
        }
    }
}
#[derive(Error, Debug)]
pub enum LoweringError {
    #[error("Invalid variable assignment")]
    InvalidVariableAssignment,
    #[error("Couldn't locate expression `{0}`")]
    UnknownExpression(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Action {
    Assign(String, String),
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CallData {
    pub name: String,
    pub target_machine: usize,
    pub return_node: usize,
}

#[derive(PartialEq, Eq, Clone, Debug, PartialOrd, Ord, Hash)]
pub enum NsmEdgeTransition {
    Move(CharMatch, usize),
    Call(CallData),
    //An edge of the NSM that returns to an unknown place and therefore
    //doesn't consume input.
    Return,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NsmEdgeData {
    pub transition: NsmEdgeTransition,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Machine {
    pub edges: Vec<Vec<NsmEdgeData>>,
    pub accept_epsilon_actions: Option<Vec<Action>>,
}

impl Machine {
    pub fn new() -> Self {
        Self {
            edges: vec![],
            accept_epsilon_actions: None,
        }
    }

    pub fn create_node(&mut self) -> usize {
        self.edges.push(Vec::new());
        self.edges.len() - 1
    }
}

pub struct MachineBuilder {
    edges: Vec<VecDeque<NsmEdgeData>>,
    names: Rc<HashMap<String, usize>>,
}

impl MachineBuilder {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            names: Rc::new(HashMap::new()),
        }
    }

    pub fn create_node(&mut self) -> usize {
        self.edges.push(VecDeque::new());
        self.edges.len() - 1
    }

    pub fn get_node_mut(&mut self, idx: usize) -> &mut VecDeque<NsmEdgeData> {
        &mut self.edges[idx]
    }

    pub fn set_names(&mut self, names: Rc<HashMap<String, usize>>) {
        self.names = names;
    }

    pub fn get_name(&self, name: &str) -> Result<usize, LoweringError> {
        self.names
            .get(name)
            .ok_or_else(|| LoweringError::UnknownExpression(name.to_string()))
            .copied()
    }
    pub fn build(self) -> Machine {
        Machine {
            edges: self
                .edges
                .into_iter()
                .map(|edge| edge.into_iter().collect())
                .collect(),
            accept_epsilon_actions: None,
        }
    }
}
