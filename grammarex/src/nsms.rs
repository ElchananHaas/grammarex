use std::collections::{HashMap, HashSet, VecDeque};

#[derive(PartialEq, Eq, Clone, Debug, PartialOrd, Ord)]
pub enum CharClass {
    Char(char),
    RangeInclusive(char, char),
}
#[derive(PartialEq, Eq, Clone, Debug, PartialOrd, Ord)]
pub enum EpsCharMatch {
    Epsilon,
    Match(CharClass),
}
//An edge of the NSM that consumes input
#[derive(PartialEq, Eq, Clone, Debug, PartialOrd, Ord)]
pub struct NsmConsumeEdge {
    char_match: CharClass,
    target_node: usize,
}
#[derive(PartialEq, Eq, Clone, Debug, PartialOrd, Ord)]
pub enum NsmEdgeTransition {
    Consume(NsmConsumeEdge),
    Call(CallData),
    //An edge of the NSM that returns to an unknown place and therefore
    //doesn't consume input.
    Return,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct NsmEdgeData {
    pub transition: NsmEdgeTransition,
    pub actions: Vec<Action>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Action {
    Assign(String, String),
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CallData {
    name: String,
    target_node: usize,
    return_node: usize,
}

#[derive(PartialEq, Eq, Clone, Debug, PartialOrd, Ord)]
pub enum EpsNsmEdgeTransition {
    Move(EpsCharMatch, usize),
    Call(CallData),
    //An edge of the NSM that returns to an unknown place and therefore
    //doesn't consume input.
    Return,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Machine<EdgeData> {
    pub edges: Vec<Vec<EdgeData>>,
    pub accepts_epsilon: bool,
    pub starting_node: usize,
}

pub struct Machines<EdgeData> {
    pub machines: Vec<Machine<EdgeData>>,
    pub names_mapping: HashMap<String, usize>,
}

impl<EdgeData> Machine<EdgeData> {
    pub fn new() -> Self {
        Self {
            edges: vec![],
            accepts_epsilon: false,
            starting_node: 0,
        }
    }
}

pub struct MachineBuilder {
    edges: Vec<VecDeque<NsmEdgeData>>,
}

impl MachineBuilder {
    pub fn new() -> Self {
        Self { edges: Vec::new() }
    }

    pub fn create_node(&mut self) -> usize {
        self.edges.push(VecDeque::new());
        self.edges.len() - 1
    }

    pub fn get_node_mut(&mut self, idx: usize) -> &mut VecDeque<NsmEdgeData> {
        &mut self.edges[idx]
    }

    pub fn build(self) -> Machine<NsmEdgeData> {
        Machine {
            edges: self
                .edges
                .into_iter()
                .map(|edge| edge.into_iter().collect())
                .collect(),
            accepts_epsilon: false,
            starting_node: 0,
        }
    }
}
