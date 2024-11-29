use std::{collections::{BTreeSet, VecDeque}, mem, ops::RangeInclusive};

use thiserror::Error;

use crate::types::GrammarEx;

#[derive(Error, Debug)]
pub enum LoweringError {
    #[error("Invalid variable assignment")]
    InvalidVariableAssignment,
}

enum CharClass {
    Char(char),
    Range(RangeInclusive<char>),
}
enum EpsNfaCharMatch {
    Epsilon,
    Match(CharClass)
}
struct EpsNfaAction {
    char_match : EpsNfaCharMatch,
    actions: Vec<Action>
}
enum Action {
    Call(String),
    CallAssign(String, String),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct EdgeIndex(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct NodeIndex(pub usize);
struct EdgeRef {
    index: EdgeIndex,
    priority: isize
}
struct Node {
    //Out edges are sorted by priority for taking them.
    out_edges: VecDeque<EdgeIndex>,
    in_edges: BTreeSet<EdgeIndex>,
}

struct Edge {
    start: NodeIndex,
    end: NodeIndex,
    action: EpsNfaAction,
}

// Invariants:
// For each edge, it is in the node it starts at's out_edges and no other out_edges
// It is in the node it ends at's in_edges and no other in_edges
// For each node, an edge is in its out_edges iff it starts at the node.
// Its in its in_edges iff it ends at that node.
struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

#[derive(Clone)]
struct EpsilonNfa {
    start: usize,
    end: usize,
}

impl Graph {
    fn create_node(&mut self) -> NodeIndex {
        self.nodes.push(Node {
            out_edges: VecDeque::new(),
            in_edges: BTreeSet::new(),
        });
        NodeIndex(self.nodes.len() - 1)
    }
    fn add_edge_lowest_priority(&mut self, start: NodeIndex, end: NodeIndex, action: EpsNfaAction) {
        let idx = self.edges.len();
        self.edges.push(Edge { start, end, action });
        self.nodes[start.0].out_edges.push_back(EdgeIndex(idx));
        self.nodes[end.0].in_edges.insert(EdgeIndex(idx));
    }

    fn add_edge_highest_priority(&mut self, start: NodeIndex, end: NodeIndex, action: EpsNfaAction) {
        let idx = self.edges.len();
        self.edges.push(Edge { start, end, action });
        self.nodes[start.0].out_edges.push_front(EdgeIndex(idx));
        self.nodes[end.0].in_edges.insert(EdgeIndex(idx));
    }
}

fn epsilon_closure(graph: &Graph, node: usize) -> Vec<usize> {
    let mut closure = BTreeSet::new();
    let mut to_process = vec![node];
    let mut res = Vec::new();
    while let Some(top) = to_process.pop() {
        if closure.contains(&top) {
            continue;
        }
        closure.insert(top);
        res.push(top);
        for &edge in &graph.nodes[top].out_edges {
            to_process.push(graph.edges[edge].end);
        }
    }
    res
}

fn eliminate_epsilon(graph: &mut Graph) {
    let node_len = graph.nodes.len();
    for i in 0..node_len {
        
    }
}

fn epsilon_no_action() -> EpsNfaAction {
    EpsNfaAction { char_match: EpsNfaCharMatch::Epsilon, actions: vec![] }
}
//Takes in an Epsilon NFA graph, an end node, and an expresion to lower. Returns the start node of the expression
//lowered as an NFA with end_node accepting.
fn lower_nfa_inner(graph: &mut Graph, end_node : NodeIndex, expr: GrammarEx) -> Result<NodeIndex, LoweringError> {
    match expr {
        GrammarEx::Epsilon => {
            Ok(end_node)
        },
        GrammarEx::Char(c) => {
            let start_node = graph.create_node();
            graph.add_edge_lowest_priority(start_node, end_node, EpsNfaAction { char_match: EpsNfaCharMatch::Match(CharClass::Char(c)), actions: vec![] });
            Ok(start_node)
        },
        GrammarEx::CharRange(range) => {
            let start_node = graph.create_node();
            graph.add_edge_lowest_priority(start_node, end_node, EpsNfaAction { char_match: EpsNfaCharMatch::Match(CharClass::Range(range)), actions: vec![] });
            Ok(start_node)
        },
        GrammarEx::Seq(mut exprs) => {
            let mut start_node = end_node;
            while let Some(expr) = exprs.pop() {
                start_node = lower_nfa_inner(graph, start_node, expr)?;
            }
            Ok(start_node)
        },
        GrammarEx::Star(expr) => {
            let start_node = lower_nfa_inner(graph, end_node, *expr)?;
            //For a star operator, looping back is always highest priority. Skipping it is lowest priority
            graph.add_edge_highest_priority(end_node, start_node, epsilon_no_action());
            graph.add_edge_lowest_priority(start_node, end_node, epsilon_no_action());
            Ok(start_node)
        },
        GrammarEx::Plus(expr) => {
            let start_node = lower_nfa_inner(graph, end_node, *expr)?;
            //For a plus operator, looping back is always highest priority. It can't be skipped.
            graph.add_edge_highest_priority(end_node, start_node, epsilon_no_action());
            Ok(start_node)
        },
        GrammarEx::Alt(mut vec) => {
            //The code could get away with not creating a node if the
            //alt is non-empty, but that would make it more complicated.
            let start_node = graph.create_node();
            let mut current = start_node;
            while let Some(expr) = vec.pop() {
                let new_current = lower_nfa_inner(graph, end_node, expr)?;
                //In an alt, going on to the next alternitive has lower priority than following the current alternitive.
                graph.add_edge_lowest_priority(current, new_current, epsilon_no_action());
                current = new_current;
            }
            Ok(start_node)
        },
        GrammarEx::Optional(grammar_ex) =>  {
            let start_node = lower_nfa_inner(graph, end_node, *grammar_ex)?;
            //An option can be skipped, the action has lowest priority over consuming input
            graph.add_edge_lowest_priority(start_node, end_node, epsilon_no_action());
            Ok(start_node)
        },
        GrammarEx::Assign(var, expr) => {
            let GrammarEx::Var(var) = *var else {
                return Err(LoweringError::InvalidVariableAssignment);
            };
            let GrammarEx::Var(expr) = *expr else {
                return Err(LoweringError::InvalidVariableAssignment);
            };
            let start_node = graph.create_node();
            graph.add_edge_lowest_priority(start_node, end_node, EpsNfaAction { char_match: EpsNfaCharMatch::Epsilon, actions: vec![Action::CallAssign(var, expr)] });
            Ok(start_node)
        },
        GrammarEx::Var(expr) => {
            let start_node = graph.create_node();
            graph.add_edge_lowest_priority(start_node, end_node, EpsNfaAction { char_match: EpsNfaCharMatch::Epsilon, actions: vec![Action::Call(expr)] });
            Ok(start_node)
        },
    }
}