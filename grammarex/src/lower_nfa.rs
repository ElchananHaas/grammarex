use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    mem,
    ops::RangeInclusive,
};

use thiserror::Error;

use crate::types::GrammarEx;

#[derive(Error, Debug)]
pub enum LoweringError {
    #[error("Invalid variable assignment")]
    InvalidVariableAssignment,
    #[error("Couldn't locate expression `{0}`")]
    UnknownExpression(String),
}

enum CharClass {
    Char(char),
    Range(RangeInclusive<char>),
}
enum EpsNfaCharMatch {
    Epsilon,
    Match(CharClass),
}
struct EpsNfaAction {
    char_match: EpsNfaCharMatch,
    actions: Vec<Action>,
}
enum Action {
    Call(CallData),
    CallAssign(CallData, String),
}

struct CallData {
    name: String,
    return_node: NodeIndex,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct EdgeIndex(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct NodeIndex(pub usize);
struct EdgeRef {
    index: EdgeIndex,
    priority: isize,
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

    fn add_edge_highest_priority(
        &mut self,
        start: NodeIndex,
        end: NodeIndex,
        action: EpsNfaAction,
    ) {
        let idx = self.edges.len();
        self.edges.push(Edge { start, end, action });
        self.nodes[start.0].out_edges.push_front(EdgeIndex(idx));
        self.nodes[end.0].in_edges.insert(EdgeIndex(idx));
    }
}

fn eliminate_epsilon(graph: &mut Graph) {
    for i in 0..graph.nodes.len() {
        add_epsilon_closure(graph, i);
    }
}

fn add_epsilon_closure(graph: &mut Graph, i: usize) {}

fn epsilon_no_action() -> EpsNfaAction {
    EpsNfaAction {
        char_match: EpsNfaCharMatch::Epsilon,
        actions: vec![],
    }
}

//Takes in an Epsilon NFA graph, a start node, a name table from expression names to start nodes and an expresion to lower.
//Returns the end node of the expression lowered as an NFA with end_node accepting.
fn lower_nfa_inner(
    graph: &mut Graph,
    start_node: NodeIndex,
    name_table: &HashMap<String, NodeIndex>,
    expr: GrammarEx,
) -> Result<NodeIndex, LoweringError> {
    match expr {
        GrammarEx::Epsilon => Ok(start_node),
        GrammarEx::Char(c) => {
            let end_node = graph.create_node();
            graph.add_edge_lowest_priority(
                start_node,
                end_node,
                EpsNfaAction {
                    char_match: EpsNfaCharMatch::Match(CharClass::Char(c)),
                    actions: vec![],
                },
            );
            Ok(end_node)
        }
        GrammarEx::CharRange(range) => {
            let end_node = graph.create_node();
            graph.add_edge_lowest_priority(
                start_node,
                end_node,
                EpsNfaAction {
                    char_match: EpsNfaCharMatch::Match(CharClass::Range(range)),
                    actions: vec![],
                },
            );
            Ok(end_node)
        }
        GrammarEx::Seq(exprs) => {
            let mut end_node = start_node;
            for expr in exprs {
                end_node = lower_nfa_inner(graph, end_node, name_table, expr)?;
            }
            Ok(end_node)
        }
        GrammarEx::Star(expr) => {
            let end_node = lower_nfa_inner(graph, start_node, name_table, *expr)?;
            //For a star operator, looping back is always highest priority. Skipping it is lowest priority
            graph.add_edge_highest_priority(end_node, start_node, epsilon_no_action());
            graph.add_edge_lowest_priority(start_node, end_node, epsilon_no_action());
            Ok(end_node)
        }
        GrammarEx::Plus(expr) => {
            let end_node = lower_nfa_inner(graph, start_node, name_table, *expr)?;
            //For a plus operator, looping back is always highest priority. It can't be skipped.
            graph.add_edge_highest_priority(end_node, start_node, epsilon_no_action());
            Ok(end_node)
        }
        GrammarEx::Alt(exprs) => {
            //The code could get away with not creating a node if the
            //alt is non-empty, but that would make it more complicated.
            let end_node = graph.create_node();
            for expr in exprs {
                let end = lower_nfa_inner(graph, start_node, name_table, expr)?;
                graph.add_edge_lowest_priority(end, end_node, epsilon_no_action());
            }
            Ok(end_node)
        }
        GrammarEx::Optional(grammar_ex) => {
            let end_node = lower_nfa_inner(graph, start_node, name_table, *grammar_ex)?;
            //An option can be skipped, skipping has lowest priority over consuming input
            graph.add_edge_lowest_priority(start_node, end_node, epsilon_no_action());
            Ok(end_node)
        }
        GrammarEx::Assign(var, expr) => {
            let GrammarEx::Var(var) = *var else {
                return Err(LoweringError::InvalidVariableAssignment);
            };
            let GrammarEx::Var(expr) = *expr else {
                return Err(LoweringError::InvalidVariableAssignment);
            };
            let expr_node = *name_table
                .get(&expr)
                .ok_or_else(|| LoweringError::UnknownExpression(expr.clone()))?;
            let return_node = graph.create_node();
            let call_data = CallData {
                name: expr.clone(),
                return_node,
            };
            graph.add_edge_lowest_priority(
                start_node,
                expr_node,
                EpsNfaAction {
                    char_match: EpsNfaCharMatch::Epsilon,
                    actions: vec![Action::CallAssign(call_data, var)],
                },
            );
            Ok(return_node)
        }
        GrammarEx::Var(expr) => {
            let expr_node = *name_table
                .get(&expr)
                .ok_or_else(|| LoweringError::UnknownExpression(expr.clone()))?;
            let return_node = graph.create_node();
            let call_data = CallData {
                name: expr,
                return_node,
            };
            graph.add_edge_lowest_priority(
                start_node,
                expr_node,
                EpsNfaAction {
                    char_match: EpsNfaCharMatch::Epsilon,
                    actions: vec![Action::Call(call_data)],
                },
            );
            Ok(return_node)
        }
    }
}
