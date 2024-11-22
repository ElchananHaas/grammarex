use std::{collections::BTreeSet, mem, ops::RangeInclusive};

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
enum Action {
    Epsilon,
    Match(CharClass),
    Call(String),
    CallAssign(String, String),
}
struct Node {
    out_edges: BTreeSet<usize>,
    in_edges: BTreeSet<usize>,
}

struct Edge {
    start: usize,
    end: usize,
    action: Action,
}

// Invariants:
// For each edge, it is in the node it starts at's out_edges and no other out_edges
// It is in the node it ends at's in_edges and no other in_edges
// For each node, an edge is in out_edges iff it starts at the node.
// It is in the in edges iff it ends at that node.
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
    fn create_node(&mut self) -> usize {
        self.nodes.push(Node {
            out_edges: BTreeSet::new(),
            in_edges: BTreeSet::new(),
        });
        self.nodes.len() - 1
    }
    fn add_edge(&mut self, start: usize, end: usize, action: Action) {
        let idx = self.edges.len();
        self.edges.push(Edge { start, end, action });
        self.nodes[start].out_edges.insert(idx);
        self.nodes[end].in_edges.insert(idx);
    }
    // Remove an edge. This method leaks the edge, this should be OK.
    fn remove_edge(&mut self, edge: usize) {
        let start = self.edges[edge].start;
        let end = self.edges[edge].end;
        self.nodes[start].out_edges.remove(&edge);
        self.nodes[end].in_edges.remove(&edge);
    }
    // Contracts the second node into the first node.
    // This method returns the index of the first node (the one still in use)
    // This method leaks the second node, but this should be OK.
    fn merge_nodes(&mut self, first: usize, second: usize) {
        if first == second {
            return;
        }
        //Steal the node's edges so I can work on them seperately from the rest of the graph.
        let mut second_out_edges = BTreeSet::new();
        let mut second_in_edges = BTreeSet::new();
        mem::swap(&mut second_out_edges, &mut self.nodes[second].out_edges);
        mem::swap(&mut second_in_edges, &mut self.nodes[second].in_edges);
        for edge in second_out_edges {
            self.edges[edge].start = first;
            self.nodes[first].out_edges.insert(edge);
            //No need to remove from the target's in edges - the edge still ends there.
        }
        for edge in second_in_edges {
            //No need to remove from the source's out edges - the edge still starts there.
            self.edges[edge].end = first;
            self.nodes[first].in_edges.insert(edge);
        }
    }
}

fn epsilon_closure(graph: &Graph, node: usize) -> BTreeSet<usize> {
    let mut closure = BTreeSet::new();
    let mut to_process = vec![node];
    while let Some(top) = to_process.pop() {
        if closure.contains(&top) {
            continue;
        }
        closure.insert(top);
        for &edge in &graph.nodes[top].out_edges {
            to_process.push(graph.edges[edge].end);
        }
    }
    closure
}

fn eliminate_epsilon_node(graph: &mut Graph, node: usize) {
    let eps_closure = epsilon_closure(graph, node);
}

fn eliminate_epsilon(graph: &mut Graph) {
    let node_len = graph.nodes.len();
    for i in 0..node_len {
        
    }
}

fn lower_nfa(graph: &mut Graph, expr: GrammarEx) -> Result<EpsilonNfa, LoweringError> {
    match expr {
        GrammarEx::Epsilon => {
            let node = graph.create_node();
            Ok(EpsilonNfa {
                start: node,
                end: node,
            })
        }
        GrammarEx::Char(c) => {
            let start = graph.create_node();
            let end = graph.create_node();
            graph.add_edge(start, end, Action::Match(CharClass::Char(c)));
            Ok(EpsilonNfa { start, end })
        }
        GrammarEx::CharRange(range) => {
            let start = graph.create_node();
            let end = graph.create_node();
            graph.add_edge(start, end, Action::Match(CharClass::Range(range)));
            Ok(EpsilonNfa { start, end })
        }
        GrammarEx::Seq(mut vec) => {
            let end = graph.create_node();
            let mut start = end;
            while let Some(current) = vec.pop() {
                let part = lower_nfa(graph, current)?;
                let new_start = part.start;
                graph.merge_nodes(part.end, start);
                start = new_start;
            }
            Ok(EpsilonNfa { start, end })
        }
        GrammarEx::Star(grammar_ex) => {
            let res = lower_nfa(graph, *grammar_ex)?;
            graph.add_edge(res.start, res.end, Action::Epsilon);
            graph.add_edge(res.end, res.start, Action::Epsilon);
            Ok(res)
        }
        GrammarEx::Alt(mut vec) => {
            if let Some(expr) = vec.pop() {
                let res = lower_nfa(graph, expr)?;
                while let Some(next) = vec.pop() {
                    let next_nfa = lower_nfa(graph, next)?;
                    graph.merge_nodes(res.start, next_nfa.start);
                    graph.merge_nodes(res.end, next_nfa.end);
                }
                Ok(res)
            } else {
                let node = graph.create_node();
                Ok(EpsilonNfa {
                    start: node,
                    end: node,
                })
            }
        }
        GrammarEx::Plus(grammar_ex) => {
            let res = lower_nfa(graph, *grammar_ex)?;
            graph.add_edge(res.end, res.start, Action::Epsilon);
            Ok(res)
        }
        GrammarEx::Optional(grammar_ex) => {
            let res = lower_nfa(graph, *grammar_ex)?;
            graph.add_edge(res.start, res.end, Action::Epsilon);
            Ok(res)
        }
        GrammarEx::Assign(var, expr) => {
            let GrammarEx::Var(var) = *var else {
                return Err(LoweringError::InvalidVariableAssignment);
            };
            let GrammarEx::Var(expr) = *expr else {
                return Err(LoweringError::InvalidVariableAssignment);
            };
            let start = graph.create_node();
            let end = graph.create_node();
            graph.add_edge(start, end, Action::CallAssign(var, expr));
            Ok(EpsilonNfa { start, end })
        }
        GrammarEx::Var(var) => {
            let start = graph.create_node();
            let end = graph.create_node();
            graph.add_edge(start, end, Action::Call(var));
            Ok(EpsilonNfa { start, end })
        }
    }
}
