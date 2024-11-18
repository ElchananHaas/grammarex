
use std::ops::RangeInclusive;

use thiserror::Error;

use crate::types::GrammarEx;

#[derive(Error, Debug)]
pub enum LoweringError {
    #[error("Invalid variable assignment")]
    InvalidVariableAssignment,
}

enum CharClass {
    Char(char),
    Range(RangeInclusive<char>)
}
enum Action {
    Epsilon,
    Match(CharClass),
    Call(String),
    CallAssign(String, String)
}
struct Node {
    edges: Vec<Edge>,
}

struct Edge {
    target: usize,
    action: Action,
}

struct Graph {
    nodes: Vec<Node>,
}

#[derive(Clone)]
struct EpsilonNfa {
    start: usize,
    end: usize
}

impl Graph {
    fn create_node(&mut self) -> usize {
        self.nodes.push(Node { edges: vec![] });
        self.nodes.len() - 1
    }
    fn add_edge(&mut self, start: usize, end: usize, action: Action) {
        self.nodes[start].edges.push(Edge {
            target: end,
            action,
        });
    }
}

//creates a pair of nodes with an epsilon transition between them.
fn create_epsilon_pair(graph: &mut Graph) -> EpsilonNfa{
    let start = graph.create_node();
    let end = graph.create_node();
    graph.add_edge(start, end, Action::Epsilon);
    EpsilonNfa {
        start, 
        end
    }
}
fn lower_nfa(graph: &mut Graph, expr: GrammarEx) -> Result<EpsilonNfa, LoweringError> {
    match expr {
        GrammarEx::Epsilon => {
            Ok(create_epsilon_pair(graph))
        },
        GrammarEx::Char(c) => {
            let start = graph.create_node();
            let end = graph.create_node();
            graph.add_edge(start, end, Action::Match(CharClass::Char(c)));
            Ok(EpsilonNfa {
                start, 
                end
            })
        },
        GrammarEx::CharRange(range) => {
            let start = graph.create_node();
            let end = graph.create_node();
            graph.add_edge(start, end, Action::Match(CharClass::Range(range)));
            Ok(EpsilonNfa {
                start, 
                end
            })
        },
        GrammarEx::Seq(mut vec) => {
            let mut res = create_epsilon_pair(graph);
            while let Some(current) = vec.pop() {
                let front_part = lower_nfa(graph, current)?;
                res = EpsilonNfa {
                    start: front_part.start, 
                    end: res.end
                }
            } 
            Ok(res)
        },
        GrammarEx::Star(grammar_ex) => {
            let res = lower_nfa(graph, *grammar_ex)?;
            graph.add_edge(res.start, res.end, Action::Epsilon);
            graph.add_edge(res.end, res.start, Action::Epsilon);
            Ok(res)
        },
        GrammarEx::Alt(vec) => {
            let start = graph.create_node();
            let end = graph.create_node();
            for expr in vec {
                let part = lower_nfa(graph, expr)?;
                graph.add_edge(start, part.start, Action::Epsilon);
                graph.add_edge(part.end, end, Action::Epsilon);
            }
            Ok(EpsilonNfa {
                start,
                end
            })
        },
        GrammarEx::Plus(grammar_ex) => {
            let res = lower_nfa(graph, *grammar_ex)?;
            graph.add_edge(res.end, res.start, Action::Epsilon);
            Ok(res)
        },
        GrammarEx::Optional(grammar_ex) => {
            let res = lower_nfa(graph, *grammar_ex)?;
            graph.add_edge(res.start, res.end, Action::Epsilon);
            Ok(res)
        },
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
            Ok(EpsilonNfa {
                start,
                end
            })
        },
        GrammarEx::Var(var) => {
            let start = graph.create_node();
            let end = graph.create_node();
            graph.add_edge(start, end, Action::Call(var));
            Ok(EpsilonNfa {
                start,
                end
            })
        },
    }
}
