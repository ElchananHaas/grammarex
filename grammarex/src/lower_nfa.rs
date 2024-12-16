use std::{
    collections::{HashMap, HashSet, VecDeque},
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

#[derive(PartialEq, Eq, Clone, Debug)]
enum CharClass {
    Char(char),
    Range(RangeInclusive<char>),
}
#[derive(PartialEq, Eq, Clone)]
enum EpsCharMatch {
    Epsilon,
    Match(CharClass),
}
//An edge of the NSM that consumes input
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct NsmConsumeEdge {
    char_match: CharClass,
    target_node: NodeIndex,
}
#[derive(PartialEq, Eq, Clone, Debug)]
pub enum NsmEdgeTransition {
    Consume(NsmConsumeEdge),
    //An edge of the NSM that returns to an unknown place and therefore
    //doesn't consume input.
    Return,
}
#[derive(Clone, Debug)]
pub struct NsmEdgeData {
    pub transition: NsmEdgeTransition,
    pub actions: Vec<Action>,
}
#[derive(Clone, Debug)]
pub enum Action {
    Assign(String, String),
}
#[derive(Clone)]
pub struct CallData {
    name: String,
    target_node: NodeIndex,
    return_node: NodeIndex,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeIndex(usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIndex(usize);

#[derive(Clone)]
pub enum EdgeTarget {
    Call(CallData),
    NodeIndex(NodeIndex),
    Return,
}

struct EpsNsmEdgeData {
    target: EdgeTarget,
    char_match: EpsCharMatch,
    actions: Vec<Action>,
}
#[derive(Debug)]
struct Node {
    //Out edges are sorted by priority for taking them.
    out_edges: VecDeque<EdgeIndex>,
}
#[derive(Clone, Debug)]
struct Edge<EdgeData> {
    data: EdgeData,
}
#[derive(Debug)]
pub struct NsmInstructions {
    pub start_node: NodeIndex,
    pub graph: Graph<NsmEdgeData>,
}

pub fn compile(
    machines: HashMap<String, GrammarEx>,
    start_machine: &String,
) -> Result<NsmInstructions, LoweringError> {
    let mut graph: Graph<EpsNsmEdgeData> = Graph {
        nodes: Vec::new(),
        edges: Vec::new(),
    };
    let machine_starts: HashMap<String, _> = machines
        .iter()
        .map(|(name, _)| ((name.clone()), graph.create_node()))
        .collect();
    let mut end_nodes = HashMap::new();
    for machine in machines {
        let end_node = lower_nsm(
            &mut graph,
            *machine_starts.get(&machine.0).expect("It exists"),
            &machine_starts,
            machine.1,
        )?;
        end_nodes.insert(machine.0.clone(), end_node);
    }
    let elim = eliminate_epsilon(&mut graph);
    Ok(NsmInstructions {
        start_node: *machine_starts
            .get(start_machine)
            .ok_or_else(|| LoweringError::UnknownExpression(start_machine.clone()))?,
        graph: elim,
    })
}
// Invariants:
// For each edge, it is in the node it starts at's out_edges and no other out_edges
// For each node, an edge is in its out_edges iff it starts at the node.
#[derive(Debug)]
pub struct Graph<EdgeData> {
    nodes: Vec<Node>,
    edges: Vec<Edge<EdgeData>>,
}

impl<EdgeData> Graph<EdgeData> {
    fn create_node(&mut self) -> NodeIndex {
        self.nodes.push(Node {
            out_edges: VecDeque::new(),
        });
        NodeIndex(self.nodes.len() - 1)
    }
    fn get_node(&self, idx: NodeIndex) -> &Node {
        &self.nodes[idx.0]
    }
    fn get_edge(&self, idx: EdgeIndex) -> &Edge<EdgeData> {
        &self.edges[idx.0]
    }
    fn add_edge_lowest_priority(&mut self, start: NodeIndex, data: EdgeData) {
        let idx = self.edges.len();
        self.edges.push(Edge { data });
        self.nodes[start.0].out_edges.push_back(EdgeIndex(idx));
    }

    fn add_edge_highest_priority(&mut self, start: NodeIndex, data: EdgeData) {
        let idx = self.edges.len();
        self.edges.push(Edge { data });
        self.nodes[start.0].out_edges.push_front(EdgeIndex(idx));
    }
}

struct ElimContext<'a> {
    start: NodeIndex,
    graph: &'a Graph<EpsNsmEdgeData>,
}

#[derive(Clone)]
struct ReturnRef {
    node: NodeIndex,
    prior: usize,
}
struct PathData<'a> {
    actions: Vec<&'a Vec<Action>>,
    // Each node in the list points back to the prior call stack, effectively
    // forming a linked list.
    return_stacks: Vec<Option<ReturnRef>>,
}

impl<'a> PathData<'a> {
    fn new() -> Self {
        PathData {
            actions: Vec::new(),
            return_stacks: Vec::new(),
        }
    }

    // Simulates following an edge. If this method returns None, then there was a return action with
    // no known caller. Returns the destination node.
    fn follow_edge(
        &mut self,
        graph: &'a Graph<EpsNsmEdgeData>,
        edge_index: EdgeIndex,
    ) -> Option<NodeIndex> {
        assert!(self.actions.len() == self.return_stacks.len());
        let edge = graph.get_edge(edge_index);
        self.actions.push(&edge.data.actions);
        match &edge.data.target {
            // In this case, the edge's target is a node index. So
            // push the actions.
            EdgeTarget::NodeIndex(target) => {
                let current_return_ref = self.return_stacks.last().cloned().unwrap_or_else(|| None);
                self.return_stacks.push(current_return_ref);
                Some(*target)
            }
            EdgeTarget::Return => {
                if let Some(ret) = &self.return_stacks.last().cloned().flatten() {
                    self.return_stacks
                        .push(self.return_stacks[ret.prior].clone());
                    Some(ret.node)
                } else {
                    //Push a none here to keep the return stacks and action stacks balanced
                    self.return_stacks.push(None);
                    None
                }
            }
            EdgeTarget::Call(call_data) => {
                let current_last_idx = self.return_stacks.len() - 1;
                self.return_stacks.push(Some(ReturnRef {
                    node: call_data.return_node,
                    prior: current_last_idx,
                }));
                Some(call_data.target_node)
            }
        }
    }
    //Reverses the latest actions.
    fn pop(&mut self) {
        self.actions.pop();
        self.return_stacks.pop();
    }

    fn get_actions(&self) -> Vec<Action> {
        let mut res = Vec::new();
        for segment in &self.actions {
            for action in *segment {
                res.push(action.clone());
            }
        }
        res
    }
}
//Takes in an epsilon graph and the accepting nodes. Returns an equivilent graph that has had epsilon elimination performed.
fn eliminate_epsilon(graph: &Graph<EpsNsmEdgeData>) -> Graph<NsmEdgeData> {
    let mut new_graph: Graph<NsmEdgeData> = Graph {
        nodes: Vec::new(),
        edges: Vec::new(),
    };
    for _ in 0..graph.nodes.len() {
        new_graph.create_node();
    }
    for i in 0..graph.nodes.len() {
        let mut visited_set: HashSet<NodeIndex> = HashSet::new();
        let mut path = PathData::new();
        let context = ElimContext {
            start: NodeIndex(i),
            graph: &graph,
        };
        replace_with_epsilon_closure(
            &context,
            NodeIndex(i),
            &mut new_graph,
            &mut visited_set,
            &mut path,
        );
    }
    new_graph
}

// Replaces a node's edges with their epsilon closure
fn replace_with_epsilon_closure<'a>(
    context: &ElimContext<'a>,
    current_node: NodeIndex,
    new_graph: &mut Graph<NsmEdgeData>,
    visit_set: &mut HashSet<NodeIndex>,
    path: &mut PathData<'a>,
) {
    visit_set.insert(current_node);
    let edges = context.graph.get_node(current_node).out_edges.clone();
    for edge_index in edges {
        let edge: &Edge<EpsNsmEdgeData> = context.graph.get_edge(edge_index);
        //Follow edge simulates the result of following the edge. This must be paired with a pop statement.
        if let Some(next_node) = path.follow_edge(&context.graph, edge_index) {
            //If we are in a transition that consumes input, put an edge into the epsilon eliminated graph consuming that input.
            if let EpsCharMatch::Match(char_match) = &edge.data.char_match {
                let new_actions = path.get_actions();
                new_graph.add_edge_lowest_priority(
                    context.start,
                    NsmEdgeData {
                        transition: NsmEdgeTransition::Consume(NsmConsumeEdge {
                            char_match: char_match.clone(),
                            target_node: next_node,
                        }),
                        actions: new_actions,
                    },
                );
            } else {
                //If the visit set contains the node, there was already a more preferred way to
                //get to this node and any nodes reachable from it. So, just return.
                //TODO if the edge is a call edge there may be some more work
                //needed to handle inductive cycles.
                if !visit_set.contains(&next_node) {
                    replace_with_epsilon_closure(context, next_node, new_graph, visit_set, path);
                }
            }
        } else {
            //In this branch, there was a return but no known caller. The process must stop here due to this.
            //Instead, DFA at runtime must be used.
            let new_actions = path.get_actions();
            new_graph.add_edge_lowest_priority(
                context.start,
                NsmEdgeData {
                    transition: NsmEdgeTransition::Return,
                    actions: new_actions,
                },
            );
        }
        path.pop();
    }
}

fn epsilon_no_actions(target: EdgeTarget) -> EpsNsmEdgeData {
    EpsNsmEdgeData {
        target,
        char_match: EpsCharMatch::Epsilon,
        actions: vec![],
    }
}
fn lower_nsm(
    graph: &mut Graph<EpsNsmEdgeData>,
    start_node: NodeIndex,
    name_table: &HashMap<String, NodeIndex>,
    expr: GrammarEx,
) -> Result<NodeIndex, LoweringError> {
    let end_node = lower_nsm_rec(graph, start_node, name_table, expr)?;
    graph.add_edge_lowest_priority(end_node, epsilon_no_actions(EdgeTarget::Return));
    Ok(end_node)
}
// Takes in an Epsilon graph, a start node, a name table from expression names to start nodes and an expresion to lower.
// Returns the end node of the expression lowered as an subgraph with end_node accepting.
fn lower_nsm_rec(
    graph: &mut Graph<EpsNsmEdgeData>,
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
                EpsNsmEdgeData {
                    target: EdgeTarget::NodeIndex(end_node),
                    char_match: EpsCharMatch::Match(CharClass::Char(c)),
                    actions: vec![],
                },
            );
            Ok(end_node)
        }
        GrammarEx::CharRange(range) => {
            let end_node = graph.create_node();
            graph.add_edge_lowest_priority(
                start_node,
                EpsNsmEdgeData {
                    target: EdgeTarget::NodeIndex(end_node),
                    char_match: EpsCharMatch::Match(CharClass::Range(range)),
                    actions: vec![],
                },
            );
            Ok(end_node)
        }
        GrammarEx::Seq(exprs) => {
            let mut end_node = start_node;
            for expr in exprs {
                end_node = lower_nsm_rec(graph, end_node, name_table, expr)?;
            }
            Ok(end_node)
        }
        GrammarEx::Star(expr) => {
            let end_node = lower_nsm_rec(graph, start_node, name_table, *expr)?;
            //For a star operator, looping back is always highest priority. Skipping it is lowest priority
            graph.add_edge_highest_priority(
                end_node,
                epsilon_no_actions(EdgeTarget::NodeIndex(start_node)),
            );
            graph.add_edge_lowest_priority(
                start_node,
                epsilon_no_actions(EdgeTarget::NodeIndex(end_node)),
            );
            Ok(end_node)
        }
        GrammarEx::Plus(expr) => {
            let end_node = lower_nsm_rec(graph, start_node, name_table, *expr)?;
            //For a plus operator, looping back is always highest priority. It can't be skipped.
            graph.add_edge_highest_priority(
                end_node,
                epsilon_no_actions(EdgeTarget::NodeIndex(start_node)),
            );
            Ok(end_node)
        }
        GrammarEx::Alt(exprs) => {
            //The code could get away with not creating a node if the
            //alt is non-empty, but that would make it more complicated.
            let end_node = graph.create_node();
            for expr in exprs {
                let end = lower_nsm_rec(graph, start_node, name_table, expr)?;
                graph.add_edge_lowest_priority(
                    end,
                    epsilon_no_actions(EdgeTarget::NodeIndex(end_node)),
                );
            }
            Ok(end_node)
        }
        GrammarEx::Optional(grammar_ex) => {
            let end_node = lower_nsm_rec(graph, start_node, name_table, *grammar_ex)?;
            //An option can be skipped, skipping has lowest priority over consuming input
            graph.add_edge_lowest_priority(
                start_node,
                epsilon_no_actions(EdgeTarget::NodeIndex(end_node)),
            );
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
                target_node: expr_node,
                name: expr,
                return_node,
            };
            graph.add_edge_lowest_priority(
                start_node,
                EpsNsmEdgeData {
                    target: EdgeTarget::Call(call_data),
                    char_match: EpsCharMatch::Epsilon,
                    actions: vec![Action::Assign("%RET".to_string(), var)],
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
                target_node: expr_node,
                name: expr,
                return_node,
            };
            graph.add_edge_lowest_priority(
                start_node,
                EpsNsmEdgeData {
                    target: EdgeTarget::Call(call_data),
                    char_match: EpsCharMatch::Epsilon,
                    actions: vec![],
                },
            );
            Ok(return_node)
        }
    }
}
