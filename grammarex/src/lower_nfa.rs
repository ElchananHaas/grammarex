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

#[derive(PartialEq, Eq, Clone)]
enum CharClass {
    Char(char),
    Range(RangeInclusive<char>),
}
#[derive(PartialEq, Eq, Clone)]
enum EpsCharMatch {
    Epsilon,
    Match(CharClass),
}
#[derive(Clone)]
struct EpsNsmAction {
    char_match: EpsCharMatch,
    actions: Vec<Action>,
}
#[derive(Clone)]
pub struct NsmAction {
    char_match: CharClass,
    actions: Vec<Action>,
}
#[derive(Default)]
pub struct NodeData {
    accepts: bool,
}
#[derive(Clone)]
enum Action {
    Call(CallData),
    CallAssign(CallData, String),
}
#[derive(Clone)]
pub struct CallData {
    name: String,
    return_node: NodeIndex,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeIndex(usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIndex(usize);

#[derive(Clone)]
pub enum EdgeTarget{
    NodeIndex(NodeIndex),
    Return
}
struct Node {
    //Out edges are sorted by priority for taking them.
    out_edges: VecDeque<EdgeIndex>,
}
#[derive(Clone)]
struct Edge<EdgeData, EdgeTarget> {
    end: EdgeTarget,
    data: EdgeData,
}

pub struct NsmInstructions {
    pub start_node: NodeIndex,
    pub graph: Graph<NsmAction, EdgeTarget>,
}

pub fn compile(
    machines: HashMap<String, GrammarEx>,
    start_machine: &String,
) -> Result<NsmInstructions, LoweringError> {
    let mut graph: Graph<EpsNsmAction, EdgeTarget> = Graph {
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
    let elim = eliminate_epsilon(
        &mut graph,
    );
    Ok(NsmInstructions {
        start_node: *machine_starts.get(start_machine).ok_or_else(|| LoweringError::UnknownExpression(start_machine.clone()))?,
        graph: elim,
    })
}
// Invariants:
// For each edge, it is in the node it starts at's out_edges and no other out_edges
// For each node, an edge is in its out_edges iff it starts at the node.
pub struct Graph<EdgeData, EdgeTarget> {
    nodes: Vec<Node>,
    edges: Vec<Edge<EdgeData, EdgeTarget>>,
}

impl<EdgeData, EdgeTarget> Graph<EdgeData, EdgeTarget> {
    fn create_node(&mut self) -> NodeIndex {
        self.nodes.push(Node {
            out_edges: VecDeque::new(),
        });
        NodeIndex(self.nodes.len() - 1)
    }
    fn get_node(&self, idx: NodeIndex) -> &Node {
        &self.nodes[idx.0]
    }
    fn get_edge(&self, idx: EdgeIndex) -> &Edge<EdgeData, EdgeTarget> {
        &self.edges[idx.0]
    }
    fn add_edge_lowest_priority(&mut self, start: NodeIndex, end: EdgeTarget, data: EdgeData) {
        let idx = self.edges.len();
        self.edges.push(Edge { end, data });
        self.nodes[start.0].out_edges.push_back(EdgeIndex(idx));
    }

    fn add_edge_highest_priority(&mut self, start: NodeIndex, end: EdgeTarget, data: EdgeData) {
        let idx = self.edges.len();
        self.edges.push(Edge { end, data });
        self.nodes[start.0].out_edges.push_front(EdgeIndex(idx));
    }
}

struct ElimContext<'a> {
    start: NodeIndex,
    graph: &'a Graph<EpsNsmAction, EdgeTarget>,
}

struct PathData {

}
//Takes in an epsilon graph and the accepting nodes. Returns an equivilent graph that has had epsilon elimination performed.
fn eliminate_epsilon(
    graph: &mut Graph<EpsNsmAction, EdgeTarget>,
) -> Graph<NsmAction, EdgeTarget> {
    let mut new_graph: Graph<NsmAction, EdgeTarget> = Graph {
        nodes: Vec::new(),
        edges: Vec::new(),
    };
    for _ in 0..graph.nodes.len() {
        new_graph.create_node();
    }
    for i in 0..graph.nodes.len() {
        let mut visited_set: HashSet<NodeIndex> = HashSet::new();
        let mut path = Vec::new();
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
fn replace_with_epsilon_closure(
    context: &ElimContext,
    current_node: NodeIndex,
    new_graph: &mut Graph<NsmAction, EdgeTarget>,
    visit_set: &mut HashSet<NodeIndex>,
    path: &mut Vec<EdgeIndex>,
) {
    visit_set.insert(current_node);
    let edges = context.graph.get_node(current_node).out_edges.clone();
    for edge_index in edges {
        let edge = context.graph.get_edge(edge_index);
        if let EpsCharMatch::Match(char_match) = &edge.data.char_match {
            //Make sure to include the last edge's actions too.
            path.push(edge_index);
            let mut new_actions = Vec::new();
            for &path_edge_index in &*path {
                for action in &context.graph.get_edge(path_edge_index).data.actions {
                    new_actions.push(action.clone());
                }
            }
            new_graph.add_edge_lowest_priority(
                context.start,
                edge.end.clone(),
                NsmAction {
                    char_match: char_match.clone(),
                    actions: new_actions,
                },
            );
            path.pop();
        } else {
            match edge.end {
                EdgeTarget::NodeIndex(end) => {
                    //If the visit set contains the node, there was already a more preferred way to
                    //get to this node and any nodes reachable from it. So, just return.
                    //TODO if the edge is a call edge there may be some more work
                    //needed to handle inductive cycles.
                    if visit_set.contains(&end) {
                        continue;
                    }
                    path.push(edge_index);
                    replace_with_epsilon_closure(context, end, new_graph, visit_set, path);
                    path.pop();
                },
                EdgeTarget::Return => {
                    //In this case, the node accepts. This lets us return to the caller and 
                    //continue. The edges to be returned to will be substituted in at run time.
            
                    //There is an extra complication if there was a previous call operation in 
                    //the path. In this case, the return node must be precomputed in order
                    //to compute the entire epsilon closure. If this isn't done then some epsilon 
                    //transitions may be missed, leading to infinite loops. 
                    //In the case where there is no previous call operation, then the return
                    //must be computed at run time. This, outside of left recursive loops, will shorten
                    //the stack and lead to eventual termination.
                    let mut new_actions = Vec::new();
                    path.push(edge_index);
                    for &path_edge_index in &*path {
                        for action in context.graph.get_edge(path_edge_index).data.actions.iter().rev() {
                            new_actions.push(action.clone());
                        }
                    }
                    path.pop();
                },
            }
        }
    }
}

fn epsilon_no_action() -> EpsNsmAction {
    EpsNsmAction {
        char_match: EpsCharMatch::Epsilon,
        actions: vec![],
    }
}

fn lower_nsm(
    graph: &mut Graph<EpsNsmAction, EdgeTarget>,
    start_node: NodeIndex,
    name_table: &HashMap<String, NodeIndex>,
    expr: GrammarEx,
) -> Result<NodeIndex, LoweringError> {
    let end_node = lower_nsm_rec(graph, start_node, name_table, expr)?;
    graph.add_edge_lowest_priority(end_node, EdgeTarget::Return, epsilon_no_action());
    Ok(end_node)
}
// Takes in an Epsilon graph, a start node, a name table from expression names to start nodes and an expresion to lower.
// Returns the end node of the expression lowered as an subgraph with end_node accepting.
fn lower_nsm_rec(
    graph: &mut Graph<EpsNsmAction, EdgeTarget>,
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
                EdgeTarget::NodeIndex(end_node),
                EpsNsmAction {
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
                EdgeTarget::NodeIndex(end_node),
                EpsNsmAction {
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
            graph.add_edge_highest_priority(end_node, EdgeTarget::NodeIndex(start_node), epsilon_no_action());
            graph.add_edge_lowest_priority(start_node,EdgeTarget::NodeIndex(end_node), epsilon_no_action());
            Ok(end_node)
        }
        GrammarEx::Plus(expr) => {
            let end_node = lower_nsm_rec(graph, start_node, name_table, *expr)?;
            //For a plus operator, looping back is always highest priority. It can't be skipped.
            graph.add_edge_highest_priority(end_node, EdgeTarget::NodeIndex(start_node), epsilon_no_action());
            Ok(end_node)
        }
        GrammarEx::Alt(exprs) => {
            //The code could get away with not creating a node if the
            //alt is non-empty, but that would make it more complicated.
            let end_node = graph.create_node();
            for expr in exprs {
                let end = lower_nsm_rec(graph, start_node, name_table, expr)?;
                graph.add_edge_lowest_priority(end, EdgeTarget::NodeIndex(end_node), epsilon_no_action());
            }
            Ok(end_node)
        }
        GrammarEx::Optional(grammar_ex) => {
            let end_node = lower_nsm_rec(graph, start_node, name_table, *grammar_ex)?;
            //An option can be skipped, skipping has lowest priority over consuming input
            graph.add_edge_lowest_priority(start_node, EdgeTarget::NodeIndex(end_node), epsilon_no_action());
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
                EdgeTarget::NodeIndex(expr_node),
                EpsNsmAction {
                    char_match: EpsCharMatch::Epsilon,
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
                EdgeTarget::NodeIndex(expr_node),
                EpsNsmAction {
                    char_match: EpsCharMatch::Epsilon,
                    actions: vec![Action::Call(call_data)],
                },
            );
            Ok(return_node)
        }
    }
}
