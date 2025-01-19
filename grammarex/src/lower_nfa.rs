use std::{
    cmp::min,
    collections::{HashMap, HashSet, VecDeque},
    fmt::Debug,
};

use thiserror::Error;

use crate::{
    nsm::{Graph, Remappable},
    types::GrammarEx,
};

#[derive(Error, Debug)]
pub enum LoweringError {
    #[error("Invalid variable assignment")]
    InvalidVariableAssignment,
    #[error("Couldn't locate expression `{0}`")]
    UnknownExpression(String),
}

#[derive(PartialEq, Eq, Clone, Debug, PartialOrd, Ord)]
enum CharClass {
    Char(char),
    RangeInclusive(char, char),
}
#[derive(PartialEq, Eq, Clone, Debug, PartialOrd, Ord)]
enum EpsCharMatch {
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

#[derive(Debug, Clone)]

struct EpsNsmEdgeData {
    transition: EpsNsmEdgeTransition,
    actions: Vec<Action>,
}

pub fn compile(
    machines: HashMap<String, GrammarEx>,
    start_machine: &String,
) -> Result<Graph<NsmEdgeData>, LoweringError> {
    let mut graph: Graph<EpsNsmEdgeData> = Graph::new(None);
    let machine_starts: HashMap<String, _> = machines
        .iter()
        .map(|(name, _)| ((name.clone()), graph.create_node()))
        .collect();
    let start_node = *machine_starts
        .get(start_machine)
        .ok_or_else(|| LoweringError::UnknownExpression(start_machine.clone()))?;
    graph.start_node = Some(start_node);
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
    let elim = eliminate_epsilon(&graph);
    let deduped = full_dedup(&elim);
    Ok(deduped)
}

//Removes all unused edges from the graph.
fn remove_unused_edges(graph: &Graph<NsmEdgeData>) -> Graph<NsmEdgeData> {
    let mut res: Graph<NsmEdgeData> = Graph::new(graph.start_node);
    let mut live_edges = vec![false; graph.edges.len()];
    for i in 0..graph.nodes.len() {
        for &edge_idx in &graph.get_node(i).out_edges {
            live_edges[edge_idx] = true;
        }
    }
    let mut remap_table: Vec<usize> = vec![0; graph.edges.len()];
    let mut new_edges = Vec::new();
    for i in 0..live_edges.len() {
        if live_edges[i] {
            remap_table[i] = new_edges.len();
            new_edges.push(graph.get_edge(i).clone());
        }
    }
    for node in &graph.nodes {
        let node_idx = res.create_node();
        res.get_node_mut(node_idx).out_edges =
            node.out_edges.iter().map(|x| remap_table[*x]).collect();
    }
    res.edges = new_edges;
    res
}

fn deduplicate_edges(graph: &Graph<NsmEdgeData>) -> Graph<NsmEdgeData> {
    let (remap_table, count) = sort_dedup_mapping(graph.edges.iter().map(|x| &*x));
    let mut new_edges = vec![None; count];
    for i in 0..remap_table.len() {
        new_edges[remap_table[i]] = Some(graph.edges[i].clone());
    }
    let new_edges = new_edges
        .into_iter()
        .map(|x| x.expect("Edge was mapped"))
        .collect();
    let mut res: Graph<NsmEdgeData> = Graph::new(graph.start_node);
    for node in &graph.nodes {
        let node_idx = res.create_node();
        let new_out_edges: VecDeque<_> = node.out_edges.iter().map(|x| remap_table[*x]).collect();
        res.get_node_mut(node_idx).out_edges = new_out_edges;
    }
    res.dedup_out_edges();
    res.edges = new_edges;
    res
}

//Performs a full deduplication of the graph.
//This deduplicates all identical edges and all nodes that have the same
//out edges. It then prunes unused edges.
fn full_dedup(graph: &Graph<NsmEdgeData>) -> Graph<NsmEdgeData> {
    let mut graph = deduplicate_edges(graph);
    loop {
        let (node_remap_table, count) = build_node_deduplicate_remap_table(&graph);
        if count == graph.nodes.len() {
            break;
        }
        graph = graph.remap_nodes(&node_remap_table, count);
        graph = deduplicate_edges(&graph);
    }
    remove_unused_edges(&graph)
}

impl Remappable for NsmEdgeData {
    fn remap(&self, remap: &Vec<Option<usize>>) -> Option<Self>
    where
        Self: Sized,
    {
        let transition = match &self.transition {
            NsmEdgeTransition::Return => NsmEdgeTransition::Return,
            NsmEdgeTransition::Call(x) => NsmEdgeTransition::Call(CallData {
                name: x.name.clone(),
                target_node: remap[x.target_node]?,
                return_node: remap[x.return_node]?,
            }),
            NsmEdgeTransition::Consume(NsmConsumeEdge {
                char_match,
                target_node,
            }) => NsmEdgeTransition::Consume(NsmConsumeEdge {
                char_match: char_match.clone(),
                target_node: remap[*target_node]?,
            }),
        };
        Some(NsmEdgeData {
            transition,
            actions: self.actions.clone(),
        })
    }
}

fn sort_dedup_mapping<'a, T: Eq + Ord + Debug + 'a>(
    input: impl Iterator<Item = &'a T>,
) -> (Vec<usize>, usize) {
    let mut enumerated: Vec<(usize, &T)> = input.into_iter().enumerate().collect();
    if enumerated.len() == 0 {
        return (vec![], 0);
    }
    enumerated.sort_by(|a, b| a.1.cmp(b.1));
    let mut remap_table = vec![0; enumerated.len()];
    let mut remap_counter = 0;
    //The 0th node is always remapped to itself.
    for i in 1..enumerated.len() {
        if Some(&enumerated[i].1) != enumerated.get(i - 1).map(|x| &x.1) {
            remap_counter += 1;
        }
        remap_table[enumerated[i].0] = remap_counter;
    }
    (remap_table, remap_counter + 1)
}

//This builds a remapping table for deduplicting nodes.
//Precondition: Edges must have been deduplicated first.
fn build_node_deduplicate_remap_table(graph: &Graph<NsmEdgeData>) -> (Vec<Option<usize>>, usize) {
    let dedup_item = graph.nodes.iter().map(|node| &node.out_edges);
    let (dedup_mapping, count) = sort_dedup_mapping(dedup_item);
    let dedup_mapping = dedup_mapping.into_iter().map(|x| Some(x)).collect();
    (dedup_mapping, count)
}

//Identifies the strongly connected components in the graph
//among consume edges.
//The function returns a table mapping nodes to their SCC and the count of SCCs.
//The result of this function is a topological sort of the SCCs.
fn identify_scc(graph: &Graph<NsmEdgeData>) -> (Vec<usize>, usize) {
    let n = graph.nodes.len();
    let mut index = 1;
    let mut on_stack = vec![false; n];
    let mut indexs = vec![0; n];
    let mut lowlinks = vec![0; n];
    let mut mapping_table = vec![0; n];
    let mut scc_count = 0;
    let mut stack = vec![];
    for i in 0..n {
        if indexs[i] == 0 {
            identify_scc_rec(
                graph,
                &mut index,
                &mut on_stack,
                &mut indexs,
                &mut lowlinks,
                &mut stack,
                &mut mapping_table,
                &mut scc_count,
                i,
            );
        }
    }
    (mapping_table, scc_count)
}

fn identify_scc_rec(
    graph: &Graph<NsmEdgeData>,
    index: &mut usize,
    on_stack: &mut Vec<bool>,
    indexs: &mut Vec<usize>,
    lowlinks: &mut Vec<usize>,
    stack: &mut Vec<usize>,
    mapping_table: &mut Vec<usize>,
    scc_count: &mut usize,
    node_index: usize,
) {
    indexs[node_index] = *index;
    lowlinks[node_index] = *index;
    *index += 1;
    stack.push(node_index);
    on_stack[node_index] = true;

    let node = graph.get_node(node_index);
    for edge_idx in &node.out_edges {
        let edge_data = &graph.get_edge(*edge_idx);
        if let NsmEdgeTransition::Consume(target) = &edge_data.transition {
            let target = target.target_node;
            if indexs[target] == 0 {
                identify_scc_rec(
                    graph,
                    index,
                    on_stack,
                    indexs,
                    lowlinks,
                    stack,
                    mapping_table,
                    scc_count,
                    target,
                );
                lowlinks[node_index] = min(lowlinks[node_index], lowlinks[target]);
            } else if on_stack[target] {
                lowlinks[node_index] = min(lowlinks[node_index], lowlinks[target]);
            }
        }
    }
    if lowlinks[node_index] == indexs[node_index] {
        while let Some(node) = stack.pop() {
            on_stack[node] = false;
            mapping_table[node] = *scc_count;
            if node == node_index {
                break;
            }
        }
        *scc_count += 1;
    }
}

//This discards all char match and action data for computing the reachability summary.
fn map_for_reachability_summary(in_graph: &Graph<NsmEdgeData>) -> Graph<NsmEdgeData> {
    let mut res = in_graph.clone();
    for i in 0..in_graph.edges.len() {
        let data = &mut res.get_edge_mut(i);
        data.actions = vec![];
        data.transition = match &data.transition {
            NsmEdgeTransition::Consume(nsm_consume_edge) => {
                NsmEdgeTransition::Consume(NsmConsumeEdge {
                    char_match: CharClass::Char('a'),
                    target_node: nsm_consume_edge.target_node,
                })
            }
            NsmEdgeTransition::Return => NsmEdgeTransition::Return,
            NsmEdgeTransition::Call(x) => NsmEdgeTransition::Call(x.clone()),
        }
    }
    res
}

pub fn liveness_to_remapping(live: &Vec<bool>) -> (Vec<Option<usize>>, usize) {
    let mut res: Vec<Option<usize>> = Vec::new();
    let mut live_count = 0;
    for i in 0..live.len() {
        if live[i] {
            res.push(Some(live_count));
            live_count += 1;
        } else {
            res.push(None);
        }
    }
    (res, live_count)
}

struct ElimContext<'a> {
    start: usize,
    graph: &'a Graph<EpsNsmEdgeData>,
}

//Takes in an epsilon graph and the accepting nodes. Returns an equivilent graph that has had epsilon elimination performed.
fn eliminate_epsilon(graph: &Graph<EpsNsmEdgeData>) -> Graph<NsmEdgeData> {
    let mut already_added_eps: HashSet<(usize, usize)> = HashSet::new();
    let mut old_size = 0;
    let mut new_graph: Graph<NsmEdgeData>;
    loop {
        new_graph= Graph::new(graph.start_node);
        for _ in 0..graph.nodes.len() {
            new_graph.create_node();
        }
        for i in 0..graph.nodes.len() {
            let mut visited_set: HashSet<usize> = HashSet::new();
            let mut path = Vec::new();
            let context = ElimContext {
                start: i,
                graph: &graph,
            };
            replace_epsilon_closure_local(&context, i, &mut new_graph, &mut visited_set, &mut path);
        }
        insert_summary_edges(&new_graph, &mut already_added_eps);
        if old_size == already_added_eps.len() {
            break;
        }
        old_size = already_added_eps.len();
    }
    new_graph
}

fn insert_summary_edges(
    graph: &Graph<NsmEdgeData>,
    already_added_eps: &mut HashSet<(usize, usize)>,
) {
    let mut new_graph: Graph<EpsNsmEdgeData> = Graph::new(graph.start_node);
    for _ in 0..graph.nodes.len() {
        new_graph.create_node();
    }
    for i in 0..graph.nodes.len() {
        let edges = &graph.get_node(i).out_edges;
        for j in 0..edges.len() {
            let edge_index = edges[j];
            let call_edge = graph.get_edge(edge_index);
            let eps_transition = match call_edge.transition.clone() {
                NsmEdgeTransition::Consume(nsm_consume_edge) => EpsNsmEdgeTransition::Move(
                    EpsCharMatch::Match(nsm_consume_edge.char_match),
                    nsm_consume_edge.target_node,
                ),
                NsmEdgeTransition::Call(call_data) => EpsNsmEdgeTransition::Call(call_data),
                NsmEdgeTransition::Return => EpsNsmEdgeTransition::Return,
            };
            let eps_copy: EpsNsmEdgeData = EpsNsmEdgeData {
                transition: eps_transition,
                actions: call_edge.actions.clone(),
            };
            new_graph.add_edge_lowest_priority(i, eps_copy);
            if let NsmEdgeTransition::Call(call_data) = &call_edge.transition {
                if already_added_eps.contains(&(i, call_data.target_node)) {
                    continue;
                }
                let target = call_data.target_node;
                for edge in &graph.get_node(target).out_edges {
                    //The call directly follows a return, so add an epsilon edge here.
                    let return_edge = graph.get_edge(*edge);
                    if let NsmEdgeTransition::Return = &return_edge.transition {
                        already_added_eps.insert((i, call_data.target_node));
                        let mut new_actions = Vec::new();
                        new_actions.append(&mut call_edge.actions.clone());
                        new_actions.append(&mut return_edge.actions.clone());
                        new_graph.add_edge_lowest_priority(
                            i,
                            EpsNsmEdgeData {
                                transition: EpsNsmEdgeTransition::Move(
                                    EpsCharMatch::Epsilon,
                                    call_data.return_node
                                ),
                                actions: new_actions,
                            },
                        );
                    }
                }
            }
        }
    }
}

// Replaces a node's edges with their epsilon closure
fn replace_epsilon_closure_local<'a>(
    context: &ElimContext<'a>,
    current_node: usize,
    new_graph: &mut Graph<NsmEdgeData>,
    visit_set: &mut HashSet<usize>,
    path: &mut Vec<usize>,
) {
    visit_set.insert(current_node);
    let edges = &context.graph.get_node(current_node).out_edges;
    for &edge_index in edges {
        let edge: &EpsNsmEdgeData = context.graph.get_edge(edge_index);
        path.push(edge_index);
        match &edge.transition {
            EpsNsmEdgeTransition::Call(call_data) => {
                new_graph.add_edge_lowest_priority(
                    context.start,
                    NsmEdgeData {
                        transition: NsmEdgeTransition::Call(call_data.clone()),
                        actions: total_actions(context.graph, &path),
                    },
                );
            }
            EpsNsmEdgeTransition::Move(char_match, next_node) => {
                if let EpsCharMatch::Match(char_match) = &char_match {
                    let new_actions = total_actions(context.graph, &path);
                    new_graph.add_edge_lowest_priority(
                        context.start,
                        NsmEdgeData {
                            transition: NsmEdgeTransition::Consume(NsmConsumeEdge {
                                char_match: char_match.clone(),
                                target_node: *next_node,
                            }),
                            actions: new_actions,
                        },
                    );
                } else {
                    //If the visit set contains the node, there was already a more preferred way to
                    //get to this node and any nodes reachable from it. So do nothing
                    if !visit_set.contains(&next_node) {
                        //In this case, there is an edge that consumes epsilon. So recursively explore more of the graph.
                        replace_epsilon_closure_local(
                            context, *next_node, new_graph, visit_set, path,
                        );
                    }
                }
            }
            EpsNsmEdgeTransition::Return => {
                new_graph.add_edge_lowest_priority(
                    context.start,
                    NsmEdgeData {
                        transition: NsmEdgeTransition::Return,
                        actions: total_actions(context.graph, &path),
                    },
                );
            }
        }
        path.pop();
    }
}

fn total_actions(graph: &Graph<EpsNsmEdgeData>, edges_taken: &Vec<usize>) -> Vec<Action> {
    let mut res = Vec::new();
    for edge in edges_taken {
        for action in &graph.get_edge(*edge).actions {
            res.push(action.clone());
        }
    }
    res
}

fn epsilon_no_actions(target: usize) -> EpsNsmEdgeData {
    EpsNsmEdgeData {
        transition: EpsNsmEdgeTransition::Move(EpsCharMatch::Epsilon, target),
        actions: vec![],
    }
}
fn lower_nsm(
    graph: &mut Graph<EpsNsmEdgeData>,
    start_node: usize,
    name_table: &HashMap<String, usize>,
    expr: GrammarEx,
) -> Result<usize, LoweringError> {
    let end_node = lower_nsm_rec(graph, start_node, name_table, expr)?;
    graph.add_edge_lowest_priority(
        end_node,
        EpsNsmEdgeData {
            transition: EpsNsmEdgeTransition::Return,
            actions: vec![],
        },
    );
    Ok(end_node)
}
// Takes in an Epsilon graph, a start node, a name table from expression names to start nodes and an expresion to lower.
// Returns the end node of the expression lowered as an subgraph with end_node accepting.
fn lower_nsm_rec(
    graph: &mut Graph<EpsNsmEdgeData>,
    start_node: usize,
    name_table: &HashMap<String, usize>,
    expr: GrammarEx,
) -> Result<usize, LoweringError> {
    match expr {
        GrammarEx::Epsilon => Ok(start_node),
        GrammarEx::Char(c) => {
            let end_node = graph.create_node();
            graph.add_edge_lowest_priority(
                start_node,
                EpsNsmEdgeData {
                    transition: EpsNsmEdgeTransition::Move(
                        EpsCharMatch::Match(CharClass::Char(c)),
                        end_node,
                    ),
                    actions: vec![],
                },
            );
            Ok(end_node)
        }
        GrammarEx::CharRangeInclusive(m, n) => {
            let end_node = graph.create_node();
            graph.add_edge_lowest_priority(
                start_node,
                EpsNsmEdgeData {
                    transition: EpsNsmEdgeTransition::Move(
                        EpsCharMatch::Match(CharClass::RangeInclusive(m, n)),
                        end_node,
                    ),
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
            //For a star operator looping back is always highest priority. Skipping it is lowest priority
            graph.add_edge_highest_priority(end_node, epsilon_no_actions(start_node));
            graph.add_edge_lowest_priority(start_node, epsilon_no_actions(end_node));
            Ok(end_node)
        }
        GrammarEx::Plus(expr) => {
            let end_node = lower_nsm_rec(graph, start_node, name_table, *expr)?;
            //For a plus operator looping back is always highest priority. It can't be skipped.
            graph.add_edge_highest_priority(end_node, epsilon_no_actions(start_node));
            Ok(end_node)
        }
        GrammarEx::Alt(exprs) => {
            //The code could get away with not creating a node if the
            //alt is non-empty, but that would make it more complicated.
            let end_node = graph.create_node();
            for expr in exprs {
                let end = lower_nsm_rec(graph, start_node, name_table, expr)?;
                graph.add_edge_lowest_priority(end, epsilon_no_actions(end_node));
            }
            Ok(end_node)
        }
        GrammarEx::Optional(grammar_ex) => {
            let end_node = lower_nsm_rec(graph, start_node, name_table, *grammar_ex)?;
            //An option can be skipped, skipping has lowest priority over consuming input
            graph.add_edge_lowest_priority(start_node, epsilon_no_actions(end_node));
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
                    transition: EpsNsmEdgeTransition::Call(call_data),
                    actions: vec![],
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
                    transition: EpsNsmEdgeTransition::Call(call_data),
                    actions: vec![],
                },
            );
            Ok(return_node)
        }
    }
}

fn pretty_print_graph<T: Clone + Debug>(graph: &Graph<T>) {
    let mut to_print = vec![];
    for i in 0..graph.nodes.len() {
        let node = graph.get_node(i);
        let mut edges = vec![];
        for edge in &node.out_edges {
            edges.push(graph.get_edge(*edge).clone());
        }
        to_print.push((i, edges));
    }
    dbg!(graph.start_node, to_print);
}
#[cfg(test)]
mod tests {
    use crate::parse_grammarex;

    use super::*;

    #[test]
    fn test_compile_machine() {
        let expr = parse_grammarex(&mut "[cba]").unwrap();
        let start = "start".to_string();
        let mut machines = HashMap::new();
        machines.insert(start.clone(), expr);
        let machine = compile(machines, &start).unwrap();
        pretty_print_graph(&machine);
    }

    #[test]
    fn test_compile_machine_multi() {
        let expr_one = parse_grammarex(&mut r#" abc = second "#).unwrap();
        let expr_two = parse_grammarex(&mut r#"[abc]"#).unwrap();
        let start = "start".to_string();
        let second = "second".to_string();
        let mut machines = HashMap::new();
        machines.insert(start.clone(), expr_one);
        machines.insert(second.clone(), expr_two);
        let machine = compile(machines, &start).unwrap();
        pretty_print_graph(&machine);
    }

    #[test]
    fn test_compile_rec() {
        let expr_one = parse_grammarex(&mut r#" "a" | \( start \) "#).unwrap();
        let start = "start".to_string();
        let mut machines = HashMap::new();
        machines.insert(start.clone(), expr_one);
        let machine = compile(machines, &start).unwrap();
        pretty_print_graph(&machine);
    }

    #[test]
    fn test_compile_rec_two() {
        let expr_one = parse_grammarex(&mut r#"   "a" "b"  | "c" start  "#).unwrap();
        let start = "start".to_string();
        let mut machines = HashMap::new();
        machines.insert(start.clone(), expr_one);
        let machine = compile(machines, &start).unwrap();
        pretty_print_graph(&machine);
    }


    #[test]
    fn test_compile_call_epsilon() {
        let expr_one = parse_grammarex(&mut r#" abc = second "#).unwrap();
        let expr_two = parse_grammarex(&mut r#" "x"? "#).unwrap();
        let start = "start".to_string();
        let second = "second".to_string();
        let mut machines = HashMap::new();
        machines.insert(start.clone(), expr_one);
        machines.insert(second.clone(), expr_two);
        let machine = compile(machines, &start).unwrap();
        pretty_print_graph(&machine);
    }

    #[test]
    fn test_compile_left_recursive_loop() {
        let expr_one = parse_grammarex(&mut r#"  start? "a"   "#).unwrap();
        let start = "start".to_string();
        let mut machines = HashMap::new();
        machines.insert(start.clone(), expr_one);
        let machine = compile(machines, &start).unwrap();
        pretty_print_graph(&machine);
    }
}
