use std::{
    cmp::min,
    collections::{HashMap, HashSet}, fmt::Debug,
};

use thiserror::Error;

use crate::{nsm::{Edge, Graph, Remappable}, types::GrammarEx};

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
#[derive(PartialEq, Eq, Clone, Debug)]
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
    //An edge of the NSM that returns to an unknown place and therefore
    //doesn't consume input.
    Return,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct  NsmEdgeData {
    pub transition: NsmEdgeTransition,
    pub actions: Vec<Action>,
    //A list of nodes to push on the return stack when transitioning this edge.
    //The VM will use separate stacks for return addresses and for data.
    pub push_return_stack: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Action {
    Assign(String, String),
}
#[derive(Clone, Debug)]
pub struct CallData {
    name: String,
    target_node: usize,
    return_node: usize,
}

#[derive(Clone, Debug)]
pub enum EdgeTarget {
    Call(CallData),
    usize(usize),
    Return,
}
#[derive(Debug)]

struct EpsNsmEdgeData {
    target: EdgeTarget,
    char_match: EpsCharMatch,
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
    //pretty_print_graph(&deduped);
    //let pruned = prune_dead_nodes(&deduped, start_node);
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
        res.get_node_mut(node_idx).out_edges = node
            .out_edges
            .iter()
            .map(|x| remap_table[*x])
            .collect();
    }
    res.edges = new_edges;
    res
}

fn deduplicate_edges(graph: &Graph<NsmEdgeData>) -> Graph<NsmEdgeData> {
    let (remap_table, count) = sort_dedup_mapping(graph.edges.iter().map(|x| &x.data));
    let mut new_edges = vec![None; count];
    for i in 0..remap_table.len() {
        new_edges[remap_table[i]] = Some(graph.edges[i].clone());
    }
    let new_edges = new_edges.into_iter().map(|x| x.expect("Edge was mapped")).collect();
    let mut res: Graph<NsmEdgeData> = Graph::new(graph.start_node);
    for node in &graph.nodes {
        let node_idx = res.create_node();
        let mut new_out_edges: Vec<_> = node
            .out_edges
            .iter()
            .map(|x| remap_table[*x])
            .collect();
        new_out_edges.sort();
        new_out_edges.dedup();
        res.get_node_mut(node_idx).out_edges = new_out_edges.into();
    }
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
    fn remap(&self, remap: &Vec<Option<usize>>) -> Option<Self> where Self: Sized {
        let transition = match &self.transition {
            NsmEdgeTransition::Return => NsmEdgeTransition::Return,
            NsmEdgeTransition::Consume(NsmConsumeEdge {
                char_match,
                target_node,
            }) => NsmEdgeTransition::Consume(NsmConsumeEdge {
                char_match: char_match.clone(),
                target_node: remap[*target_node]?,
            }),
        };
        let mut push_return_stack = Vec::new(); 
        for item in &self.push_return_stack {
            push_return_stack.push(remap[*item]?);
        }
        Some(NsmEdgeData {
                transition,
                actions: self.actions.clone(),
                push_return_stack,
        })
    }
}

fn sort_dedup_mapping<'a, T: Eq + Ord + Debug + 'a>(input: impl Iterator<Item = &'a T>) -> (Vec<usize>, usize){
    let mut enumerated: Vec<(usize, &T)> = input.into_iter().enumerate().collect();
    if enumerated.len() == 0 {
        return (vec![], 0)
    }
    enumerated.sort_by(|a,b| a.1.cmp(b.1));
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
    let dedup_item = graph
        .nodes
        .iter()
        .map(|node| &node.out_edges);
    let (dedup_mapping, count) =sort_dedup_mapping(dedup_item);
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
        let edge_data = &graph.get_edge(*edge_idx).data;
        if let NsmEdgeTransition::Consume(target) = &edge_data.transition {
            let target = target.target_node;
            if edge_data.push_return_stack.len() == 0 {
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
        let data = &mut res.get_edge_mut(i).data;
        data.actions = vec![];
        data.transition = match &data.transition {
            NsmEdgeTransition::Consume(nsm_consume_edge) => {
                NsmEdgeTransition::Consume(NsmConsumeEdge {
                    char_match: CharClass::Char('a'),
                    target_node: nsm_consume_edge.target_node,
                })
            },
            NsmEdgeTransition::Return =>  NsmEdgeTransition::Return,
        }
    }
    res
}

//Returns a summary of which nodes are reachable from which other nodes starting
//and ending with an empty stack.
fn reachability_summary(in_graph: &Graph<NsmEdgeData>) -> (Graph<NsmEdgeData>, Vec<usize>) {
    //This overall mapping holds what nodes are mapped to in the resulting graph.
    let mut overall_mapping: Vec<usize> = (0..in_graph.nodes.len()).collect();
    let mut contracted = map_for_reachability_summary(in_graph);
    contracted = deduplicate_edges(&contracted);
    loop {
        let (scc, num_scc) = identify_scc(&contracted);
        for i in 0..scc.len() {
            overall_mapping[i] = scc[overall_mapping[i]];
        }
        let remap_table = scc.iter().map(|x| Some(*x)).collect();
        contracted = contracted.remap_nodes(&remap_table, num_scc);
        contracted = deduplicate_edges(&contracted);
        let can_reach_return = can_reach_return(&contracted);
        let mut made_progress = false;
        for i in 0..contracted.nodes.len() {
            for &edge in &contracted.get_node(i).out_edges.clone() {
                let edge_data = &mut contracted.get_edge_mut(edge).data;
                if let NsmEdgeTransition::Consume(nsm_consume_edge) = &edge_data.transition.clone() {
                    while let Some(node) = edge_data.push_return_stack.pop() {
                        if !can_reach_return[node] {
                            edge_data.push_return_stack.push(node);
                            break;
                        }
                        //made_progress = true;
                        edge_data.transition = NsmEdgeTransition::Consume(NsmConsumeEdge {
                            char_match: nsm_consume_edge.char_match.clone(),
                            target_node: node,
                        });
                    }
                }
            }
        }
        if !made_progress {
            break;
        }
    }
    (contracted, overall_mapping)
}

fn can_reach_return(contracted: &Graph<NsmEdgeData>) -> Vec<bool> { 
    let mut can_reach_return = vec![false; contracted.nodes.len()];
    //First, mark the nodes that can reach a return. Since identify_scc returns a topologically sorted
    //graph, all thats needed is a single pass.
    for i in 0..contracted.nodes.len() {
        for &edge in &contracted.get_node(i).out_edges {
            let edge_data = &contracted.get_edge(edge).data;
            match &edge_data.transition {
                NsmEdgeTransition::Consume(nsm_consume_edge) => {
                    if edge_data.push_return_stack.len() == 0 {
                        can_reach_return[i] =
                            can_reach_return[i] || can_reach_return[nsm_consume_edge.target_node];
                    }
                }
                NsmEdgeTransition::Return => {
                    can_reach_return[i] = true;
                }
            }
        }
    }
    can_reach_return
}

fn identify_live_nodes(in_graph: &Graph<NsmEdgeData>, start: usize) -> Vec<bool> {
    let (graph, mapping) = reachability_summary(in_graph);
    let num_nodes = graph.nodes.len();
    let mut to_process = vec![mapping[start]];
    //If a node can't be reached from the start it is dead.
    let mut processed = vec![false; num_nodes];
    while let Some(node) = to_process.pop() {
        if processed[node] {
            continue;
        }
        processed[node] = true;
        for edge in &graph.get_node(node).out_edges {
            let data = &graph.get_edge(*edge).data;
            match &data.transition {
                NsmEdgeTransition::Consume(nsm_consume_edge) => {to_process.push(nsm_consume_edge.target_node);},
                //The reachability summary lets us ignore return edges.
                NsmEdgeTransition::Return => {},
            }
        }
    }
    //If a node can't reach a return, it is dead.
    let can_reach_return = can_reach_return(&graph);
    let mut live = vec![false; num_nodes];
    for i in 0..num_nodes {
        live[i] = can_reach_return[i] && processed[i];
    }
    let mut in_live = vec![false; in_graph.nodes.len()];
    for i in 0..in_graph.nodes.len() {
        in_live[i] = live[mapping[i]];
    }
    in_live
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

fn prune_dead_nodes(in_graph: &Graph<NsmEdgeData>, start: usize) -> Graph<NsmEdgeData> {
    let node_liveness = identify_live_nodes(in_graph, start);
    let (remapping, node_count) = liveness_to_remapping(&node_liveness);
    let res = in_graph.remap_nodes(&remapping, node_count);
    let res = remove_unused_edges(&res);
    res
}
struct ElimContext<'a> {
    start: usize,
    graph: &'a Graph<EpsNsmEdgeData>,
}

#[derive(Clone)]
struct ReturnRef {
    node: usize,
    prior: usize,
}

struct RewindableStack {
    // Each node in the list points back to the prior call stack, effectively
    // forming a linked list.
    return_stacks: Vec<Option<ReturnRef>>,
}

impl RewindableStack {
    fn new() -> Self {
        RewindableStack {
            //Start with an empty stack at index 0. This simplifies later code. This must not be popped.
            return_stacks: vec![None],
        }
    }

    fn follow_edge(
        &mut self,
        graph: &Graph<EpsNsmEdgeData>,
        edge_index: usize,
    ) -> Option<usize> {
        let edge = graph.get_edge(edge_index);
        match &edge.data.target {
            // In this case, the edge's target is a node index. So
            // push the actions.
            EdgeTarget::usize(target) => {
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
                    //Push a none here to match the future pop.
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

    fn len(&self) -> usize {
        return self.return_stacks.len() - 1;
    }
    //Reverses the latest actions.
    fn pop(&mut self) {
        //The stack at index 0 is a guard and must not be popped.
        assert!(self.return_stacks.len() > 1);
        self.return_stacks.pop();
    }

    fn get_return_stack(&self) -> Vec<usize> {
        let mut res = Vec::new();
        let mut idx = self.return_stacks.len() - 1;
        while let Some(ret) = &self.return_stacks[idx] {
            res.push(ret.node);
            idx = ret.prior;
        }
        res.reverse();
        res
    }
}
struct PathData<'a> {
    actions: Vec<&'a Vec<Action>>,
    stack: RewindableStack,
}

impl<'a> PathData<'a> {
    fn new() -> Self {
        PathData {
            actions: vec![],
            stack: RewindableStack::new(),
        }
    }

    // Simulates following an edge. If this method returns None, then there was a return action with
    // no known caller. Returns the destination node.
    fn follow_edge(
        &mut self,
        graph: &'a Graph<EpsNsmEdgeData>,
        edge_index: usize,
    ) -> Option<usize> {
        assert!(self.actions.len() == self.stack.len());
        let edge = graph.get_edge(edge_index);
        self.actions.push(&edge.data.actions);
        self.stack.follow_edge(graph, edge_index)
    }
    //Reverses the latest actions.
    fn pop(&mut self) {
        self.actions.pop();
        self.stack.pop();
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

    fn get_return_stack(&self) -> Vec<usize> {
        self.stack.get_return_stack()
    }
}
//Takes in an epsilon graph and the accepting nodes. Returns an equivilent graph that has had epsilon elimination performed.
fn eliminate_epsilon(graph: &Graph<EpsNsmEdgeData>) -> Graph<NsmEdgeData> {
    let mut new_graph: Graph<NsmEdgeData> = Graph::new(graph.start_node);
    for _ in 0..graph.nodes.len() {
        new_graph.create_node();
    }
    for i in 0..graph.nodes.len() {
        let mut visited_set: HashSet<usize> = HashSet::new();
        let mut path = PathData::new();
        let context = ElimContext {
            start: i,
            graph: &graph,
        };
        replace_with_epsilon_closure(
            &context,
            i,
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
    current_node: usize,
    new_graph: &mut Graph<NsmEdgeData>,
    visit_set: &mut HashSet<usize>,
    path: &mut PathData<'a>,
) {
    visit_set.insert(current_node);
    let edges = &context.graph.get_node(current_node).out_edges;
    for &edge_index in edges {
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
                        push_return_stack: path.get_return_stack(),
                    },
                );
            } else {
                //If the visit set contains the node, there was already a more preferred way to
                //get to this node and any nodes reachable from it. So do nothing
                //TODO if the edge is a call edge there may be some more work
                //needed to handle inductive cycles, since the node can return to itself.
                if !visit_set.contains(&next_node) {
                    //In this case, there is an edge that consumes epsilon. So recursively explore more of the graph.
                    replace_with_epsilon_closure(context, next_node, new_graph, visit_set, path);
                }
            }
        } else {
            //In this branch, there was a return but no known caller. The process must stop here due to this.
            //Instead, Depth First Search at runtime must be used.
            let new_actions = path.get_actions();
            //Push return stack is  because if it wasn't, there would be a known node to return to.
            new_graph.add_edge_lowest_priority(
                context.start,
                NsmEdgeData {
                    transition: NsmEdgeTransition::Return,
                    actions: new_actions,
                    push_return_stack: Vec::new(),
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
    start_node: usize,
    name_table: &HashMap<String, usize>,
    expr: GrammarEx,
) -> Result<usize, LoweringError> {
    let end_node = lower_nsm_rec(graph, start_node, name_table, expr)?;
    graph.add_edge_lowest_priority(end_node, epsilon_no_actions(EdgeTarget::Return));
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
                    target: EdgeTarget::usize(end_node),
                    char_match: EpsCharMatch::Match(CharClass::Char(c)),
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
                    target: EdgeTarget::usize(end_node),
                    char_match: EpsCharMatch::Match(CharClass::RangeInclusive(m, n)),
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
            graph.add_edge_highest_priority(
                end_node,
                epsilon_no_actions(EdgeTarget::usize(start_node)),
            );
            graph.add_edge_lowest_priority(
                start_node,
                epsilon_no_actions(EdgeTarget::usize(end_node)),
            );
            Ok(end_node)
        }
        GrammarEx::Plus(expr) => {
            let end_node = lower_nsm_rec(graph, start_node, name_table, *expr)?;
            //For a plus operator looping back is always highest priority. It can't be skipped.
            graph.add_edge_highest_priority(
                end_node,
                epsilon_no_actions(EdgeTarget::usize(start_node)),
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
                    epsilon_no_actions(EdgeTarget::usize(end_node)),
                );
            }
            Ok(end_node)
        }
        GrammarEx::Optional(grammar_ex) => {
            let end_node = lower_nsm_rec(graph, start_node, name_table, *grammar_ex)?;
            //An option can be skipped, skipping has lowest priority over consuming input
            graph.add_edge_lowest_priority(
                start_node,
                epsilon_no_actions(EdgeTarget::usize(end_node)),
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
                    target: EdgeTarget::Call(call_data),
                    char_match: EpsCharMatch::Epsilon,
                    actions: vec![],
                },
            );
            Ok(return_node)
        }
    }
}

fn pretty_print_graph(graph: &Graph<NsmEdgeData>) {
    let mut to_print = vec![]; 
    for i in 0..graph.nodes.len() {
        let node = graph.get_node(i);
        let mut edges = vec![];
        for edge in &node.out_edges {
            edges.push(graph.get_edge(*edge).data.clone());
        }
        to_print.push((i,edges));
    }
    dbg!(to_print);
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
}
