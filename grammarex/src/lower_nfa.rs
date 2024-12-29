use std::{
    cmp::min,
    collections::{HashMap, HashSet, VecDeque},
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

#[derive(PartialEq, Eq, Clone, Debug, PartialOrd, Ord)]
enum CharClass {
    Char(char),
    RangeInclusive(char, char),
}
#[derive(PartialEq, Eq, Clone)]
enum EpsCharMatch {
    Epsilon,
    Match(CharClass),
}
//An edge of the NSM that consumes input
#[derive(PartialEq, Eq, Clone, Debug, PartialOrd, Ord)]
pub struct NsmConsumeEdge {
    char_match: CharClass,
    target_node: NodeIndex,
}
#[derive(PartialEq, Eq, Clone, Debug, PartialOrd, Ord)]
pub enum NsmEdgeTransition {
    Consume(NsmConsumeEdge),
    //An edge of the NSM that returns to an unknown place and therefore
    //doesn't consume input.
    Return,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct NsmEdgeData {
    pub transition: NsmEdgeTransition,
    pub actions: Vec<Action>,
    //A list of nodes to push on the return stack when transitioning this edge.
    //The VM will use separate stacks for return addresses and for data.
    pub push_return_stack: Vec<NodeIndex>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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
    let elim = eliminate_epsilon(&graph);
    let deduped = full_dedup(&elim);
    let start_node = *machine_starts
        .get(start_machine)
        .ok_or_else(|| LoweringError::UnknownExpression(start_machine.clone()))?;
    Ok(NsmInstructions {
        start_node,
        graph: deduped,
    })
}

//Removes all unused edges from the graph.
fn remove_unused_edges(graph: &Graph<NsmEdgeData>) -> Graph<NsmEdgeData> {
    let mut res: Graph<NsmEdgeData> = Graph::new();
    let mut live_edges = vec![false; graph.edges.len()];
    for i in 0..graph.nodes.len() {
        for &edge_idx in &graph.get_node(NodeIndex(i)).out_edges {
            live_edges[edge_idx.0] = true;
        }
    }
    let mut remap_table = vec![0; graph.edges.len()];
    let mut new_edges = Vec::new();
    for i in 0..live_edges.len() {
        if live_edges[i] {
            remap_table[i] = new_edges.len();
            new_edges.push(graph.get_edge(EdgeIndex(i)).clone());
        }
    }
    for node in &graph.nodes {
        let node_idx = res.create_node();
        res.get_node_mut(node_idx).out_edges = node
            .out_edges
            .iter()
            .map(|x| EdgeIndex(remap_table[x.0]))
            .collect();
    }
    res.edges = new_edges;
    res
}

fn deduplicate_edges(graph: &Graph<NsmEdgeData>) -> Graph<NsmEdgeData> {
    let mut edge_idx: Vec<(usize, &Edge<NsmEdgeData>)> = graph.edges.iter().enumerate().collect();
    edge_idx.sort_by(|edge1, edge2| edge1.1.data.cmp(&edge2.1.data));
    let mut remap_table = vec![0; graph.edges.len()];
    remap_table[edge_idx[0].0] = 0;
    let mut new_edges = Vec::new();
    new_edges.push(graph.get_edge(EdgeIndex(edge_idx[0].0)).clone());
    for i in 1..graph.edges.len() {
        if edge_idx[i].1.data != edge_idx[i - 1].1.data {
            new_edges.push(graph.get_edge(EdgeIndex(edge_idx[i].0)).clone());
        }
        remap_table[edge_idx[i].0] = new_edges.len() - 1;
    }

    let mut res: Graph<NsmEdgeData> = Graph::new();
    for node in &graph.nodes {
        let node_idx = res.create_node();
        let mut new_out_edges: Vec<_> = node
            .out_edges
            .iter()
            .map(|x| EdgeIndex(remap_table[x.0]))
            .collect();
        new_out_edges.sort();
        new_out_edges.dedup();
        res.get_node_mut(node_idx).out_edges = new_out_edges.into();
    }
    res.edges = new_edges;
    res
}
//Merges all nodes with duplicate sets of out_edges. Out edges are duplicate
//if they are in the same order, point to the same nodes and have the same actions.
fn deduplicate_nodes(graph: &Graph<NsmEdgeData>) -> Graph<NsmEdgeData> {
    let (node_remap_table, count) = build_node_deduplicate_remap_table(graph);
    remap_graph_nodes(graph, &node_remap_table, count)
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
        graph = remap_graph_nodes(&graph, &node_remap_table, count);
        graph = deduplicate_edges(&graph);
    }
    remove_unused_edges(&graph)
}
//Takes in a graph, a mapping from node to node and the number of nodes in the remapped graph.
//It maps the ndoes and edges from the original graph to new ones following the table.
//If multiple nodes are mapped onto the same node, all of their edges will be
//concatenated.
fn remap_graph_nodes(
    graph: &Graph<NsmEdgeData>,
    node_remap_table: &Vec<usize>,
    new_node_count: usize,
) -> Graph<NsmEdgeData> {
    let mut res: Graph<NsmEdgeData> = Graph::new();
    for _ in 0..new_node_count {
        res.create_node();
    }
    for node in graph.nodes() {
        res.get_node_mut(remap_node(node_remap_table, node))
            .out_edges
            .append(&mut graph.get_node(node).out_edges.clone());
    }
    let new_edges: Vec<Edge<NsmEdgeData>> = graph
        .edges
        .iter()
        .map(|edge| {
            let transition = match &edge.data.transition {
                NsmEdgeTransition::Return => NsmEdgeTransition::Return,
                NsmEdgeTransition::Consume(NsmConsumeEdge {
                    char_match,
                    target_node,
                }) => NsmEdgeTransition::Consume(NsmConsumeEdge {
                    char_match: char_match.clone(),
                    target_node: remap_node(&node_remap_table, *target_node),
                }),
            };
            let push_return_stack = edge
                .data
                .push_return_stack
                .iter()
                .map(|x| remap_node(&node_remap_table, *x))
                .collect();
            Edge {
                data: NsmEdgeData {
                    transition,
                    actions: edge.data.actions.clone(),
                    push_return_stack,
                },
            }
        })
        .collect();
    res.edges = new_edges;
    res
}

//This builds a remapping table for deduplicting nodes.
//Precondition: Edges must have been deduplicated first.
fn build_node_deduplicate_remap_table(graph: &Graph<NsmEdgeData>) -> (Vec<usize>, usize) {
    let mut nodes: Vec<(usize, VecDeque<EdgeIndex>)> = graph
        .nodes
        .iter()
        .map(|node| node.out_edges.clone())
        .enumerate()
        .collect();
    //Sort by edge data to identify duplicates.
    nodes.sort_by(|a, b| a.1.cmp(&b.1));
    let mut node_remap_table = vec![0; nodes.len()];
    let mut remap_counter = 0;
    //The 0th node is always remapped to itself.
    node_remap_table[0] = 0;
    for i in 1..nodes.len() {
        if Some(&nodes[i].1) != nodes.get(i - 1).map(|x| &x.1) {
            remap_counter += 1;
        }
        node_remap_table[i] = remap_counter;
    }
    (node_remap_table, remap_counter + 1)
}
fn remap_node(table: &Vec<usize>, node_index: NodeIndex) -> NodeIndex {
    NodeIndex(table[node_index.0])
}

//Identifies the strongly connected components in the graph
//among edges that aren't call or return edges.
//The result of this function is a topological sort.
fn identify_scc(graph: &Graph<NsmEdgeData>) -> Vec<usize> {
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
            contract_scc_rec(
                graph,
                &mut index,
                &mut on_stack,
                &mut indexs,
                &mut lowlinks,
                &mut stack,
                &mut mapping_table,
                &mut scc_count,
                NodeIndex(i),
            );
        }
    }
    mapping_table
}

fn contract_scc_rec(
    graph: &Graph<NsmEdgeData>,
    index: &mut usize,
    on_stack: &mut Vec<bool>,
    indexs: &mut Vec<usize>,
    lowlinks: &mut Vec<usize>,
    stack: &mut Vec<usize>,
    mapping_table: &mut Vec<usize>,
    scc_count: &mut usize,
    node_index: NodeIndex,
) {
    indexs[node_index.0] = *index;
    lowlinks[node_index.0] = *index;
    *index += 1;
    stack.push(node_index.0);
    on_stack[node_index.0] = true;

    let node = graph.get_node(node_index);
    for edge_idx in &node.out_edges {
        let edge_data = &graph.get_edge(*edge_idx).data;
        if let NsmEdgeTransition::Consume(target) = &edge_data.transition {
            let target = target.target_node;
            if edge_data.push_return_stack.len() == 0 {
                if indexs[target.0] == 0 {
                    contract_scc_rec(
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
                    lowlinks[node_index.0] = min(lowlinks[node_index.0], lowlinks[target.0]);
                } else if on_stack[target.0] {
                    lowlinks[node_index.0] = min(lowlinks[node_index.0], lowlinks[target.0]);
                }
            }
        }
    }
    if lowlinks[node_index.0] == indexs[node_index.0] {
        while let Some(node) = stack.pop() {
            on_stack[node] = false;
            mapping_table[node] = *scc_count;
            if node == node_index.0 {
                break;
            }
        }
        *scc_count += 1;
    }
}

//Returns a summary of which nodes are reachable from which other nodes starting
//and ending with an empty stack.
fn reachability_summary(graph: &Graph<NsmEdgeData>) -> Graph<()> {
    let mut table = Vec::new();
    let contains_return = contains_return_edge(graph);
    for _ in 0..graph.nodes.len() {
        table.push(vec![false; graph.nodes.len()]);
    }
    for i in 0..graph.nodes.len() {
        let node = graph.get_node(NodeIndex(i));
        for edge_idx in &node.out_edges {
            let edge_data = &graph.get_edge(*edge_idx).data;
            if let NsmEdgeTransition::Consume(transition) = &edge_data.transition {
                table[i][transition.target_node.0] = true;
            }
        }
    }
    let mut res: Graph<()> = Graph::new();
    for _ in 0..graph.nodes.len() {
        res.create_node();
    }
    for node in &graph.nodes {
        for &edge in &node.out_edges {
            let edge = graph.get_edge(edge);
        }
    }
    res
}

fn contains_return_edge(graph: &Graph<NsmEdgeData>) -> Vec<bool> {
    let mut res = vec![false; graph.nodes.len()];
    for i in 0..graph.nodes.len() {
        let node = graph.get_node(NodeIndex(i));
        for edge_idx in &node.out_edges {
            let edge_data = &graph.get_edge(*edge_idx).data;
            res[i] = res[i] || edge_data.transition == NsmEdgeTransition::Return;
        }
    }
    res
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
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }
    fn create_node(&mut self) -> NodeIndex {
        self.nodes.push(Node {
            out_edges: VecDeque::new(),
        });
        NodeIndex(self.nodes.len() - 1)
    }
    fn get_node(&self, idx: NodeIndex) -> &Node {
        &self.nodes[idx.0]
    }
    fn get_node_mut(&mut self, idx: NodeIndex) -> &mut Node {
        &mut self.nodes[idx.0]
    }
    fn nodes(&self) -> impl Iterator<Item = NodeIndex> {
        (0..self.nodes.len()).map(|x| NodeIndex(x))
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
        edge_index: EdgeIndex,
    ) -> Option<NodeIndex> {
        let edge = graph.get_edge(edge_index);
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

    fn get_return_stack(&self) -> Vec<NodeIndex> {
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
        edge_index: EdgeIndex,
    ) -> Option<NodeIndex> {
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

    fn get_return_stack(&self) -> Vec<NodeIndex> {
        self.stack.get_return_stack()
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
        GrammarEx::CharRangeInclusive(m, n) => {
            let end_node = graph.create_node();
            graph.add_edge_lowest_priority(
                start_node,
                EpsNsmEdgeData {
                    target: EdgeTarget::NodeIndex(end_node),
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
        dbg!(machine);
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
        dbg!(machine);
    }
}
