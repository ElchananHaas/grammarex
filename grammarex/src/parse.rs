use crate::{
    lower_nfa::{NsmEdgeData, NsmEdgeTransition},
    nsm::Graph,
};

#[derive(PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
struct StateId {
    arena: usize,
    //If the node is in the free list this is reused to hold 1 plus the
    //index of the next item in the free list, or 0 if there is no such item.
    index: PackedIndex,
}

#[derive(PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
struct PackedIndex(usize);

impl PackedIndex {
    pub fn new(index: usize, offset: u8) -> Self {
        assert!(index < 1 << 56, "Index must be less than 2^56");
        PackedIndex((index << 8) | (offset as usize))
    }
    pub fn index(&self) -> usize {
        self.0 >> 8
    }
    pub fn offset(&self) -> usize {
        self.0 & (1 << 8)
    }
}
impl StateId {
    pub fn new(arena: usize, index: usize, offset: u8) -> Self {
        Self {
            arena,
            index: PackedIndex::new(index, offset),
        }
    }
    pub fn offset(&self) -> usize {
        self.index.offset()
    }
    pub fn index(&self) -> usize {
        self.index.index()
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
struct NodeRef {
    prev: StateId,
    next: StateId,
}
#[derive(PartialEq, Eq, Clone, Debug, PartialOrd, Ord)]
struct NodeState {
    refcount: usize,
    callers_ref: NodeRef,
    callees_ref: Vec<NodeRef>,
}

struct NodeStateArena {
    out_edges: Vec<NsmEdgeData>,
    num_calls: usize,
    nodes: Vec<NodeState>,
    free_list_start: Option<usize>,
}

struct NodeStates {
    arenas: Vec<NodeStateArena>,
}
impl NodeStateArena {
    fn new(out_edges: Vec<NsmEdgeData>) -> Self {
        let num_calls = out_edges
            .iter()
            .filter(|&edge| {
                if let NsmEdgeTransition::Call(_) = edge.transition {
                    true
                } else {
                    false
                }
            })
            .count();
        Self {
            out_edges,
            num_calls,
            nodes: Vec::new(),
            free_list_start: None,
        }
    }
    //Allocates a new node within the arena. Returns its index.
    fn alloc(&mut self) -> usize {
        if let Some(start) = self.free_list_start {
            let pointed_to = self.nodes[start].callers_ref.next.index;
            self.free_list_start = if pointed_to.offset() == 0 {
                None
            } else {
                Some(pointed_to.index())
            };
            //Make the newly allocated node's next point to itself.
            self.nodes[start].callers_ref.next.index = PackedIndex::new(start, 0);
            return start;
        } else {
            let res = self.nodes.len();
            let self_ref = NodeRef {
                prev: StateId::new(0, res, 0),
                next: StateId::new(0, res, 0),
            };
            self.nodes.push(NodeState {
                refcount: 0,
                callers_ref: self_ref,
                callees_ref: vec![self_ref; self.num_calls],
            });
            return res;
        }
    }

    fn drop(&mut self, index: usize) {
        if let Some(start) = self.free_list_start {
            self.nodes[index].callers_ref.next.index = PackedIndex::new(start, 1);
        } else {
            self.nodes[index].callers_ref.next.index = PackedIndex::new(0, 0);
        }
        self.free_list_start = Some(index);
    }
}

impl NodeStates {
    pub fn new(data: Vec<Vec<NsmEdgeData>>) -> Self {
        let mut arenas = Vec::new();
        for out_edges in data {
            arenas.push(NodeStateArena::new(out_edges));
        }
        NodeStates { arenas }
    }

    pub fn create_state(&mut self, arena: usize) -> StateId {
        let index = self.arenas[arena].alloc();
        StateId::new(arena, index, 0)
    }

    pub fn add_child(&mut self, node: StateId, child: StateId) {
        let next_state = self.get_state(node).callees_ref[node.offset()].next;
        self.get_state_mut(node).callees_ref[node.offset()].next = child;
        self.get_state_mut(node).refcount += 1;
        self.get_state_mut(next_state).callees_ref[next_state.offset()].prev = child;
        self.get_state_mut(child).callers_ref = NodeRef {
            prev: node,
            next: next_state,
        }
    }

    pub fn drop(&mut self, node: StateId) {
        let callers = self.get_state(node).callers_ref;
        self.arenas[node.arena].drop(node.index());
        self.get_state_mut(callers.prev).callees_ref[callers.prev.offset()].next = callers.next;
        self.get_state_mut(callers.next).callees_ref[callers.next.offset()].prev = callers.prev;
        let no_more_callees = callers.prev == callers.next;
        if no_more_callees {
            self.get_state_mut(callers.prev).refcount -= 1;
        }
        if self.get_state(callers.prev).refcount == 0 {
            self.drop(callers.prev);
        }
    }

    fn get_state_mut(&mut self, state: StateId) -> &mut NodeState {
        &mut self.arenas[state.arena].nodes[state.index.index()]
    }

    fn get_state(&self, state: StateId) -> &NodeState {
        &self.arenas[state.arena].nodes[state.index.index()]
    }
}

struct GllStates {
    graph: Graph<NsmEdgeData>,
    gss: NodeStates,
}

impl GllStates {}
fn parse(states: &mut NodeStates, input: &str) {
    for char in input.chars() {}
}
