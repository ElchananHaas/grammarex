use crate::lower_nfa::{NsmEdgeData, NsmEdgeTransition};

#[derive(PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
struct NodeRef {
    parent_node: usize,
    parent_index: usize,
    child_node: usize,
    child_index: usize,
}
#[derive(PartialEq, Eq, Clone, Debug, PartialOrd, Ord)]
struct NodeState {
    caller_ref: NodeRef,
    callee_ref: Vec<NodeRef>,
}

struct NodeStateArena {
    out_edges: Vec<NsmEdgeData>,
    num_calls: usize,
    nodes: Vec<NodeState>,
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
        }
    }
}
