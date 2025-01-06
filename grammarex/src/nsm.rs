use std::{collections::VecDeque};


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Node {
    //Out edges are sorted by priority for taking them.
    pub out_edges: VecDeque<usize>,
}
#[derive(Clone, Debug)]
pub struct Edge<EdgeData> {
    pub data: EdgeData,
}


// Invariants:
// For each edge, it is in the node it starts at's out_edges and no other out_edges
// For each node, an edge is in its out_edges iff it starts at the node.
#[derive(Debug, Clone)]
pub struct Graph<EdgeData> {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge<EdgeData>>,
    pub start_node: Option<usize>,
}

impl<EdgeData> Graph<EdgeData> {
    pub fn new(start_node: Option<usize>) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            start_node
        }
    }

    pub fn create_node(&mut self) -> usize {
        self.nodes.push(Node {
            out_edges: VecDeque::new(),
        });
        self.nodes.len() - 1
    }
    pub fn get_node(&self, idx: usize) -> &Node {
        &self.nodes[idx]
    }
    pub fn get_node_mut(&mut self, idx: usize) -> &mut Node {
        &mut self.nodes[idx]
    }
    pub fn nodes(&self) -> impl Iterator<Item = usize> {
        0..self.nodes.len()
    }
    pub fn get_edge(&self, idx: usize) -> &Edge<EdgeData> {
        &self.edges[idx]
    }
    pub fn get_edge_mut(&mut self, idx: usize) -> &mut Edge<EdgeData> {
        &mut self.edges[idx]
    }
    pub fn add_edge_lowest_priority(&mut self, start: usize, data: EdgeData) {
        let idx = self.edges.len();
        self.edges.push(Edge { data });
        self.nodes[start].out_edges.push_back(idx);
    }

    pub fn add_edge_highest_priority(&mut self, start: usize, data: EdgeData) {
        let idx = self.edges.len();
        self.edges.push(Edge { data });
        self.nodes[start].out_edges.push_front(idx);
    }

    pub fn sort_dedup_out_edges(&mut self) {
        for i in 0..self.nodes.len() {
            let mut out_edges: Vec<usize> = self.get_node_mut(i).out_edges.clone().into();
            out_edges.sort();
            out_edges.dedup();
            self.get_node_mut(i).out_edges = out_edges.into();
        }
    }
}

pub trait Remappable {
    //Remaps the referenced nodes in an edge according to the remap table.
    //If this can't be done, return None.
    fn remap(&self, remap: &Vec<Option<usize>>) -> Option<Self> where Self: Sized;
}

impl<EdgeData: Remappable> Graph<EdgeData> { 
    //Remaps graph nodes. All nodes that map to the same node_remap are merged.
    pub fn remap_nodes(&self, 
                   node_remap: &Vec<Option<usize>>, 
                   nodes_after_remap: usize) -> Self {
        assert_eq!(node_remap.len(),self.nodes.len());
        let mut res = Self::new(None);
        for _ in 0..nodes_after_remap {
            res.create_node();
        }
        let mut remapped_edges = Vec::new();
        let mut edge_remap_table = Vec::new();
        for edge in &self.edges {
            if let Some(mapped_data) = edge.data.remap(node_remap) {
                edge_remap_table.push(Some(remapped_edges.len()));
                remapped_edges.push(Edge {
                    data: mapped_data
                });
            } else {
                edge_remap_table.push(None);
            }
        }
        for i in 0..node_remap.len() {
            for &edge in &self.get_node(i).out_edges {
                if let Some(mapped_edge) = edge_remap_table[edge] {
                    if let Some(mapped_node) = node_remap[i] {
                        res.get_node_mut(mapped_node).out_edges.push_back(mapped_edge);
                    }
                }
            }
        }
        res.sort_dedup_out_edges();
        res.edges = remapped_edges;
        res.start_node = self.start_node.and_then(|start|node_remap[start]);
        res
    }
}