use std::collections::{BinaryHeap, HashSet, VecDeque};

use crate::{data::SparseVector, index::neighbor::NeighborNode};

use super::SSGIndex;

impl SSGIndex {
    pub(super) fn search_bfs(&self, query_vector: &SparseVector, k: usize) -> Vec<SparseVector> {
        let mut visited = HashSet::new();
        let mut heap = BinaryHeap::new();
        let mut search_queue = VecDeque::new();

        self.process_initial_nodes(&mut heap, &mut search_queue, &mut visited, query_vector, k);

        while let Some(id) = search_queue.pop_front() {
            if let Some(node_vec) = self.graph.get(id) {
                node_vec.iter().for_each(|&neighbor_id| {
                    if !visited.insert(neighbor_id) {
                        return;
                    }

                    let distance = query_vector.euclidean_distance(&self.vectors[neighbor_id]);
                    let neighbor_node = NeighborNode::new(neighbor_id, distance);
                    heap.push(neighbor_node);
                    search_queue.push_back(neighbor_id);
                });
            }

            if heap.len() > k {
                heap.pop();
            }
        }

        let mut result = Vec::with_capacity(heap.len());
        while let Some(node) = heap.pop() {
            result.push(self.vectors[node.id].clone());
        }

        result.reverse();
        result
    }

    fn process_initial_nodes(
        &self,
        heap: &mut BinaryHeap<NeighborNode>,
        search_queue: &mut VecDeque<usize>,
        visited: &mut HashSet<usize>,
        query_vector: &SparseVector,
        k: usize,
    ) {
        // Sort the root nodes by distance to the query vector.
        let mut initial_nodes = self
            .root_nodes
            .iter()
            .map(|&n| {
                let distance = query_vector.euclidean_distance(&self.vectors[n]);
                NeighborNode::new(n, distance)
            })
            .collect::<Vec<_>>();
        initial_nodes.sort();

        // Add the k closest root nodes to the heap to initialize the search.
        initial_nodes.iter().for_each(|node| {
            if heap.len() < k {
                heap.push(node.clone());
                search_queue.push_back(node.id);
            }
            visited.insert(node.id);
        });
    }
}
