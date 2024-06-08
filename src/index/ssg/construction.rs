use std::collections::BinaryHeap;

use crate::index::neighbor::NeighborNode;

use super::SSGIndex;

impl SSGIndex {
    pub(super) fn construct_knn_graph(&mut self, k: usize) {
        // TODO: Consider parallel processing.
        self.graph = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, current_vector)| {
                let mut neighbor_heap = BinaryHeap::with_capacity(k + 1); // Extra capacity for efficiency.

                self.vectors
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| i != *j)
                    .for_each(|(j, node)| {
                        let distance = current_vector.euclidean_distance(node);
                        neighbor_heap.push(NeighborNode::new(j, distance));

                        // Ensures the heap does not grow beyond k elements.
                        if neighbor_heap.len() > k {
                            neighbor_heap.pop();
                        }
                    });

                // Collects the k-nearest neighbors from the heap.
                let mut neighbors = Vec::with_capacity(neighbor_heap.len());
                while let Some(neighbor_node) = neighbor_heap.pop() {
                    neighbors.push(neighbor_node.id);
                }

                neighbors
            })
            .collect();
    }
}
