use std::collections::BinaryHeap;

use crate::index::neighbor::NeighborNode;

use super::SSGIndex;

impl SSGIndex {
    pub(super) fn construct_knn_graph(&mut self, k: usize) {
        self.graph = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, current_vector)| {
                // TODO: Consider using MinHeap data structure
                let mut neighbor_heap: BinaryHeap<_> = self
                    .vectors
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| i != *j)
                    .map(|(j, node)| {
                        let distance = current_vector.euclidean_distance(node);
                        NeighborNode::new(j, distance)
                    })
                    .collect();

                while neighbor_heap.len() > k {
                    neighbor_heap.pop();
                }

                neighbor_heap
                    .into_sorted_vec()
                    .into_iter()
                    .map(|node| node.id)
                    .collect()
            })
            .collect();
    }
}
