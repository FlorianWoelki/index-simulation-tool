use std::collections::{BinaryHeap, HashSet, VecDeque};

use crate::{data::HighDimVector, index::neighbor::NeighborNode};

use super::SSGIndex;

impl SSGIndex {
    /// Perform a breadth-first search to find the k closest vectors to the query vector.
    /// The search starts from the root nodes of the graph and expands to the neighbors of the nodes
    /// that are closer to the query vector.
    ///
    /// # Arguments
    ///
    /// * `query_vector` - The query vector.
    /// * `k` - The number of closest vectors to return.
    ///
    /// # Returns
    ///
    /// A vector containing the k closest vectors to the query vector.
    pub(super) fn search_bfs(&self, query_vector: &HighDimVector, k: usize) -> Vec<HighDimVector> {
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

                    let distance = query_vector.distance(&self.vectors[neighbor_id], self.metric);
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

    /// Process the initial nodes to start the search.
    /// The initial nodes are always the root nodes of the graph.
    ///
    /// # Arguments
    ///
    /// * `heap` - The heap to store the k closest nodes.
    /// * `search_queue` - The queue to store the nodes to be visited.
    /// * `visited` - The set to store the visited nodes.
    /// * `query_vector` - The query vector.
    /// * `k` - The number of closest nodes to return.
    fn process_initial_nodes(
        &self,
        heap: &mut BinaryHeap<NeighborNode>,
        search_queue: &mut VecDeque<usize>,
        visited: &mut HashSet<usize>,
        query_vector: &HighDimVector,
        k: usize,
    ) {
        // Sort the root nodes by distance to the query vector.
        let mut initial_nodes = self
            .root_nodes
            .iter()
            .map(|&n| {
                let distance = query_vector.distance(&self.vectors[n], self.metric);
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

#[cfg(test)]
mod tests {
    use crate::index::{DistanceMetric, Index};

    use super::*;

    #[test]
    fn test_search() {
        let mut index = SSGIndex::new(DistanceMetric::Euclidean);
        index.root_size = 10;
        for i in 0..10 {
            let v = HighDimVector::new(i, vec![i as f64, i as f64]);
            index.add_vector(v);
        }
        index.build();

        let query_vector = HighDimVector::new(99, vec![5.0, 5.0]);
        let results = index.search_bfs(&query_vector, 3);

        assert_eq!(results.len(), 3, "Should return exactly 3 results");
        let expected_ids = vec![5, 4, 6];
        let result_ids: Vec<usize> = results.iter().map(|v| v.id).collect();
        assert!(
            expected_ids.iter().all(|id| result_ids.contains(id)),
            "The closest vectors should include the ids 4, 5, and 6"
        );
    }

    #[test]
    fn test_search_empty_graph() {
        let mut index = SSGIndex::new(DistanceMetric::Euclidean);
        index.root_size = 0;
        index.build();
        let query_vector = HighDimVector::new(99, vec![5.0, 5.0]);
        let results = index.search_bfs(&query_vector, 3);

        assert!(results.is_empty(), "Should return an empty result");
    }
}
