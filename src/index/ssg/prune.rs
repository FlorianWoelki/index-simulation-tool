use std::collections::HashSet;

use ordered_float::OrderedFloat;

use crate::index::neighbor::NeighborNode;

use super::SSGIndex;

impl SSGIndex {
    /// Prunes the graph for the given query ID and populates the pruned graph vector.
    ///
    /// # Arguments
    ///
    /// * `query_id` - The ID of the query node.
    /// * `expand_neighbors` - A mutable vector to store the expanded neighbors.
    /// * `pruned_graph` - A mutable vector to store the pruned neighbors.
    pub(super) fn prune_graph(
        &mut self,
        query_id: usize,
        expand_neighbors: &mut Vec<NeighborNode>,
        pruned_graph: &mut [NeighborNode],
    ) {
        let visited = self.graph[query_id]
            .iter()
            .cloned()
            .collect::<HashSet<usize>>();
        // Expand the neighbors of the query node.
        expand_neighbors.extend(
            self.graph[query_id]
                .iter()
                .filter(|&linked_id| !visited.contains(linked_id))
                .map(|&linked_id| {
                    let distance =
                        self.vectors[query_id].distance(&self.vectors[linked_id], self.metric);
                    NeighborNode::new(linked_id, distance)
                }),
        );
        expand_neighbors.sort_unstable();

        let mut result = Vec::new();
        // Add the neighbors to the result vector if they are not occluded.
        expand_neighbors
            .iter()
            .filter(|n| n.id != query_id)
            .take(self.index_size)
            .cloned()
            .for_each(|node| {
                if !self.is_occluded(&result, &node) {
                    result.push(node);
                }
            });
        self.populate_pruned_graph(pruned_graph, &result, query_id);
    }

    /// Populates the pruned graph vector with the given result vector for the specified query ID.
    ///
    /// # Arguments
    ///
    /// * `pruned_graph` - A mutable vector to store the pruned neighbors.
    /// * `result` - A reference to the vector containing the pruned neighbors.
    /// * `query_id` - The ID of the query node.
    fn populate_pruned_graph(
        &self,
        pruned_graph: &mut [NeighborNode],
        result: &[NeighborNode],
        query_id: usize,
    ) {
        let base_index = query_id * self.index_size;
        for (i, node) in result.iter().enumerate() {
            pruned_graph[base_index + i].id = node.id;
            pruned_graph[base_index + i].distance = node.distance;
        }
        for i in result.len()..self.index_size {
            pruned_graph[base_index + i].id = self.vectors.len();
            pruned_graph[base_index + i].distance = f64::MAX.into();
        }
    }

    /// Checks if a candidate node is occluded by any existing node in the result vector.
    ///
    /// # Arguments
    ///
    /// * `result` - A reference to the vector containing the existing nodes.
    /// * `candidate` - A reference to the candidate node.
    ///
    /// # Returns
    ///
    /// `true` if the candidate node is occuluded by any existing node, `false` otherwise.
    fn is_occluded(&self, result: &[NeighborNode], candidate: &NeighborNode) -> bool {
        result.iter().any(|existing| {
            let djk = self.vectors[existing.id].distance(&self.vectors[candidate.id], self.metric);
            let cos_ij = (candidate.distance.powi(2) + existing.distance.powi(2) - djk.powi(2))
                / (2.0 * (candidate.distance.into_inner() * existing.distance.into_inner()));
            cos_ij > self.threshold
        })
    }

    /// Interconnects the pruned neighbors for the given node index.
    ///
    /// # Arguments
    ///
    /// * `node_index` - The index of the node for which to interconnect the pruned neighbors.
    /// * `max_neighbors` - The maximum number of neighbors to interconnect.
    /// * `pruned_graph` - A mutable vector containing the pruned neighbors.
    pub(super) fn interconnect_pruned_neighbors(
        &self,
        node_index: usize,
        max_neighbors: usize,
        pruned_graph: &mut [NeighborNode],
    ) {
        for i in 0..max_neighbors {
            let current_node = &pruned_graph[node_index + i];
            if current_node.distance.into_inner() == f64::MAX {
                continue;
            }

            let neighbor_node = NeighborNode::new(node_index, current_node.distance.into_inner());
            let destination_id = current_node.id;
            let start_index = destination_id * self.index_size;

            let mut neighbors =
                self.collect_neighbors(pruned_graph, start_index, max_neighbors, node_index);
            if neighbors.is_empty() {
                continue;
            }

            neighbors.push(neighbor_node.clone());
            self.update_pruned_neighbors_list(
                pruned_graph,
                start_index,
                max_neighbors,
                &neighbor_node,
                &mut neighbors,
            );
        }
    }

    /// Collects the neighbors of a node from the pruned graph vector.
    /// The neighbors are pruned to the specified range.
    ///
    /// # Arguments
    ///
    /// * `pruned_graph` - A reference to the pruned graph vector.
    /// * `max_neighbors` - The maximum number of neighbors to collect.
    ///
    /// # Returns
    ///
    /// A vector containing the pruned neighbors.
    fn prune_neighbors(
        &self,
        neighbors: &mut [NeighborNode],
        max_neighbors: usize,
    ) -> Vec<NeighborNode> {
        let mut result = Vec::new();
        neighbors.sort_unstable();
        result.push(neighbors[0].clone());

        let mut start = 1;
        while result.len() < max_neighbors && start < neighbors.len() {
            let p = &neighbors[start];
            let occluded = result
                .iter()
                .any(|rt| p.id == rt.id || self.is_occluded(&result, p));

            if !occluded {
                result.push(p.clone());
            }

            start += 1;
        }

        result
    }

    /// Collects the neighbors of a node from the pruned graph vector.
    ///
    /// # Arguments
    ///
    /// * `pruned_graph` - A reference to the vector containing the pruned neighbors.
    /// * `start_index` - The starting index in the `pruned_graph` vector.
    /// * `max_neighbors` - The maximum number of neighbors to collect.
    /// * `current_node_index` - The index of the current node.
    ///
    /// # Returns
    ///
    /// A vector containing the collected neighbors.
    fn collect_neighbors(
        &self,
        pruned_graph: &[NeighborNode],
        start_index: usize,
        max_neighbors: usize,
        current_node_index: usize,
    ) -> Vec<NeighborNode> {
        let mut has_duplicate = false;

        let neighbors = (0..max_neighbors)
            .filter(|i| {
                let neighbor = &pruned_graph[start_index + i];
                neighbor.distance.into_inner() != f64::MAX
            })
            .map(|i| {
                let neighbor = &pruned_graph[start_index + i];
                if current_node_index == neighbor.id {
                    has_duplicate = true;
                }
                neighbor.clone()
            })
            .collect();

        if has_duplicate {
            return Vec::new();
        }

        neighbors
    }

    /// Updates the pruned neighbors list for a node in the pruned graph vector.
    ///
    /// # Arguments
    ///
    /// * `pruned_graph` - A mutable reference to the vector containing the pruned neighbors.
    /// * `start_index` - The starting index in the `pruned_graph` vector.
    /// * `max_neighbors` - The maximum number of neighbors to consider.
    /// * `neighbor_node` - A reference to the neighbor node to update.
    /// * `neighbors` - A reference to the vector containing the neighbors to update.
    fn update_pruned_neighbors_list(
        &self,
        pruned_graph: &mut [NeighborNode],
        start_index: usize,
        max_neighbors: usize,
        neighbor_node: &NeighborNode,
        neighbors: &mut [NeighborNode],
    ) {
        if neighbors.len() > max_neighbors {
            let result = self.prune_neighbors(neighbors, max_neighbors);
            (0..result.len()).for_each(|t| {
                pruned_graph[t + start_index] = result[t].clone();
            });

            if result.len() < max_neighbors {
                pruned_graph[result.len() + start_index].distance = OrderedFloat(f64::MAX);
            }
        } else {
            (0..max_neighbors).for_each(|i| {
                if pruned_graph[i + start_index].distance.into_inner() != f64::MAX {
                    return;
                }
                pruned_graph[i + start_index] = neighbor_node.clone();
                if (i + 1) < max_neighbors {
                    pruned_graph[i + start_index].distance = OrderedFloat(f64::MAX);
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use crate::{
        data::HighDimVector,
        index::{DistanceMetric, Index},
    };

    use super::*;

    #[test]
    fn test_prune_graph() {
        let mut index = SSGIndex::new(DistanceMetric::Euclidean);
        index.threshold = 0.2;
        let v0 = HighDimVector::new(0, vec![1.0, 2.0, 3.0]);
        let v1 = HighDimVector::new(1, vec![2.0, 3.0, 4.0]);
        let v2 = HighDimVector::new(2, vec![3.0, 4.0, 5.0]);

        index.add_vector(v0.clone());
        index.add_vector(v1.clone());
        index.add_vector(v2.clone());
        index.construct_knn_graph(3);

        let mut expanded_neighbors = vec![NeighborNode::new(1, 0.1), NeighborNode::new(2, 0.3)];
        let len = index.vectors.len() * index.index_size;
        let mut pruned_graph = vec![NeighborNode::new(0, 0.0); len];

        index.prune_graph(0, &mut expanded_neighbors, &mut pruned_graph);

        assert_eq!(pruned_graph.len(), 300, "Pruned graph length is incorrect");
        assert!(
            pruned_graph
                .iter()
                .any(|node| node.id == 1 && node.distance == OrderedFloat(0.1)),
            "Node 1 not found"
        );
    }

    #[test]
    fn test_populate_pruned_graph() {
        let mut index = SSGIndex::new(DistanceMetric::Euclidean);
        index.index_size = 10;
        let query_id = 0;
        let mut pruned_graph = vec![NeighborNode::new(usize::MAX, f64::MAX.into()); 10];
        let result = vec![NeighborNode::new(1, 10.0), NeighborNode::new(2, 20.0)];

        index.populate_pruned_graph(&mut pruned_graph, &result, query_id);

        assert_eq!(pruned_graph[0].id, 1, "First node id is incorrect");
        assert_eq!(
            pruned_graph[0].distance,
            OrderedFloat(10.0),
            "First node distance is incorrect"
        );
        assert_eq!(pruned_graph[1].id, 2, "Second node id is incorrect");
        assert_eq!(
            pruned_graph[1].distance,
            OrderedFloat(20.0),
            "Second node distance is incorrect"
        );
        assert_eq!(
            pruned_graph[2].id,
            index.vectors.len(),
            "Third node id is incorrect"
        );
        assert_eq!(
            pruned_graph[2].distance.into_inner(),
            f64::MAX,
            "Third node distance is incorrect"
        );
    }

    #[test]
    fn test_is_occluded() {
        let mut index = SSGIndex::new(DistanceMetric::Euclidean);
        index.threshold = 0.0;
        index.vectors = vec![
            HighDimVector::new(0, vec![0.0, 1.0]),
            HighDimVector::new(1, vec![1.0, 0.0]),
            HighDimVector::new(2, vec![0.5, 0.5]),
        ];

        let result = vec![NeighborNode::new(0, 1.414)]; // From origin to (1, 0)
        let candidate = NeighborNode::new(2, 0.707); // From origin to (0.5, 0.5)

        let is_occluded = index.is_occluded(&result, &candidate);

        assert!(
            is_occluded,
            "The candidate should be considered as occluded based on the threshold."
        );
    }

    #[test]
    fn test_interconnect_pruned_neighbors() {
        let mut index = SSGIndex::new(DistanceMetric::Euclidean);
        index.index_size = 3;
        for i in 0..5 {
            index.add_vector(HighDimVector::new(i, vec![i as f64, i as f64 * 2.0]));
        }
        let mut pruned_graph = vec![
            NeighborNode::new(1, 10.0), // Node 0 connects to Node 1
            NeighborNode::new(2, 20.0), // Node 0 connects to Node 2
            NeighborNode::new(3, 30.0), // Node 0 connects to Node 3
            NeighborNode::new(0, 10.0), // Node 1 connects back to Node 0
            NeighborNode::new(usize::MAX, f64::MAX),
            NeighborNode::new(usize::MAX, f64::MAX),
        ];

        pruned_graph.resize(
            index.vectors.len() * index.index_size,
            NeighborNode::new(usize::MAX, f64::MAX),
        );
        index.interconnect_pruned_neighbors(0, index.index_size, &mut pruned_graph);
        //assert!(false, "TODO");
    }

    #[test]
    fn test_collect_neighbors() {
        let index = SSGIndex::new(DistanceMetric::Euclidean);
        let pruned_graph = vec![
            NeighborNode::new(1, 10.0),
            NeighborNode::new(2, 20.0),
            NeighborNode::new(1, 10.0), // Duplicate node
            NeighborNode::new(usize::MAX, f64::MAX),
        ];

        let neighbors = index.collect_neighbors(&pruned_graph, 0, 3, 2);
        assert!(
            neighbors.is_empty(),
            "Should return an empty vector due to duplicate detection"
        );

        let neighbors = index.collect_neighbors(&pruned_graph, 0, 3, 3);
        assert_eq!(
            neighbors.len(),
            3,
            "Should collect two neighbors before hitting a placeholder"
        );
        assert_eq!(neighbors[0].id, 1, "First neighbor should be node 1");
        assert_eq!(neighbors[1].id, 2, "Second neighbor should be node 2");
    }

    #[test]
    fn test_update_pruned_neighbors_list() {
        let mut index = SSGIndex::new(DistanceMetric::Euclidean);
        index.threshold = 0.1;
        index.index_size = 3;

        index.add_vector(HighDimVector::new(0, vec![0.0, 0.0]));
        index.add_vector(HighDimVector::new(1, vec![1.0, 1.0]));
        index.add_vector(HighDimVector::new(2, vec![2.0, 2.0]));
        index.add_vector(HighDimVector::new(3, vec![5.0, 5.0]));

        let mut pruned_graph = vec![
            NeighborNode::new(1, 1.0),
            NeighborNode::new(2, 2.0),
            NeighborNode::new(3, 4.0),
            NeighborNode::new(usize::MAX, f64::MAX),
            NeighborNode::new(usize::MAX, f64::MAX),
        ];

        let mut neighbors = index.collect_neighbors(&pruned_graph, 0, index.index_size, 0);
        neighbors.sort_unstable();

        let neighbor_node = NeighborNode::new(0, pruned_graph[0].distance.into_inner());
        index.update_pruned_neighbors_list(
            &mut pruned_graph,
            0,
            index.index_size,
            &neighbor_node,
            &mut neighbors,
        );

        assert_eq!(
            pruned_graph.len(),
            5,
            "Pruned graph should remain the same size"
        );
        assert_eq!(pruned_graph[0].id, 1, "First node should be node 1");
        assert_eq!(
            pruned_graph[0].distance.into_inner(),
            1.0,
            "First node distance should be 1.0"
        );
        assert_eq!(pruned_graph[1].id, 2, "Second node should be node 2");
        assert_eq!(
            pruned_graph[1].distance.into_inner(),
            2.0,
            "Second node distance should be 2.0"
        );
        assert_eq!(
            pruned_graph.last().unwrap().id,
            usize::MAX,
            "Last node should be a placeholder"
        );
    }
}
