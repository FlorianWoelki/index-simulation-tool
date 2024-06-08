use std::collections::HashSet;

use ordered_float::OrderedFloat;

use crate::index::neighbor::NeighborNode;

use super::SSGIndex;

impl SSGIndex {
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
                        self.vectors[query_id].euclidean_distance(&self.vectors[linked_id]);
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
            pruned_graph[base_index + i].distance = f32::MAX.into();
        }
    }

    fn is_occluded(&self, result: &[NeighborNode], candidate: &NeighborNode) -> bool {
        result.iter().any(|existing| {
            let djk = self.vectors[existing.id].euclidean_distance(&self.vectors[candidate.id]);
            let cos_ij = (candidate.distance.powi(2) + existing.distance.powi(2) - djk.powi(2))
                / (2.0 * (candidate.distance.into_inner() * existing.distance.into_inner()));
            cos_ij > self.threshold
        })
    }

    pub(super) fn interconnect_pruned_neighbors(
        &self,
        node_index: usize,
        max_neighbors: usize,
        pruned_graph: &mut [NeighborNode],
    ) {
        for i in 0..max_neighbors {
            let current_node = &pruned_graph[node_index + i];
            if current_node.distance.into_inner() == f32::MAX {
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
                neighbor.distance.into_inner() != f32::MAX
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
                pruned_graph[result.len() + start_index].distance = OrderedFloat(f32::MAX);
            }
        } else {
            (0..max_neighbors).for_each(|i| {
                if pruned_graph[i + start_index].distance.into_inner() != f32::MAX {
                    return;
                }
                pruned_graph[i + start_index] = neighbor_node.clone();
                if (i + 1) < max_neighbors {
                    pruned_graph[i + start_index].distance = OrderedFloat(f32::MAX);
                }
            });
        }
    }
}
