use std::collections::HashSet;

use ordered_float::OrderedFloat;

use crate::index::neighbor::NeighborNode;

use super::SSGIndex;

impl SSGIndex {
    pub(super) fn prune_and_link_graph(&mut self) {
        let len = self.vectors.len() * self.pool_size;
        let mut pruned_graph = (0..len)
            .map(|i| NeighborNode::new(i, 0.0))
            .collect::<Vec<NeighborNode>>();

        self.link_each_node(&mut pruned_graph);

        for i in 0..self.vectors.len() {
            let offset = i * self.pool_size;
            let pool_size = (0..self.pool_size)
                .take_while(|j| pruned_graph[offset + j].distance == OrderedFloat(f32::MAX))
                .count()
                .max(1);
            self.graph[i] = (0..pool_size)
                .map(|j| pruned_graph[offset + j].id)
                .collect();
        }
    }

    fn expand_neighbors(&self, query_point: usize, expand_neighbors: &mut Vec<NeighborNode>) {
        let mut visited = HashSet::with_capacity(self.expanded_neighbor_size);
        visited.insert(query_point);

        for &neighbor_id in self.graph[query_point].iter() {
            for &second_neighbor_id in &self.graph[neighbor_id] {
                if second_neighbor_id != query_point && visited.insert(second_neighbor_id) {
                    let distance = self.vectors[query_point]
                        .euclidean_distance(&self.vectors[second_neighbor_id]);
                    expand_neighbors.push(NeighborNode::new(second_neighbor_id, distance));
                }
            }
        }
    }

    fn link_each_node(&mut self, pruned_graph: &mut [NeighborNode]) {
        let mut expanded_neighbors = Vec::new();

        for i in 0..self.vectors.len() {
            expanded_neighbors.clear();
            self.expand_neighbors(i, &mut expanded_neighbors);
            self.prune_graph(i, &mut expanded_neighbors, pruned_graph);
        }

        for i in 0..self.vectors.len() {
            self.interconnect_pruned_neighbors_list(i, self.pool_size, pruned_graph);
        }
    }

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
        for node in expand_neighbors
            .iter()
            .filter(|n| n.id != query_id)
            .take(self.pool_size)
            .cloned()
        {
            if !self.is_occluded(&result, &node) {
                result.push(node);
            }
        }
        self.update_pruned_graph(pruned_graph, &result, query_id);
    }

    fn update_pruned_graph(
        &self,
        pruned_graph: &mut [NeighborNode],
        result: &[NeighborNode],
        query_id: usize,
    ) {
        let base_index = query_id * self.pool_size;

        // Populate the pruned graph with result nodes.
        result.iter().enumerate().for_each(|(i, node)| {
            pruned_graph[base_index + i] = *node;
        });

        // Fill the remaining entries with default values.
        for i in result.len()..self.pool_size {
            pruned_graph[base_index + i] = NeighborNode {
                id: self.vectors.len(),
                distance: OrderedFloat(f32::MAX),
            }
        }
    }

    fn is_occluded(&self, result: &[NeighborNode], candidate: &NeighborNode) -> bool {
        result.iter().any(|existing| {
            let djk = self.vectors[existing.id].euclidean_distance(&self.vectors[candidate.id]);
            let cos_ij = (candidate.distance.powi(2) + existing.distance.powi(2) - djk.powi(2))
                / (2.0 * (candidate.distance.into_inner() * existing.distance.into_inner()));
            cos_ij > self.occlusion_threshold
        })
    }

    fn interconnect_pruned_neighbors_list(
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
            let start_index = destination_id * self.pool_size;

            let mut neighbors =
                self.get_neighbors(pruned_graph, start_index, max_neighbors, node_index);
            if neighbors.is_empty() {
                continue;
            }

            neighbors.push(neighbor_node.clone());
            self.update_neighbors_list(
                pruned_graph,
                start_index,
                max_neighbors,
                &neighbor_node,
                &mut neighbors,
            );
        }
    }

    fn prune_neighbor_list(
        &self,
        neighbors: &mut [NeighborNode],
        max_neighbors: usize,
    ) -> Vec<NeighborNode> {
        let mut pruned = Vec::with_capacity(max_neighbors);
        neighbors.sort_unstable();

        if let Some(first_neighbor) = neighbors.first() {
            pruned.push(first_neighbor.clone());
        }

        for neighbor in neighbors.iter().skip(1) {
            if pruned.len() >= max_neighbors {
                break;
            }

            if !pruned
                .iter()
                .any(|existing| neighbor.id == existing.id || self.is_occluded(&pruned, neighbor))
            {
                pruned.push(neighbor.clone());
            }
        }

        pruned
    }

    fn get_neighbors(
        &self,
        pruned_graph: &[NeighborNode],
        start_index: usize,
        max_neighbors: usize,
        current_node_index: usize,
    ) -> Vec<NeighborNode> {
        let mut has_duplicate = false;

        let neighbors: Vec<NeighborNode> = (0..max_neighbors)
            .filter_map(|i| {
                let neighbor = &pruned_graph[start_index + i];
                if neighbor.distance == f32::MAX {
                    None
                } else {
                    if current_node_index == neighbor.id {
                        has_duplicate = true;
                    }
                    Some(neighbor.clone())
                }
            })
            .collect();

        if has_duplicate {
            return Vec::new();
        }

        neighbors
    }

    fn update_neighbors_list(
        &self,
        pruned_graph: &mut [NeighborNode],
        start_index: usize,
        max_neighbors: usize,
        neighbor_node: &NeighborNode,
        neighbors: &mut [NeighborNode],
    ) {
        if neighbors.len() > max_neighbors {
            let result = self.prune_neighbor_list(neighbors, max_neighbors);
            for (i, node) in result.iter().enumerate() {
                pruned_graph[start_index + i] = node.clone();
            }

            if result.len() < max_neighbors {
                pruned_graph[start_index + result.len()].distance = OrderedFloat(f32::MAX);
            }
        } else {
            for i in 0..max_neighbors {
                if pruned_graph[i + start_index].distance != OrderedFloat(f32::MAX) {
                    return;
                }

                pruned_graph[start_index + i] = neighbor_node.clone();
                if (i + 1) < max_neighbors {
                    pruned_graph[start_index + i + 1].distance = OrderedFloat(f32::MAX);
                }
            }
        }
    }
}
