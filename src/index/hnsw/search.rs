use std::collections::{BinaryHeap, HashSet};

use ordered_float::OrderedFloat;

use crate::{data::SparseVector, index::neighbor::NeighborNode};

use super::HNSWIndex;

impl HNSWIndex {
    pub(super) fn search_knn(
        &self,
        query_vector: &SparseVector,
        k: usize,
    ) -> BinaryHeap<NeighborNode> {
        let mut top_candidate = BinaryHeap::new();
        if self.n_indexed_vectors == 0 {
            return top_candidate;
        }
        let mut current_id = self.root_node_id;
        let mut current_distance = self.vectors[current_id].euclidean_distance(query_vector);
        let mut current_level = self.current_level;
        loop {
            let mut changed = true;
            while changed {
                changed = false;
                let current_neighbors =
                    self.get_neighbor(current_id, current_level).read().unwrap();
                for neighbor in current_neighbors.iter() {
                    let distance = self.vectors[current_id].euclidean_distance(query_vector);
                    if distance < current_distance {
                        current_distance = distance;
                        current_id = *neighbor;
                        changed = true;
                    }
                }
            }
            if current_level == 0 {
                break;
            }
            current_level -= 1;
        }

        let search_range = if self.ef_search > k {
            self.ef_search
        } else {
            k
        };

        top_candidate = self.search_layer(current_id, query_vector, 0, search_range);
        while top_candidate.len() > k {
            top_candidate.pop();
        }
        top_candidate
    }

    fn search_layer(
        &self,
        root: usize,
        query_vector: &SparseVector,
        level: usize,
        ef: usize,
    ) -> BinaryHeap<NeighborNode> {
        let mut visited = HashSet::new();
        let mut top_candidates = BinaryHeap::new();
        let mut candidates = BinaryHeap::new();

        let distance = self.vectors[root].euclidean_distance(query_vector);
        top_candidates.push(NeighborNode::new(root, distance));
        candidates.push(NeighborNode::new(root, -distance));
        let mut lower_bound = distance;

        visited.insert(root);

        while !candidates.is_empty() {
            let current_neighbor = candidates.peek().unwrap();
            let current_distance = -current_neighbor.distance;
            let current_id = current_neighbor.id;
            candidates.pop();
            if current_distance.into_inner() > lower_bound {
                break;
            }

            let current_neighbors = self.get_neighbor(current_id, level).read().unwrap();
            current_neighbors.iter().for_each(|neigh| {
                if visited.contains(neigh) {
                    return;
                }
                visited.insert(*neigh);
                let distance = self.vectors[*neigh].euclidean_distance(query_vector);
                if top_candidates.len() < ef || distance < lower_bound {
                    candidates.push(NeighborNode::new(*neigh, -distance));
                    top_candidates.push(NeighborNode::new(*neigh, distance));

                    if top_candidates.len() > ef {
                        top_candidates.pop();
                    }

                    if !top_candidates.is_empty() {
                        lower_bound = top_candidates.peek().unwrap().distance.into_inner();
                    }
                }
            });
        }

        top_candidates
    }

    pub(super) fn search_layer_with_candidate(
        &self,
        search_data: &SparseVector,
        sorted_candidates: &[NeighborNode],
        visited: &mut HashSet<usize>,
        level: usize,
    ) -> BinaryHeap<NeighborNode> {
        let mut candidates = BinaryHeap::new();
        let mut top_candidates = BinaryHeap::new();
        for neighbor in sorted_candidates.iter() {
            let root = neighbor.id;
            let distance = search_data.euclidean_distance(&self.vectors[root]);
            top_candidates.push(NeighborNode::new(root, distance));
            candidates.push(NeighborNode::new(root, -distance));
            visited.insert(root);
        }

        let mut lower_bound = if top_candidates.is_empty() {
            OrderedFloat(f32::MAX)
        } else {
            top_candidates.peek().unwrap().distance
        };

        while !candidates.is_empty() {
            let current_neighbor = candidates.peek().unwrap();
            let current_distance = -current_neighbor.distance;
            let current_id = current_neighbor.id;
            candidates.pop();
            if current_distance > lower_bound {
                break;
            }
            let current_neighbors = self.get_neighbor(current_id, level).read().unwrap();
            current_neighbors.iter().for_each(|neighbor| {
                if visited.contains(neighbor) {
                    return;
                }

                visited.insert(*neighbor);
                let distance = search_data.euclidean_distance(&self.vectors[*neighbor]);
                if top_candidates.len() < self.ef_construction
                    || distance < lower_bound.into_inner()
                {
                    candidates.push(NeighborNode::new(*neighbor, -distance));
                    top_candidates.push(NeighborNode::new(*neighbor, distance));

                    if top_candidates.len() > self.ef_construction {
                        top_candidates.pop();
                    }

                    if !top_candidates.is_empty() {
                        lower_bound = top_candidates.peek().unwrap().distance;
                    }
                }
            });
        }

        top_candidates
    }
}
