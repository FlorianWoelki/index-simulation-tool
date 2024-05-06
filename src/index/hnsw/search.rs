use std::collections::{BinaryHeap, HashSet};

use ordered_float::OrderedFloat;

use crate::data::HighDimVector;

use super::{neighbor::NeighborNode, HNSWIndex};

impl HNSWIndex {
    /// Conducts a k-nearest neighbors search using a query vector, returning a binary heap
    /// of neighbors sorted by their distance to the query vector.
    ///
    /// # Arguments
    ///
    /// * `query_vector` - The vector to find neighbors for.
    /// * `k` - The number of nearest neighbors to retrieve.
    ///
    /// # Returns
    /// A `BinaryHeap` containing up to `k` closest neighbors.
    pub(super) fn search_knn(
        &self,
        query_vector: &HighDimVector,
        k: usize,
    ) -> BinaryHeap<NeighborNode> {
        let mut top_candidate = BinaryHeap::new();
        if self.n_indexed_vectors == 0 {
            return top_candidate;
        }
        let mut current_id = self.root_node_id;
        let mut current_distance = self.vectors[current_id].distance(query_vector, self.metric);
        let mut current_level = self.current_level;
        loop {
            let mut changed = true;
            while changed {
                changed = false;
                let current_neighbors =
                    self.get_neighbor(current_id, current_level).read().unwrap();
                for neighbor in current_neighbors.iter() {
                    let distance = self.vectors[current_id].distance(query_vector, self.metric);
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

    /// Searches within a specific layer of the index starting from a root node to find
    /// the nearest neighbors within a range defined by `ef`.
    ///
    /// # Arguments
    ///
    /// * `root` - Starting point for the search.
    /// * `query_vector` - The vector to find neighbors for.
    /// * `level` - The layer of the graph to perform the search.
    /// * `ef` - The size of the dynamic candidate list.
    ///
    /// # Returns
    /// A `BinaryHeap` containing the nearest neighbors found during the search.
    fn search_layer(
        &self,
        root: usize,
        query_vector: &HighDimVector,
        level: usize,
        ef: usize,
    ) -> BinaryHeap<NeighborNode> {
        let mut visited = HashSet::new();
        let mut top_candidates = BinaryHeap::new();
        let mut candidates = BinaryHeap::new();

        let distance = self.vectors[root].distance(query_vector, self.metric);
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
                let distance = self.vectors[*neigh].distance(query_vector, self.metric);
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
        search_data: &HighDimVector,
        sorted_candidates: &Vec<NeighborNode>,
        visited: &mut HashSet<usize>,
        level: usize,
    ) -> BinaryHeap<NeighborNode> {
        let mut candidates = BinaryHeap::new();
        let mut top_candidates = BinaryHeap::new();
        for neighbor in sorted_candidates.iter() {
            let root = neighbor.id;
            let distance = search_data.distance(&self.vectors[root], self.metric);
            top_candidates.push(NeighborNode::new(root, distance));
            candidates.push(NeighborNode::new(root, -distance));
            visited.insert(root);
        }

        let mut lower_bound = if top_candidates.is_empty() {
            OrderedFloat(f64::MAX)
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
                let distance = search_data.distance(&self.vectors[*neighbor], self.metric);
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

#[cfg(test)]
mod tests {
    use crate::index::{DistanceMetric, Index};

    use super::*;

    #[test]
    fn test_search_knn_basic_functionality() {
        let mut index = HNSWIndex::new(DistanceMetric::Euclidean);
        let v0 = HighDimVector::new(0, vec![1.0, 2.0, 3.0]);
        let v1 = HighDimVector::new(1, vec![4.0, 5.0, 6.0]);
        let v2 = HighDimVector::new(2, vec![7.0, 8.0, 9.0]);

        index.add_vector(v0);
        index.add_vector(v1);
        index.add_vector(v2);
        index.build();

        let query_vector = HighDimVector::new(3, vec![1.5, 2.5, 3.5]);
        let result = index.search_knn(&query_vector, 2);

        assert_eq!(result.len(), 2);

        let mut result_vec = result.into_sorted_vec();
        assert_eq!(result_vec.pop().unwrap().id, 1);
        assert_eq!(result_vec.pop().unwrap().id, 0);
    }

    #[test]
    fn test_search_knn_boundary_conditions() {
        let mut index = HNSWIndex::new(DistanceMetric::Euclidean);
        let v0 = HighDimVector::new(0, vec![1.0, 2.0, 3.0]);
        index.add_vector(v0);
        index.build();

        let query_vector = HighDimVector::new(1, vec![1.5, 2.5, 3.5]);

        let result_zero = index.search_knn(&query_vector, 0);
        assert_eq!(result_zero.len(), 0);

        let result_excess = index.search_knn(&query_vector, 10);
        assert_eq!(result_excess.len(), 1);
    }

    #[test]
    fn test_search_layer() {
        let mut index = HNSWIndex::new(DistanceMetric::Euclidean);
        let v0 = HighDimVector::new(0, vec![1.0, 2.0, 3.0]);
        let v1 = HighDimVector::new(1, vec![4.0, 5.0, 6.0]);
        let v2 = HighDimVector::new(2, vec![7.0, 8.0, 9.0]);

        index.add_vector(v0);
        index.add_vector(v1);
        index.add_vector(v2);
        index.build();

        let query_vector = HighDimVector::new(3, vec![1.5, 2.5, 3.5]);
        let ef = 2;
        let result = index.search_layer(0, &query_vector, 0, ef);

        assert_eq!(result.len(), ef);
        assert!(result.iter().all(|n| [0, 1].contains(&n.id)));
    }

    #[test]
    fn test_search_layer_with_candidate() {
        let mut index = HNSWIndex::new(DistanceMetric::Euclidean);
        let v0 = HighDimVector::new(0, vec![1.0, 2.0, 3.0]);
        let v1 = HighDimVector::new(1, vec![4.0, 5.0, 6.0]);
        let v2 = HighDimVector::new(2, vec![7.0, 8.0, 9.0]);

        index.add_vector(v0);
        index.add_vector(v1);
        index.add_vector(v2);
        index.build();

        let sorted_candidates = vec![NeighborNode::new(0, 5.0), NeighborNode::new(1, 6.0)];
        let mut visited = HashSet::new();
        let query_vector = HighDimVector::new(3, vec![1.5, 2.5, 3.5]);
        let result =
            index.search_layer_with_candidate(&query_vector, &sorted_candidates, &mut visited, 0);

        assert_eq!(result.len(), 3);
        assert!(visited.contains(&1) && visited.contains(&2));
    }
}
