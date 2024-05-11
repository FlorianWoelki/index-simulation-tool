use std::collections::{HashSet, VecDeque};

use rand::thread_rng;

use crate::{data::HighDimVector, kmeans::kmeans};

use super::{neighbor::NeighborNode, DistanceMetric, Index};

mod construction;
mod prune;
mod search;

pub struct SSGIndex {
    pub(super) vectors: Vec<HighDimVector>,
    pub(super) metric: DistanceMetric,
    pub(super) threshold: f64,
    pub(super) index_size: usize,
    pub(super) graph: Vec<Vec<usize>>,
    pub(super) root_size: usize,
    pub(super) root_nodes: Vec<usize>,
    pub(super) neighbor_neighbor_size: usize,
}

impl Index for SSGIndex {
    fn new(metric: super::DistanceMetric) -> Self
    where
        Self: Sized,
    {
        SSGIndex {
            vectors: Vec::new(),
            metric,
            threshold: 0.0,
            index_size: 100,
            graph: Vec::new(),
            root_size: 30,
            neighbor_neighbor_size: 100,
            root_nodes: Vec::new(),
        }
    }

    fn add_vector(&mut self, vector: HighDimVector) {
        self.vectors.push(vector);
    }

    fn build(&mut self) {
        self.construct_knn_graph(100);

        let len = self.vectors.len() * self.index_size;
        let mut pruned_graph: Vec<NeighborNode> = Vec::with_capacity(len);
        for i in 0..len {
            pruned_graph.push(NeighborNode::new(i, 0.0));
        }
        self.link_each_nodes(&mut pruned_graph);

        for i in 0..self.vectors.len() {
            let offset = i * self.index_size;
            let pool_size = (0..self.index_size)
                .take_while(|j| {
                    return pruned_graph[offset + j].distance.into_inner() == f64::MAX;
                })
                .count()
                .max(1);
            self.graph[i] = (0..pool_size)
                .map(|j| pruned_graph[offset + j].id)
                .collect();
        }

        self.ensure_graph_connectivity();

        // init root nodes
        self.root_nodes = kmeans(
            self.root_size,
            256,
            &self.vectors,
            self.metric,
            &mut thread_rng(),
        );
    }

    fn search(&self, query_vector: &HighDimVector, k: usize) -> Vec<HighDimVector> {
        self.search_bfs(query_vector, k)
    }
}

impl SSGIndex {
    /// Populates the `expanded_neighbors` vector with neighbor nodes of the given `query_point`.
    /// This function explores neighbors of the query_point's immediate neighbors to find
    /// distinct second-level neighbors, avoiding self-loops and repeated nodes. The process stops
    /// once the specified number of neighbors (`neighbor_neighbor_size`) has been collected.
    ///
    /// # Arguments
    ///
    /// * `query_point` - The index of the query point in the graph whose expanded neighbors are to be found.
    /// * `expand_neighbors` - A mutable reference to a vector where the found neighbors will be stored.
    fn populate_expanded_neighbors(
        &self,
        query_point: usize,
        expand_neighbors: &mut Vec<NeighborNode>,
    ) {
        let mut visited = HashSet::with_capacity(self.neighbor_neighbor_size);
        visited.insert(query_point);

        for neighbor_id in self.graph[query_point].iter() {
            if *neighbor_id == query_point {
                continue;
            }

            // Iterate over the neighbors of the current neighbors.
            for second_neighbor_id in self.graph[*neighbor_id].iter() {
                if *second_neighbor_id == query_point || *neighbor_id == *second_neighbor_id {
                    continue;
                }

                if visited.insert(*second_neighbor_id) {
                    let distance = self.vectors[query_point]
                        .distance(&self.vectors[*second_neighbor_id], self.metric);
                    expand_neighbors.push(NeighborNode::new(*second_neighbor_id, distance));

                    if expand_neighbors.len() >= self.neighbor_neighbor_size {
                        return;
                    }
                }
            }
        }
    }

    /// Links each node in the graph with its neighbors after pruning.
    /// This function consolidates neighbor connections to optimize the graph based
    /// on the pruning strategy.
    ///
    /// # Arguments
    ///
    /// * `pruned_graph_tmp` - The pruned graph.
    fn link_each_nodes(&mut self, pruned_graph: &mut Vec<NeighborNode>) {
        let mut expanded_neighbors = Vec::new();

        for i in 0..self.vectors.len() {
            expanded_neighbors.clear();
            self.populate_expanded_neighbors(i, &mut expanded_neighbors);
            self.prune_graph(i, &mut expanded_neighbors, pruned_graph);
        }

        for i in 0..self.vectors.len() {
            self.interconnect_pruned_neighbors(i, self.index_size, pruned_graph);
        }
    }

    /// Expands the connectivity of the graph to ensure all nodes are at least loosely connected.
    /// This helps in improving the reachability and robustness of the search algorithm.
    fn ensure_graph_connectivity(&mut self) {
        self.initialize_root_nodes();

        for &root_id in &self.root_nodes {
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back(root_id);
            visited.insert(root_id);

            let mut unknown_set: Vec<usize> = Vec::with_capacity(1);
            while !unknown_set.is_empty() {
                while !queue.is_empty() {
                    let current_node = queue.pop_front().unwrap();

                    for j in 0..self.graph[current_node].len() {
                        let adjacent_node = self.graph[current_node][j];
                        if visited.contains(&adjacent_node) {
                            continue;
                        }

                        visited.insert(adjacent_node);
                        queue.push_back(adjacent_node);
                    }
                }

                unknown_set.clear();
                for j in 0..self.vectors.len() {
                    if visited.contains(&j) {
                        continue;
                    }
                    unknown_set.push(j);
                }
                if !unknown_set.is_empty() {
                    for j in 0..self.vectors.len() {
                        if visited.contains(&j) && self.graph[j].len() < self.index_size {
                            self.graph[j].push(unknown_set[0]);
                            break;
                        }
                    }

                    queue.push_back(unknown_set[0]);
                    visited.insert(unknown_set[0]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensure_all_nodes_connected() {
        let mut index = SSGIndex::new(DistanceMetric::Euclidean);
        index.add_vector(HighDimVector::new(0, vec![1.0]));
        index.add_vector(HighDimVector::new(1, vec![2.0]));
        index.add_vector(HighDimVector::new(2, vec![3.0]));
        index.construct_knn_graph(100);
        index.ensure_graph_connectivity();

        let all_connected = index.graph.iter().all(|neighbors| !neighbors.is_empty());
        assert!(all_connected);
    }

    #[test]
    fn test_ensure_graph_connectivity_through_root_nodes() {
        let mut index = SSGIndex::new(DistanceMetric::Euclidean);
        for i in 0..10 {
            index.add_vector(HighDimVector::new(i, vec![i as f64]));
        }
        index.construct_knn_graph(100);
        index.root_nodes = vec![0, 5]; // Manually setting root nodes for predictability.
        index.ensure_graph_connectivity();

        let connected_to_root = |node: usize| -> bool {
            index.graph[node].contains(&0) || index.graph[node].contains(&5)
        };
        let all_connected_to_root = (0..index.vectors.len()).all(connected_to_root);
        assert!(all_connected_to_root);
    }

    #[test]
    fn test_no_extra_connections_added() {
        let mut index = SSGIndex::new(DistanceMetric::Euclidean);
        index.add_vector(HighDimVector::new(0, vec![1.0]));
        index.add_vector(HighDimVector::new(1, vec![2.0]));
        // Create a fully connected graph.
        index.graph = vec![vec![1], vec![0]];
        index.root_nodes = vec![0];

        index.ensure_graph_connectivity();
        assert_eq!(index.graph[0], vec![1]);
        assert_eq!(index.graph[1], vec![0]);
    }

    #[test]
    fn test_link_each_nodes() {
        let mut index = SSGIndex::new(DistanceMetric::Euclidean);
        for i in 0..10 {
            index.add_vector(HighDimVector::new(i, vec![i as f64]));
        }
        index.construct_knn_graph(100);

        let mut pruned_graph = vec![];
        let len = index.vectors.len() * index.index_size;
        (0..len).for_each(|i| {
            pruned_graph.push(NeighborNode::new(i, 0.0));
        });
        index.link_each_nodes(&mut pruned_graph);

        for i in 0..index.vectors.len() {
            assert!(index.graph[i].len() <= index.index_size);
            for &neighbor in &index.graph[i] {
                assert_ne!(i, neighbor);
            }
        }
    }

    #[test]
    fn test_populate_expanded_neighbors() {
        let mut index = SSGIndex::new(DistanceMetric::Euclidean);
        let mut expanded_neighbors = Vec::new();
        let query_point = 0;
        for i in 0..5 {
            index.add_vector(HighDimVector::new(i, vec![i as f64 * 10.0]));
        }
        index.graph = vec![vec![1, 2], vec![0, 3], vec![0, 4], vec![1], vec![2]];

        index.populate_expanded_neighbors(query_point, &mut expanded_neighbors);

        let expected_ids: Vec<usize> = vec![3, 4];
        let result_ids: Vec<usize> = expanded_neighbors.into_iter().map(|n| n.id).collect();

        assert_eq!(
            result_ids.len(),
            expected_ids.len(),
            "Unexpected number of neighbors populated."
        );
        assert!(
            result_ids.contains(&3),
            "Node 3 should be a second-level neighbor of node 0."
        );
        assert!(
            result_ids.contains(&4),
            "Node 4 should be a second-level neighbor of node 0."
        );
    }

    #[test]
    fn test_build() {
        let mut index = SSGIndex::new(DistanceMetric::Euclidean);
        for i in 0..10 {
            index.add_vector(HighDimVector::new(i, vec![i as f64]));
        }
        index.build();

        assert!(false);
    }
}
