use std::collections::{BinaryHeap, HashSet, VecDeque};

use crate::{
    data::{QueryResult, SparseVector},
    kmeans::kmeans_index,
};

use super::{neighbor::NeighborNode, SparseIndex};

mod construction;
mod prune;

pub struct SSGIndex {
    pub(super) vectors: Vec<SparseVector>,
    pub(super) threshold: f32,
    pub(super) index_size: usize,
    pub(super) graph: Vec<Vec<usize>>,
    pub(super) num_clusters: usize,
    pub(super) root_nodes: Vec<usize>,
    pub(super) neighbor_neighbor_size: usize,
    pub(super) init_k: usize,
    pub(super) iterations: usize,
    pub(super) random_seed: u64,
}

impl SSGIndex {
    fn new(num_clusters: usize, iterations: usize, random_seed: u64) -> Self {
        SSGIndex {
            vectors: Vec::new(),
            threshold: 60.0,
            index_size: 100,
            init_k: 100,
            graph: Vec::new(),
            neighbor_neighbor_size: 100,
            root_nodes: Vec::new(),
            iterations,
            num_clusters,
            random_seed,
        }
    }

    fn add_vector(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
    }

    fn build(&mut self) {
        self.construct_knn_graph(self.init_k);

        let len = self.vectors.len() * self.index_size;
        let mut pruned_graph = (0..len)
            .map(|i| NeighborNode::new(i, 0.0))
            .collect::<Vec<NeighborNode>>();
        self.link_each_nodes(&mut pruned_graph);

        (0..self.vectors.len()).enumerate().for_each(|(i, _)| {
            let offset = i * self.index_size;
            let pool_size = (0..self.index_size)
                .take_while(|j| pruned_graph[offset + j].distance.into_inner() == f32::MAX)
                .count()
                .max(1);
            self.graph[i] = (0..pool_size)
                .map(|j| pruned_graph[offset + j].id)
                .collect();
        });

        self.root_nodes = kmeans_index(
            &self.vectors,
            self.num_clusters,
            self.iterations,
            0.01, // TODO: Change this to an argument
            self.random_seed,
        );
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let mut visited = HashSet::new();
        let mut heap = BinaryHeap::new();
        let mut search_queue = VecDeque::new();

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
            result.push(QueryResult {
                index: node.id,
                score: node.distance,
            });
        }

        result.reverse();
        result
    }
}

impl SSGIndex {
    fn populate_expanded_neighbors(
        &self,
        query_point: usize,
        expand_neighbors: &mut Vec<NeighborNode>,
    ) {
        let mut visited = HashSet::with_capacity(self.neighbor_neighbor_size);
        visited.insert(query_point);

        for neighbor_id in self.graph[query_point].iter() {
            self.graph[*neighbor_id]
                .iter()
                .filter(|node| **node != query_point && *neighbor_id != **node)
                .for_each(|second_neighbor_id| {
                    if visited.insert(*second_neighbor_id) {
                        let distance = self.vectors[query_point]
                            .euclidean_distance(&self.vectors[*second_neighbor_id]);
                        expand_neighbors.push(NeighborNode::new(*second_neighbor_id, distance));

                        // if expand_neighbors.len() >= self.neighbor_neighbor_size {
                        //     return;
                        // }
                    }
                });
        }
    }

    fn link_each_nodes(&mut self, pruned_graph: &mut [NeighborNode]) {
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
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;
    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn test_ssg_index_simple() {
        let random_seed = 42;
        let num_clusters = 5;
        let iterations = 256;
        let mut index = SSGIndex::new(num_clusters, iterations, random_seed);

        let vectors = vec![
            SparseVector {
                indices: vec![0, 1, 2],
                values: vec![OrderedFloat(0.1), OrderedFloat(0.2), OrderedFloat(0.3)],
            },
            SparseVector {
                indices: vec![1, 2, 3],
                values: vec![OrderedFloat(0.4), OrderedFloat(0.5), OrderedFloat(0.6)],
            },
            SparseVector {
                indices: vec![2, 3, 4],
                values: vec![OrderedFloat(0.7), OrderedFloat(0.8), OrderedFloat(0.9)],
            },
            SparseVector {
                indices: vec![3, 4, 5],
                values: vec![OrderedFloat(1.0), OrderedFloat(1.1), OrderedFloat(1.2)],
            },
            SparseVector {
                indices: vec![4, 5, 6],
                values: vec![OrderedFloat(1.3), OrderedFloat(1.4), OrderedFloat(1.5)],
            },
        ];

        for vector in &vectors {
            index.add_vector(vector);
        }

        index.build();

        let results = index.search(&vectors[0], 3);

        println!("{:?}", results);
        assert!(true);
    }

    #[test]
    fn test_ssg_index_complex() {
        let random_seed = thread_rng().gen::<u64>();
        let num_clusters = 40;
        let iterations = 256;
        let mut index = SSGIndex::new(num_clusters, iterations, random_seed);

        let mut vectors = vec![];
        for i in 0..100 {
            vectors.push(SparseVector {
                indices: vec![i % 10, (i / 10) % 10],
                values: vec![OrderedFloat((i % 10) as f32), OrderedFloat((i / 10) as f32)],
            });
        }

        for vector in &vectors {
            index.add_vector(vector);
        }

        index.build();

        let query_vector = SparseVector {
            indices: vec![5, 9],
            values: vec![OrderedFloat(5.0), OrderedFloat(9.0)],
        };
        let results = index.search(&query_vector, 10);
        println!("Results for search on query vector: {:?}", results);
        println!("Top Search: {:?}", vectors[results[0].index]);
        println!("Groundtruth: {:?}", query_vector);

        assert!(true);
    }
}
