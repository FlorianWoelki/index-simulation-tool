use std::collections::{BinaryHeap, HashSet, VecDeque};

use crate::{
    data::{QueryResult, SparseVector},
    kmeans::kmeans_index,
};

use super::{neighbor::NeighborNode, SparseIndex};

mod construction;
mod prune;

pub struct SSGIndex {
    /// The vectors to be indexed.
    pub(super) vectors: Vec<SparseVector>,
    /// The threshold value used for occlusion checking.
    pub(super) occlusion_threshold: f32,
    /// The size of the pool of neighbors to consider for each vector.
    pub(super) pool_size: usize,
    /// The graph of k-nearest neighbors for each vector.
    pub(super) graph: Vec<Vec<usize>>,
    /// The number of clusters to use for k-means clustering.
    pub(super) num_clusters: usize,
    /// The indices of the root nodes in the graph.
    pub(super) root_nodes: Vec<usize>,
    /// The size of the expanded neighbor set to consider for each vector.
    pub(super) expanded_neighbor_size: usize,
    /// The initial value of k to use for constructing the k-nearest neighbors graph.
    pub(super) initial_k: usize,
    /// The number of iterations to use for k-means clustering.
    pub(super) iterations: usize,
    pub(super) random_seed: u64,
}

impl SSGIndex {
    fn new(num_clusters: usize, iterations: usize, random_seed: u64) -> Self {
        SSGIndex {
            vectors: Vec::new(),
            occlusion_threshold: 60.0,
            pool_size: 100,
            initial_k: 100,
            graph: Vec::new(),
            expanded_neighbor_size: 100,
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
        self.construct_knn_graph(self.initial_k);
        self.prune_and_link_graph();

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
        let mut initial_nodes: Vec<_> = self
            .root_nodes
            .iter()
            .map(|&n| {
                let distance = query_vector.euclidean_distance(&self.vectors[n]);
                NeighborNode::new(n, distance)
            })
            .collect();
        initial_nodes.sort();

        // Add the k closest root nodes to the heap to initialize the search.
        for node in &initial_nodes {
            if heap.len() < k {
                heap.push(node.clone());
                search_queue.push_back(node.id);
            }
            visited.insert(node.id);
        }

        while let Some(id) = search_queue.pop_front() {
            if let Some(node_vec) = self.graph.get(id) {
                for &neighbor_id in node_vec {
                    if visited.insert(neighbor_id) {
                        let distance = query_vector.euclidean_distance(&self.vectors[neighbor_id]);
                        let neighbor_node = NeighborNode::new(neighbor_id, distance);
                        heap.push(neighbor_node);
                        search_queue.push_back(neighbor_id);
                    }
                }
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

        assert!(false);
    }
}
