use std::{
    collections::HashSet,
    fs::File,
    io::{BufReader, BufWriter},
    sync::Mutex,
};

use ordered_float::OrderedFloat;
use rand::Rng;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    data::{QueryResult, SparseVector},
    data_structures::min_heap::MinHeap,
};

use super::{DistanceMetric, SparseIndex};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Node {
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    indices: Vec<usize>,
}

impl Node {
    fn build(
        vectors: &Vec<SparseVector>,
        indices: &[usize],
        max_points: usize,
        metric: &DistanceMetric,
    ) -> Self {
        if indices.len() <= max_points {
            // Base case: create a leaf node.
            return Node {
                left: None,
                right: None,
                indices: indices.to_vec(),
            };
        }

        // Randomly select two pivot points.
        let mut rng = rand::thread_rng();
        let pivot_idx1 = indices[rng.gen_range(0..indices.len())];
        let pivot_idx2 = indices[rng.gen_range(0..indices.len())];

        let pivot1 = vectors[pivot_idx1].clone();
        let pivot2 = vectors[pivot_idx2].clone();

        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        // Split indices into left and right based on distance to pivots.
        for &i in indices {
            if vectors[i].distance(&pivot1, metric) < vectors[i].distance(&pivot2, metric) {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        let left = if !left_indices.is_empty() {
            Some(Box::new(Node::build(
                vectors,
                &left_indices,
                max_points,
                metric,
            )))
        } else {
            None
        };

        let right = if !right_indices.is_empty() {
            Some(Box::new(Node::build(
                vectors,
                &right_indices,
                max_points,
                metric,
            )))
        } else {
            None
        };

        Node {
            left,
            right,
            indices: indices.to_vec(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Tree {
    root: Node,
}

#[derive(Serialize, Deserialize)]
pub struct AnnoyIndex {
    trees: Vec<Tree>,
    vectors: Vec<SparseVector>,
    /// Controls the number of trees to be constructed in the index.
    /// Increasing this value improves search accuracy at the cost of longer
    /// build times and more memory usage.
    n_trees: usize,
    /// The maximum number of points allowed in a leaf node.
    /// Lower values create deeper trees, potentially improving accuracy
    /// but increasing search time.
    /// Higher values create shallower trees, which can be faster to search
    /// but may reduce accuracy.
    max_points: usize,
    /// The number of nodes to explore during the search phase.
    /// Higher values increase search accuracy but also increase query time.
    /// Lower values result in faster searches but may reduce accuracy.
    search_k: usize,
    metric: DistanceMetric,
}

impl AnnoyIndex {
    pub fn new(n_trees: usize, max_points: usize, search_k: usize, metric: DistanceMetric) -> Self {
        AnnoyIndex {
            trees: Vec::new(),
            vectors: Vec::new(),
            n_trees,
            max_points,
            search_k,
            metric,
        }
    }
}

impl SparseIndex for AnnoyIndex {
    fn add_vector_before_build(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
    }

    fn add_vector(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
        self.build();
    }

    fn remove_vector(&mut self, id: usize) -> Option<SparseVector> {
        if id >= self.vectors.len() {
            return None;
        }

        let removed_vector = self.vectors.remove(id);
        self.build();
        Some(removed_vector)
    }

    fn build(&mut self) {
        let trees: Vec<Tree> = (0..self.n_trees)
            .into_par_iter()
            .map(|_| {
                let indices: Vec<usize> = (0..self.vectors.len()).collect();
                let root = Node::build(&self.vectors, &indices, self.max_points, &self.metric);
                Tree { root }
            })
            .collect();

        self.trees = trees;
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let candidates = Mutex::new(HashSet::new());

        self.trees.par_iter().for_each(|tree| {
            let mut local_candidates: HashSet<usize> = HashSet::new();
            let mut nodes = vec![(0.0, &tree.root)];
            let mut visited = 0;

            while let Some((_, node)) = nodes.pop() {
                visited += 1;
                if visited > self.search_k {
                    break;
                }

                local_candidates.extend(&node.indices);

                if let Some(ref left) = node.left {
                    nodes.push((0.0, left));
                }
                if let Some(ref right) = node.right {
                    nodes.push((0.0, right));
                }
            }

            let mut global_candidates = candidates.lock().unwrap();
            global_candidates.extend(local_candidates);
        });

        let candidates: Vec<_> = candidates.into_inner().unwrap().into_iter().collect();

        let results: Vec<_> = candidates
            .into_par_iter()
            .map(|point| {
                let distance =
                    OrderedFloat(query_vector.distance(&self.vectors[point], &self.metric));
                (distance, point)
            })
            .collect();

        let mut heap: MinHeap<QueryResult> = MinHeap::new();
        // Evaluate distance for each candidate and maintain max-heap.
        for (distance, point) in results {
            if heap.len() < k || distance > heap.peek().unwrap().score {
                heap.push(
                    QueryResult {
                        index: point,
                        score: -distance,
                    },
                    -distance,
                );
                if heap.len() > k {
                    heap.pop();
                }
            }
        }

        heap.into_sorted_vec()
            .iter()
            .map(|query_result| QueryResult {
                index: query_result.index,
                score: OrderedFloat(-query_result.score.into_inner()),
            })
            .collect()
    }

    fn save(&self, file: &mut File) {
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &self).expect("Failed to serialize");
    }

    fn load(&self, file: &File) -> Self {
        let reader = BufReader::new(file);
        bincode::deserialize_from(reader).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{get_complex_vectors, get_simple_vectors, is_in_actual_result};

    use super::*;

    #[test]
    fn test_serde() {
        let (data, _) = get_simple_vectors();
        let mut index = AnnoyIndex::new(3, 2, 10, DistanceMetric::Cosine);
        for vector in &data {
            index.add_vector_before_build(vector);
        }

        let bytes = bincode::serialize(&index).unwrap();
        let reconstructed: AnnoyIndex = bincode::deserialize(&bytes).unwrap();

        assert_eq!(index.vectors, reconstructed.vectors);
        assert_eq!(index.metric, reconstructed.metric);
        assert_eq!(index.trees, reconstructed.trees);
        assert_eq!(index.n_trees, reconstructed.n_trees);
        assert_eq!(index.search_k, reconstructed.search_k);
        assert_eq!(index.max_points, reconstructed.max_points);
    }

    #[test]
    fn test_remove_vector() {
        let mut index = AnnoyIndex::new(3, 2, 10, DistanceMetric::Euclidean);

        let (vectors, query_vectors) = get_simple_vectors();
        for vector in &vectors {
            index.add_vector_before_build(vector);
        }

        index.build();

        assert_eq!(index.vectors.len(), vectors.len());

        index.remove_vector(2);

        assert_eq!(index.vectors.len(), vectors.len() - 1);
        assert_eq!(index.vectors[0], vectors[0]);
        assert_eq!(index.vectors[2], vectors[3]);

        let results = index.search(&query_vectors[0], 2);
        assert_eq!(results[0].index, 3);
        assert_eq!(results[1].index, 0);
    }

    #[test]
    fn test_annoy_index_simple() {
        let (data, query_vectors) = get_simple_vectors();
        let mut index = AnnoyIndex::new(10, 10, 10, DistanceMetric::Cosine);
        for vector in &data {
            index.add_vector_before_build(vector);
        }
        index.build();

        let results = index.search(&query_vectors[0], 2);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
    }

    #[test]
    fn test_annoy_index_complex() {
        let mut index = AnnoyIndex::new(10, 10, 10, DistanceMetric::Cosine);

        let (data, query_vector) = get_complex_vectors();
        for vector in &data {
            index.add_vector_before_build(vector);
        }

        index.build();

        let results = index.search(&query_vector, 2);
        assert!(is_in_actual_result(&data, &query_vector, &results));
    }
}
