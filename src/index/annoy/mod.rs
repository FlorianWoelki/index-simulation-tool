use std::collections::{BinaryHeap, HashSet};

use ordered_float::OrderedFloat;
use rand::Rng;

use crate::data::{QueryResult, SparseVector};

use super::DistanceMetric;

struct Node {
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    indices: Vec<usize>,
}

impl Node {
    fn build(
        vectors: &Vec<SparseVector>,
        indices: &[usize],
        n_dims: usize,
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
                n_dims,
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
                n_dims,
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

struct Tree {
    root: Node,
    n_dims: usize,
}

pub struct AnnoyIndex {
    trees: Vec<Tree>,
    vectors: Vec<SparseVector>,
    n_trees: usize,    // Larger values means larger index but better accuracy
    max_points: usize, // Maximum number of points in a leaf node
    search_k: usize, // Larger value will give more accurate results, but will take longer to run (k * n_trees where k is the amount of ANN)
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

    pub fn add_vector(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
    }

    /// Needs rebuilding after removing vector.
    pub fn remove_vector(&mut self, id: usize) -> Option<SparseVector> {
        if id >= self.vectors.len() {
            return None;
        }

        let removed_vector = self.vectors.remove(id);
        Some(removed_vector)
    }

    pub fn build(&mut self) {
        let n_dims = self
            .vectors
            .iter()
            .map(|v| v.indices.len())
            .max()
            .unwrap_or(0);
        let mut trees = Vec::with_capacity(self.n_trees);

        for _ in 0..self.n_trees {
            let indices: Vec<usize> = (0..self.vectors.len()).collect();
            let root = Node::build(
                &self.vectors,
                &indices,
                n_dims,
                self.max_points,
                &self.metric,
            );
            trees.push(Tree { root, n_dims });
        }

        self.trees = trees;
    }

    pub fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let mut heap = BinaryHeap::with_capacity(k);
        let mut candidates = HashSet::new();

        // Traverse each tree to collect candidate points.
        for tree in &self.trees {
            let mut nodes = vec![(0.0, &tree.root)];
            let mut visited = 0;

            while let Some((_, node)) = nodes.pop() {
                visited += 1;
                if visited > self.search_k {
                    break;
                }
                candidates.extend(&node.indices);

                if let Some(ref left) = node.left {
                    nodes.push((0.0, left));
                }
                if let Some(ref right) = node.right {
                    nodes.push((0.0, right));
                }
            }
        }

        let candidates: Vec<_> = candidates.into_iter().collect();

        // Evaluate distance for each candidate and maintain max-heap.
        for point in candidates {
            let distance = OrderedFloat(query_vector.distance(&self.vectors[point], &self.metric));
            if heap.len() < k {
                heap.push((distance, point));
            } else if distance < heap.peek().unwrap().0 {
                heap.pop();
                heap.push((distance, point));
            }
        }

        let mut results = heap.into_iter().collect::<Vec<_>>();
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        results
            .into_iter()
            .map(|(distance, i)| QueryResult {
                index: i,
                score: distance,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::get_simple_vectors;

    use super::*;

    #[test]
    fn test_remove_vector() {
        let mut index = AnnoyIndex::new(3, 2, 10, DistanceMetric::Euclidean);

        let (vectors, query_vectors) = get_simple_vectors();
        for vector in &vectors {
            index.add_vector(vector);
        }

        index.build();

        assert_eq!(index.vectors.len(), vectors.len());

        index.remove_vector(2);

        assert_eq!(index.vectors.len(), vectors.len() - 1);
        assert_eq!(index.vectors[0], vectors[0]);
        assert_eq!(index.vectors[2], vectors[3]);

        index.build();

        let results = index.search(&query_vectors[0], 2);
        println!("{:?}", results);
        assert!(true);
    }

    #[test]
    fn test_annoy_index_simple() {
        let data = vec![
            SparseVector {
                indices: vec![0, 2],
                values: vec![OrderedFloat(1.0), OrderedFloat(2.0)],
            },
            SparseVector {
                indices: vec![1, 3],
                values: vec![OrderedFloat(3.0), OrderedFloat(4.0)],
            },
            SparseVector {
                indices: vec![0, 2],
                values: vec![OrderedFloat(5.0), OrderedFloat(6.0)],
            },
            SparseVector {
                indices: vec![1, 3],
                values: vec![OrderedFloat(7.0), OrderedFloat(8.0)],
            },
            SparseVector {
                indices: vec![0, 2],
                values: vec![OrderedFloat(9.0), OrderedFloat(10.0)],
            },
        ];

        let mut index = AnnoyIndex::new(10, 10, 10, DistanceMetric::Cosine);
        for vector in &data {
            index.add_vector(vector);
        }
        index.build();

        let query = SparseVector {
            indices: vec![0, 2],
            values: vec![OrderedFloat(6.0), OrderedFloat(7.0)],
        };
        let neighbors = index.search(&query, 2);
        println!("Nearest neighbors: {:?}", neighbors);
        assert!(true);
    }

    #[test]
    fn test_annoy_index_complex() {
        let mut index = AnnoyIndex::new(10, 10, 10, DistanceMetric::Cosine);

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
