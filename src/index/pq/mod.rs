use crate::{
    data::{QueryResult, SparseVector},
    data_structures::min_heap::MinHeap,
    kmeans::kmeans,
};
use ordered_float::OrderedFloat;
use std::vec;

struct PQIndex {
    num_subvectors: usize,
    num_clusters: usize,
    vectors: Vec<SparseVector>,
    codewords: Vec<Vec<SparseVector>>,
    encoded_codes: Vec<Vec<usize>>,
    iterations: usize,
    tolerance: f32,
    random_seed: u64,
}

impl PQIndex {
    pub fn new(
        num_subvectors: usize,
        num_clusters: usize,
        iterations: usize,
        tolerance: f32,
        random_seed: u64,
    ) -> Self {
        PQIndex {
            num_subvectors,
            num_clusters,
            random_seed,
            iterations,
            tolerance,
            vectors: Vec::new(),
            codewords: Vec::new(),
            encoded_codes: Vec::new(),
        }
    }

    pub fn add_vector(&mut self, vector: SparseVector) {
        self.vectors.push(vector);
    }

    pub fn build(&mut self) {
        let sub_vec_dims = self.vectors[0].indices.len() / self.num_subvectors;
        self.codewords = Vec::new();
        for m in 0..self.num_subvectors {
            let mut sub_vectors_m: Vec<SparseVector> = Vec::new();
            for vec in &self.vectors {
                let indices = vec.indices[(m * sub_vec_dims)..((m + 1) * sub_vec_dims)].to_vec();
                let values = vec.values[(m * sub_vec_dims)..((m + 1) * sub_vec_dims)].to_vec();
                sub_vectors_m.push(SparseVector { indices, values });
            }

            let codewords_m = kmeans(
                sub_vectors_m,
                self.num_clusters,
                self.iterations,
                self.tolerance,
                self.random_seed,
            );
            self.codewords.push(codewords_m);
        }

        self.encoded_codes = self.encode(&self.vectors);
    }

    pub fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let mut heap: MinHeap<QueryResult> = MinHeap::new();
        let sub_vec_dims = self.vectors[0].indices.len() / self.num_subvectors;

        let mut scores = vec![0.0; self.encoded_codes.len()];
        for (n, code) in self.encoded_codes.iter().enumerate() {
            let mut distance = 0.0;
            for m in 0..self.num_subvectors {
                let query_sub_indices =
                    query_vector.indices[m * sub_vec_dims..((m + 1) * sub_vec_dims)].to_vec();
                let query_sub_values =
                    query_vector.values[m * sub_vec_dims..((m + 1) * sub_vec_dims)].to_vec();

                let query_sub = SparseVector {
                    indices: query_sub_indices,
                    values: query_sub_values,
                };
                let sub_distance = &query_sub.euclidean_distance(&self.codewords[m][code[m]]);
                distance += sub_distance;
            }

            scores[n] += distance;
        }

        for (index, &score) in scores.iter().enumerate() {
            if heap.len() < k || score > heap.peek().unwrap().score.into_inner() {
                heap.push(
                    QueryResult {
                        index,
                        score: OrderedFloat(-score),
                    },
                    OrderedFloat(-score),
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

    fn encode(&self, vectors: &Vec<SparseVector>) -> Vec<Vec<usize>> {
        let sub_vec_dims: usize = vectors[0].indices.len() / self.num_subvectors;

        let mut vector_codes: Vec<Vec<usize>> = Vec::new();
        for vec in vectors {
            let mut subvectors: Vec<SparseVector> = Vec::new();
            for m in 0..self.num_subvectors {
                let indices = vec.indices[(m * sub_vec_dims)..((m + 1) * sub_vec_dims)].to_vec();
                let values = vec.values[(m * sub_vec_dims)..((m + 1) * sub_vec_dims)].to_vec();
                subvectors.push(SparseVector { indices, values });
            }
            vector_codes.push(self.vector_quantize(&subvectors));
        }

        vector_codes
    }

    fn vector_quantize(&self, vectors: &[SparseVector]) -> Vec<usize> {
        let mut codes: Vec<usize> = Vec::new();

        for (m, subvector) in vectors.iter().enumerate() {
            let mut min_distance = f32::MAX;
            let mut min_distance_code_index = 0;

            for (k, code) in self.codewords[m].iter().enumerate() {
                let distance = subvector.euclidean_distance(&code);
                if distance < min_distance {
                    min_distance = distance;
                    min_distance_code_index = k;
                }
            }

            codes.push(min_distance_code_index);
        }

        codes
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use super::*;

    #[test]
    fn test_pq_index_simple() {
        let num_subvectors = 2;
        let num_clusters = 3;
        let iterations = 10;
        let mut pq_index = PQIndex::new(num_subvectors, num_clusters, iterations, 0.01, 42);

        let vectors = vec![
            SparseVector {
                indices: vec![0, 2, 5],
                values: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            },
            SparseVector {
                indices: vec![1, 3, 4],
                values: vec![OrderedFloat(4.0), OrderedFloat(5.0), OrderedFloat(6.0)],
            },
            SparseVector {
                indices: vec![0, 2, 5],
                values: vec![OrderedFloat(7.0), OrderedFloat(8.0), OrderedFloat(9.0)],
            },
        ];

        for vector in vectors {
            pq_index.add_vector(vector);
        }

        pq_index.build();

        let query_vector = SparseVector {
            indices: vec![0, 2, 5],
            values: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
        };

        let result = pq_index.search(&query_vector, 10);
        println!("{:?}", result);

        assert!(true);
    }

    #[test]
    fn test_encode() {
        let num_subvectors = 2;
        let num_clusters = 2;
        let iterations = 10;
        let mut pq_index = PQIndex::new(num_subvectors, num_clusters, iterations, 0.01, 42);

        pq_index.codewords = vec![
            vec![
                SparseVector {
                    indices: vec![0, 1],
                    values: vec![OrderedFloat(1.0), OrderedFloat(1.0)],
                },
                SparseVector {
                    indices: vec![0, 1],
                    values: vec![OrderedFloat(5.0), OrderedFloat(5.0)],
                },
            ],
            vec![
                SparseVector {
                    indices: vec![2, 3],
                    values: vec![OrderedFloat(3.0), OrderedFloat(3.0)],
                },
                SparseVector {
                    indices: vec![2, 3],
                    values: vec![OrderedFloat(7.0), OrderedFloat(7.0)],
                },
            ],
        ];

        let vectors = vec![
            SparseVector {
                indices: vec![0, 1, 2, 3],
                values: vec![
                    OrderedFloat(1.0),
                    OrderedFloat(1.0),
                    OrderedFloat(3.0),
                    OrderedFloat(3.0),
                ],
            },
            SparseVector {
                indices: vec![0, 1, 2, 3],
                values: vec![
                    OrderedFloat(5.0),
                    OrderedFloat(5.0),
                    OrderedFloat(7.0),
                    OrderedFloat(7.0),
                ],
            },
        ];

        let encoded_vectors = pq_index.encode(&vectors);

        println!("{:?}", encoded_vectors);
        assert!(true)
    }

    #[test]
    fn test_pq_index_complex() {
        let num_subvectors = 2;
        let num_clusters = 10;
        let iterations = 256;
        let random_seed = 42;
        let mut index = PQIndex::new(num_subvectors, num_clusters, iterations, 0.01, random_seed);

        let mut vectors = vec![];
        for i in 0..100 {
            vectors.push(SparseVector {
                indices: vec![i % 10, (i / 10) % 10],
                values: vec![OrderedFloat((i % 10) as f32), OrderedFloat((i / 10) as f32)],
            });
        }

        for vector in &vectors {
            index.add_vector(vector.clone());
        }

        index.build();

        let query_vector = SparseVector {
            indices: vec![1, 7],
            values: vec![OrderedFloat(1.5), OrderedFloat(7.0)],
        };
        let results = index.search(&query_vector, 10);
        println!("Results for search on query vector: {:?}", results);
        println!("Top Search: {:?}", vectors[results[0].index]);
        println!("Groundtruth: {:?}", query_vector);

        assert!(true);
    }
}
