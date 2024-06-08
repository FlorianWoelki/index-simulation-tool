use crate::{
    data::{QueryResult, SparseVector},
    data_structures::min_heap::MinHeap,
};
use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::{collections::HashMap, vec};

struct PQIndex {
    num_subvectors: usize,
    num_clusters: usize,
    vectors: Vec<SparseVector>,
    codewords: Vec<Vec<SparseVector>>,
    encoded_codes: Vec<Vec<usize>>,
    iterations: usize,
    random_seed: u64,
}

impl PQIndex {
    pub fn new(
        num_subvectors: usize,
        num_clusters: usize,
        iterations: usize,
        random_seed: u64,
    ) -> Self {
        PQIndex {
            num_subvectors,
            num_clusters,
            random_seed,
            iterations,
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

            let codewords_m = self.kmeans(sub_vectors_m);
            self.codewords.push(codewords_m);
        }

        self.encoded_codes = self.encode(&self.vectors);
    }

    fn kmeans(&self, vectors: Vec<SparseVector>) -> Vec<SparseVector> {
        let mut centroids = self.init_centroids(vectors.clone());
        for _ in 0..self.iterations {
            let mut clusters = vec![Vec::new(); self.num_clusters];
            for vector in vectors.iter() {
                let mut min_distance = f32::MAX;
                let mut cluster_index = 0;
                for (i, centroid) in centroids.iter().enumerate() {
                    let distance = self.euclidean_distance(vector, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        cluster_index = i;
                    }
                }
                clusters[cluster_index].push(vector.clone());
            }
            centroids = self.update_centroids(clusters);
        }
        centroids
    }

    fn update_centroids(&self, clusters: Vec<Vec<SparseVector>>) -> Vec<SparseVector> {
        let mut centroids = Vec::new();
        for cluster in clusters {
            if cluster.is_empty() {
                centroids.push(SparseVector {
                    indices: vec![],
                    values: vec![],
                });
                continue;
            }

            let mut sum_indices = HashMap::new();
            for vector in cluster {
                for (i, &index) in vector.indices.iter().enumerate() {
                    let value = vector.values[i].into_inner();
                    sum_indices.entry(index).or_insert((0.0, 0)).0 += value;
                    sum_indices.entry(index).or_insert((0.0, 0)).1 += 1;
                }
            }

            let mut centroid_indices = Vec::new();
            let mut centroid_values = Vec::new();

            for (index, (sum, count)) in sum_indices {
                centroid_indices.push(index);
                centroid_values.push(OrderedFloat(sum / count as f32));
            }

            let centroid = SparseVector {
                indices: centroid_indices,
                values: centroid_values,
            };
            centroids.push(centroid);
        }
        centroids
    }

    fn init_centroids(&self, vectors: Vec<SparseVector>) -> Vec<SparseVector> {
        let mut rng = StdRng::seed_from_u64(self.random_seed);
        let mut centroids = Vec::new();

        let mut indices: Vec<usize> = (0..vectors.len()).collect();
        indices.shuffle(&mut rng);

        for &index in indices.iter().take(self.num_clusters) {
            centroids.push(vectors[index].clone());
        }
        centroids
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
                let sub_distance = self.euclidean_distance(&query_sub, &self.codewords[m][code[m]]);
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
                let distance = self.euclidean_distance(&subvector, &code);
                if distance < min_distance {
                    min_distance = distance;
                    min_distance_code_index = k;
                }
            }

            codes.push(min_distance_code_index);
        }

        codes
    }

    fn euclidean_distance(&self, vec1: &SparseVector, vec2: &SparseVector) -> f32 {
        let mut p = 0;
        let mut q = 0;
        let mut distance = 0.0;

        while p < vec1.indices.len() && q < vec2.indices.len() {
            if vec1.indices[p] == vec2.indices[q] {
                let diff = vec1.values[p].into_inner() - vec2.values[q].into_inner();
                distance += diff * diff;
                p += 1;
                q += 1;
            } else if vec1.indices[p] < vec2.indices[q] {
                distance += vec1.values[p].into_inner() * vec1.values[p].into_inner();
                p += 1;
            } else {
                distance += vec2.values[q].into_inner() * vec2.values[q].into_inner();
                q += 1;
            }
        }

        while p < vec1.indices.len() {
            distance += vec1.values[p].into_inner() * vec1.values[p].into_inner();
            p += 1;
        }

        while q < vec2.indices.len() {
            distance += vec2.values[q].into_inner() * vec2.values[q].into_inner();
            q += 1;
        }

        distance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use super::*;

    #[test]
    fn test_pq_index() {
        let num_subvectors = 2;
        let num_clusters = 3;
        let iterations = 10;
        let mut pq_index = PQIndex::new(num_subvectors, num_clusters, iterations, 42);

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

        assert!(false);
    }

    #[test]
    fn test_encode() {
        let num_subvectors = 2;
        let num_clusters = 2;
        let iterations = 10;
        let mut pq_index = PQIndex::new(num_subvectors, num_clusters, iterations, 42);

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
}
