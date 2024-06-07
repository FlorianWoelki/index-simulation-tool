use crate::{
    data::{QueryResult, SparseVector},
    data_structures::min_heap::MinHeap,
};
use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::{collections::HashMap, vec};

struct PQIndex {
    n_subvectors: usize,
    n_codes: usize,
    src_vec_dims: usize,
    codewords: Vec<Vec<SparseVector>>,
    encoded_codes: Vec<Vec<usize>>,
    seed: u64,
}

impl PQIndex {
    pub fn new(n_subvectors: usize, n_codes: usize, src_vec_dims: usize, seed: u64) -> Self {
        assert!(
            n_subvectors <= src_vec_dims,
            "`n_subvectors` has to be smaller than or equal to `src_vec_dims`"
        );
        PQIndex {
            n_subvectors,
            n_codes,
            src_vec_dims,
            seed,
            codewords: Vec::new(),
            encoded_codes: Vec::new(),
        }
    }

    pub fn add_vectors(&mut self, vectors: &Vec<SparseVector>, iterations: usize) {
        let sub_vec_dims = self.src_vec_dims / self.n_subvectors;
        self.codewords = Vec::new();
        for m in 0..self.n_subvectors {
            let mut sub_vectors_m: Vec<SparseVector> = Vec::new();
            for vec in vectors {
                let indices = vec.indices[(m * sub_vec_dims)..((m + 1) * sub_vec_dims)].to_vec();
                let values = vec.values[(m * sub_vec_dims)..((m + 1) * sub_vec_dims)].to_vec();
                sub_vectors_m.push(SparseVector { indices, values });
            }

            let codewords_m = self.kmeans(sub_vectors_m, self.n_codes, iterations, self.seed);
            self.codewords.push(codewords_m);
        }

        self.encoded_codes = self.encode(&vectors);
    }

    fn kmeans(
        &self,
        vectors: Vec<SparseVector>,
        k: usize,
        iterations: usize,
        seed: u64,
    ) -> Vec<SparseVector> {
        let mut centroids = self.init_centroids(vectors.clone(), k, seed);
        for iter in 0..iterations {
            let mut clusters = vec![Vec::new(); k];
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
            println!("Iteration {}: Centroids {:?}", iter, centroids);
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

    fn init_centroids(&self, vectors: Vec<SparseVector>, k: usize, seed: u64) -> Vec<SparseVector> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut centroids = Vec::new();

        let mut indices: Vec<usize> = (0..vectors.len()).collect();
        indices.shuffle(&mut rng);

        for &index in indices.iter().take(k) {
            centroids.push(vectors[index].clone());
        }
        centroids
    }

    pub fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let mut heap: MinHeap<QueryResult> = MinHeap::new();
        let sub_vec_dims = self.src_vec_dims / self.n_subvectors;

        let mut scores = vec![0.0; self.encoded_codes.len()];
        for (n, code) in self.encoded_codes.iter().enumerate() {
            let mut distance = 0.0;
            for m in 0..self.n_subvectors {
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
                println!(
                    "Query Subvector {}: Distance to codeword {} = {}",
                    m, code[m], sub_distance
                );
            }

            println!("Vector {}: Total distance: {}", n, distance);
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

    pub fn encode(&self, vectors: &Vec<SparseVector>) -> Vec<Vec<usize>> {
        let sub_vec_dims: usize = self.src_vec_dims / self.n_subvectors;

        let mut vector_codes: Vec<Vec<usize>> = Vec::new();
        for vec in vectors {
            let mut subvectors: Vec<SparseVector> = Vec::new();
            for m in 0..self.n_subvectors {
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
                println!("Subvector {} - Code {}: Distance = {}", m, k, distance);
                if distance < min_distance {
                    min_distance = distance;
                    min_distance_code_index = k;
                }
            }

            codes.push(min_distance_code_index);
            println!(
                "Chosen code for subvector {}: {} with distance {}",
                m, min_distance_code_index, min_distance
            );
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
        let n_subvectors = 2;
        let n_codes = 2;
        let src_vec_dims = 3;
        let mut pq_index = PQIndex::new(n_subvectors, n_codes, src_vec_dims, 42);

        let vec1 = SparseVector {
            indices: vec![0, 2, 5],
            values: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
        };
        let vec2 = SparseVector {
            indices: vec![1, 3, 4],
            values: vec![OrderedFloat(4.0), OrderedFloat(5.0), OrderedFloat(6.0)],
        };
        let vec3 = SparseVector {
            indices: vec![0, 2, 5],
            values: vec![OrderedFloat(7.0), OrderedFloat(8.0), OrderedFloat(9.0)],
        };
        let vectors = vec![vec1, vec2, vec3];

        pq_index.add_vectors(&vectors, 10);

        let query_vector = SparseVector {
            indices: vec![0, 2, 5],
            values: vec![OrderedFloat(7.0), OrderedFloat(8.0), OrderedFloat(9.0)],
        };

        let result = pq_index.search(&query_vector, 10);
        println!("{:?}", result);

        assert!(true);
    }

    #[test]
    fn test_encode() {
        let n_subvectors = 2;
        let n_codes = 2;
        let src_vec_dims = 4;
        let mut pq_index = PQIndex::new(n_subvectors, n_codes, src_vec_dims, 42);

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
