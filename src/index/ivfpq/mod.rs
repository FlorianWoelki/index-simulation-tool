use std::sync::Mutex;

use ordered_float::OrderedFloat;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    data::{QueryResult, SparseVector},
    data_structures::min_heap::MinHeap,
    kmeans::kmeans,
};

use super::DistanceMetric;

pub struct IVFPQIndex {
    num_subvectors: usize,
    num_clusters: usize,
    num_coarse_clusters: usize,
    vectors: Vec<SparseVector>,
    coarse_centroids: Vec<SparseVector>,
    sub_quantizers: Vec<Vec<Vec<SparseVector>>>,
    coarse_codes: Vec<usize>,
    pq_codes: Vec<Vec<usize>>,
    iterations: usize,
    tolerance: f32,
    metric: DistanceMetric,
    random_seed: u64,
}

impl IVFPQIndex {
    pub fn new(
        num_subvectors: usize,
        num_clusters: usize,
        num_coarse_clusters: usize,
        iterations: usize,
        tolerance: f32,
        metric: DistanceMetric,
        random_seed: u64,
    ) -> Self {
        IVFPQIndex {
            num_subvectors,
            num_clusters,
            num_coarse_clusters,
            random_seed,
            iterations,
            tolerance,
            metric,
            vectors: Vec::new(),
            coarse_centroids: Vec::new(),
            sub_quantizers: Vec::new(),
            coarse_codes: Vec::new(),
            pq_codes: Vec::new(),
        }
    }

    pub fn add_vector(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
    }

    pub fn remove_vector(&mut self, id: usize) -> Option<SparseVector> {
        if id >= self.vectors.len() {
            return None;
        }

        let removed_vector = self.vectors.remove(id);
        self.coarse_codes.remove(id);
        self.pq_codes.remove(id);

        if self.vectors.is_empty() {
            self.coarse_centroids.clear();
            self.sub_quantizers.clear();
        } else {
            // Optionally: rebuild the index here (`self.build()`).
            // Can be computationally expensive, depending on the dataset.
        }

        Some(removed_vector)
    }

    pub fn build(&mut self) {
        // Perform coarse quantization using k-means clustering while assigning
        // each vector to the nearest centroid.
        self.coarse_centroids = kmeans(
            &self.vectors,
            self.num_coarse_clusters,
            self.iterations,
            self.tolerance,
            self.random_seed,
        );
        self.coarse_codes = self.encode_coarse(&self.vectors);

        // Perform product quantization within each Voronoi cell where each sub-vector
        // is quantized separately.
        self.sub_quantizers = vec![Vec::new(); self.num_coarse_clusters];
        for c in 0..self.num_coarse_clusters {
            let mut cluster_vectors: Vec<SparseVector> = Vec::new();
            for (i, &code) in self.coarse_codes.iter().enumerate() {
                if code == c {
                    cluster_vectors.push(self.vectors[i].clone());
                }
            }

            if cluster_vectors.is_empty() {
                continue;
            }

            let mut cluster_codewords = Vec::new();
            for m in 0..self.num_subvectors {
                let mut sub_vectors_m: Vec<SparseVector> = Vec::new();
                for vec in &cluster_vectors {
                    let sub_vec_dims = vec.indices.len() / self.num_subvectors;
                    let start_idx = m * sub_vec_dims;
                    let end_idx = ((m + 1) * sub_vec_dims).min(vec.indices.len());
                    let indices = vec.indices[start_idx..end_idx].to_vec();
                    let values = vec.values[start_idx..end_idx].to_vec();
                    sub_vectors_m.push(SparseVector { indices, values });
                }

                let codewords_m = kmeans(
                    &sub_vectors_m,
                    self.num_clusters,
                    self.iterations,
                    self.tolerance,
                    self.random_seed,
                );
                cluster_codewords.push(codewords_m);
            }
            self.sub_quantizers[c] = cluster_codewords;
        }

        self.pq_codes = self.encode(&self.vectors);
    }

    pub fn build_parallel(&mut self) {
        self.coarse_centroids = kmeans(
            &self.vectors,
            self.num_coarse_clusters,
            self.iterations,
            self.tolerance,
            self.random_seed,
        );
        self.coarse_codes = self.encode_coarse(&self.vectors);

        self.sub_quantizers = vec![Vec::new(); self.num_coarse_clusters];

        let cluster_vectors: Vec<Mutex<Vec<SparseVector>>> = (0..self.num_coarse_clusters)
            .map(|_| Mutex::new(Vec::new()))
            .collect();

        self.vectors.par_iter().enumerate().for_each(|(i, vec)| {
            let code = self.coarse_codes[i];
            cluster_vectors[code].lock().unwrap().push(vec.clone());
        });

        self.sub_quantizers
            .par_iter_mut()
            .enumerate()
            .for_each(|(c, cluster_codewords)| {
                let cluster_vecs = cluster_vectors[c].lock().unwrap();
                if cluster_vecs.is_empty() {
                    return;
                }

                *cluster_codewords = (0..self.num_subvectors)
                    .into_par_iter()
                    .map(|m| {
                        let sub_vectors_m: Vec<SparseVector> = cluster_vecs
                            .par_iter()
                            .map(|vec| {
                                let sub_vec_dims = vec.indices.len() / self.num_subvectors;
                                let start_idx = m * sub_vec_dims;
                                let end_idx = ((m + 1) * sub_vec_dims).min(vec.indices.len());
                                let indices = vec.indices[start_idx..end_idx].to_vec();
                                let values = vec.values[start_idx..end_idx].to_vec();
                                SparseVector { indices, values }
                            })
                            .collect();

                        kmeans(
                            &sub_vectors_m,
                            self.num_clusters,
                            self.iterations,
                            self.tolerance,
                            self.random_seed + c as u64 + m as u64,
                        )
                    })
                    .collect();
            });

        self.pq_codes = self.encode(&self.vectors);
    }

    pub fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let mut coarse_distances: Vec<(usize, f32)> = self
            .coarse_centroids
            .iter()
            .enumerate()
            .map(|(i, coarse_codeword)| (i, query_vector.distance(coarse_codeword, &self.metric)))
            .collect();
        coarse_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let sub_vec_dims = query_vector.indices.len() / self.num_subvectors;
        let mut scores: Vec<(f32, usize)> = Vec::with_capacity(self.vectors.len());

        for &(coarse_index, _) in coarse_distances.iter().take(k) {
            for (n, &coarse_code) in self.coarse_codes.iter().enumerate() {
                if coarse_code != coarse_index {
                    continue;
                }

                let distance: f32 = (0..self.num_subvectors)
                    .map(|m| {
                        let start_idx = m * sub_vec_dims;
                        let end_idx = ((m + 1) * sub_vec_dims).min(query_vector.indices.len());
                        let query_sub = SparseVector {
                            indices: query_vector.indices[start_idx..end_idx].to_vec(),
                            values: query_vector.values[start_idx..end_idx].to_vec(),
                        };
                        query_sub.distance(
                            &self.sub_quantizers[coarse_index][m][self.pq_codes[n][m]],
                            &self.metric,
                        )
                    })
                    .sum();

                scores.push((distance, n));
            }
        }

        let mut heap: MinHeap<QueryResult> = MinHeap::new();
        for (score, index) in scores.iter() {
            if heap.len() < k || *score > heap.peek().unwrap().score.into_inner() {
                heap.push(
                    QueryResult {
                        index: *index,
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

    fn encode_coarse(&self, vectors: &Vec<SparseVector>) -> Vec<usize> {
        let mut coarse_codes: Vec<usize> = Vec::new();
        for vec in vectors {
            let mut min_distance = f32::MAX;
            let mut min_distance_code_index = 0;

            for (i, coarse_codeword) in self.coarse_centroids.iter().enumerate() {
                let distance = vec.distance(&coarse_codeword, &self.metric);
                if distance < min_distance {
                    min_distance = distance;
                    min_distance_code_index = i;
                }
            }

            coarse_codes.push(min_distance_code_index);
        }

        coarse_codes
    }

    fn encode(&self, vectors: &Vec<SparseVector>) -> Vec<Vec<usize>> {
        let mut vector_codes: Vec<Vec<usize>> = Vec::new();
        for (i, vec) in vectors.iter().enumerate() {
            let coarse_code = self.coarse_codes[i];
            let sub_vec_dims: usize = vec.indices.len() / self.num_subvectors;
            let mut subvectors: Vec<SparseVector> = Vec::new();
            for m in 0..self.num_subvectors {
                let start_idx = m * sub_vec_dims;
                let end_idx = ((m + 1) * sub_vec_dims).min(vec.indices.len());
                let indices = vec.indices[start_idx..end_idx].to_vec();
                let values = vec.values[start_idx..end_idx].to_vec();
                subvectors.push(SparseVector { indices, values });
            }
            vector_codes.push(self.vector_quantize(&subvectors, coarse_code));
        }

        vector_codes
    }

    fn vector_quantize(&self, vectors: &[SparseVector], coarse_code: usize) -> Vec<usize> {
        let mut codes: Vec<usize> = Vec::new();

        for (m, subvector) in vectors.iter().enumerate() {
            let mut min_distance = f32::MAX;
            let mut min_distance_code_index = 0;

            for (k, code) in self.sub_quantizers[coarse_code][m].iter().enumerate() {
                let distance = subvector.distance(&code, &self.metric);
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
    use rand::{thread_rng, Rng};

    use crate::test_utils::get_simple_vectors;

    use super::*;

    #[test]
    fn test_build_parallel() {
        let random_seed = 42;
        let num_subvectors = 2;
        let num_clusters = 3;
        let num_coarse_clusters = 2;
        let iterations = 10;
        let mut index = IVFPQIndex::new(
            num_subvectors,
            num_clusters,
            num_coarse_clusters,
            iterations,
            0.01,
            DistanceMetric::Euclidean,
            random_seed,
        );

        let (data, query_vectors) = get_simple_vectors();

        for vector in &data {
            index.add_vector(vector);
        }
        index.build_parallel();

        let neighbors = index.search(&query_vectors[0], 2);
        println!("Nearest neighbors: {:?}", neighbors);
        assert!(true);
    }

    #[test]
    fn test_remove_vector() {
        let mut index = IVFPQIndex::new(2, 4, 2, 10, 0.001, DistanceMetric::Cosine, 42);

        let (vectors, _) = get_simple_vectors();
        for vector in &vectors {
            index.add_vector(vector);
        }

        index.build();

        assert_eq!(index.vectors.len(), 5);
        assert_eq!(index.coarse_codes.len(), 5);
        assert_eq!(index.pq_codes.len(), 5);

        let result = index.remove_vector(1);
        assert_eq!(result, Some(vectors[1].clone()));
        assert_eq!(index.vectors.len(), 4);
        assert_eq!(index.coarse_codes.len(), 4);
        assert_eq!(index.pq_codes.len(), 4);

        assert_eq!(index.vectors[0], vectors[0]);
        assert_eq!(index.vectors[1], vectors[2]);
    }

    #[test]
    fn test_remove_vector_out_of_bounds() {
        let mut index = IVFPQIndex::new(2, 4, 2, 10, 0.001, DistanceMetric::Cosine, 42);

        let (vectors, _) = get_simple_vectors();
        for vector in &vectors {
            index.add_vector(vector);
        }

        index.build();

        let result = index.remove_vector(10);
        assert!(result.is_none());
    }

    #[test]
    fn test_ivfpq_index_simple() {
        let random_seed = 42;
        let num_subvectors = 2;
        let num_clusters = 3;
        let num_coarse_clusters = 2;
        let iterations = 10;
        let mut index = IVFPQIndex::new(
            num_subvectors,
            num_clusters,
            num_coarse_clusters,
            iterations,
            0.01,
            DistanceMetric::Euclidean,
            random_seed,
        );

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
    fn test_ivfpq_index_complex() {
        let random_seed = thread_rng().gen::<u64>();
        let num_subvectors = 2;
        let num_clusters = 10;
        let num_coarse_clusters = 40;
        let iterations = 256;
        let mut index = IVFPQIndex::new(
            num_subvectors,
            num_clusters,
            num_coarse_clusters,
            iterations,
            0.01,
            DistanceMetric::Euclidean,
            random_seed,
        );

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

    #[test]
    fn test_encode_coarse() {
        let num_coarse_clusters = 4;
        let num_vectors = 10;
        let dim = 128;
        let mut vectors: Vec<SparseVector> = Vec::new();
        for _ in 0..num_vectors {
            let mut indices: Vec<usize> = Vec::new();
            let mut values: Vec<OrderedFloat<f32>> = Vec::new();
            for _ in 0..dim {
                let idx = rand::thread_rng().gen_range(0..dim);
                let val = rand::thread_rng().gen_range(0.0..1.0);
                indices.push(idx);
                values.push(OrderedFloat(val));
            }
            vectors.push(SparseVector { indices, values });
        }

        let mut index = IVFPQIndex::new(
            4,
            256,
            num_coarse_clusters,
            10,
            1e-6,
            DistanceMetric::Euclidean,
            42,
        );

        for vector in &vectors {
            index.add_vector(vector);
        }

        index.build();

        let coarse_codes = index.encode_coarse(&vectors);
        assert_eq!(coarse_codes.len(), num_vectors);
        for code in &coarse_codes {
            assert!(*code < num_coarse_clusters);
        }
    }

    #[test]
    fn test_encode() {
        let num_coarse_clusters = 4;
        let num_vectors = 10;
        let dim = 128;
        let mut vectors: Vec<SparseVector> = Vec::new();
        for _ in 0..num_vectors {
            let mut indices: Vec<usize> = Vec::new();
            let mut values: Vec<OrderedFloat<f32>> = Vec::new();
            for _ in 0..dim {
                let idx = rand::thread_rng().gen_range(0..dim);
                let val = rand::thread_rng().gen_range(0.0..1.0);
                indices.push(idx);
                values.push(OrderedFloat(val));
            }
            vectors.push(SparseVector { indices, values });
        }

        let mut index = IVFPQIndex::new(
            4,
            256,
            num_coarse_clusters,
            10,
            1e-6,
            DistanceMetric::Euclidean,
            42,
        );

        for vector in &vectors {
            index.add_vector(vector);
        }

        index.build();

        let pq_codes = index.encode(&vectors);
        assert_eq!(pq_codes.len(), num_vectors);
        for codes in &pq_codes {
            assert_eq!(codes.len(), 4);
            for code in codes {
                assert!(*code < 256);
            }
        }
    }
}
