use std::{
    fs::File,
    io::{BufReader, BufWriter},
    sync::Mutex,
};

use ordered_float::OrderedFloat;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};

use crate::{
    data::{QueryResult, SparseVector},
    data_structures::min_heap::MinHeap,
    kmeans::{kmeans, kmeans_parallel},
};

use super::{DistanceMetric, SparseIndex};

#[derive(Serialize, Deserialize)]
pub struct IVFPQIndex {
    /// Number of subvectors for PQ.
    /// Higher values increase granularity of vector encoding, potentially
    /// improving accuracy but increasing memory usage and computational cost.
    /// Lower values result in more compact representations but may reduce
    /// accuracy.
    num_subvectors: usize,
    /// Number of clusters for each subquantizer in PQ.
    /// Higher values increase precision of vector encoding, improving accuracy
    /// but increasing memory usage and computational cost.
    num_clusters: usize,
    /// Number of clusters for the coarse quantizer (IVF part).
    /// Higher values increase granularity of the first-level clustering,
    /// potentially improving search accuracy but increasing memory usage and
    /// search time.
    /// Lower values result in fewer, larger clusters, which can speed up
    /// searches but may reduce accuracy.
    num_coarse_clusters: usize,
    vectors: Vec<SparseVector>,
    coarse_centroids: Vec<SparseVector>,
    sub_quantizers: Vec<Vec<Vec<SparseVector>>>,
    coarse_codes: Vec<usize>,
    pq_codes: Vec<Vec<usize>>,
    /// Number of iterations for k-means clustering.
    /// Lower values speed up index construction but may result in suboptimal
    /// clustering.
    kmeans_iterations: usize,
    tolerance: f32,
    metric: DistanceMetric,
    random_seed: u64,
}

impl IVFPQIndex {
    pub fn new(
        num_subvectors: usize,
        num_clusters: usize,
        num_coarse_clusters: usize,
        kmeans_iterations: usize,
        tolerance: f32,
        metric: DistanceMetric,
        random_seed: u64,
    ) -> Self {
        IVFPQIndex {
            num_subvectors,
            num_clusters,
            num_coarse_clusters,
            random_seed,
            kmeans_iterations,
            tolerance,
            metric,
            vectors: Vec::new(),
            coarse_centroids: Vec::new(),
            sub_quantizers: Vec::new(),
            coarse_codes: Vec::new(),
            pq_codes: Vec::new(),
        }
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

impl SparseIndex for IVFPQIndex {
    fn add_vector(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
    }

    fn remove_vector(&mut self, id: usize) -> Option<SparseVector> {
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

    fn build(&mut self) {
        // Perform coarse quantization using k-means clustering while assigning
        // each vector to the nearest centroid.
        self.coarse_centroids = kmeans(
            &self.vectors,
            self.num_coarse_clusters,
            self.kmeans_iterations,
            self.tolerance,
            self.random_seed,
            &self.metric,
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
                    self.kmeans_iterations,
                    self.tolerance,
                    self.random_seed,
                    &self.metric,
                );
                cluster_codewords.push(codewords_m);
            }
            self.sub_quantizers[c] = cluster_codewords;
        }

        self.pq_codes = self.encode(&self.vectors);
    }

    fn build_parallel(&mut self) {
        self.coarse_centroids = kmeans_parallel(
            &self.vectors,
            self.num_coarse_clusters,
            self.kmeans_iterations,
            self.tolerance,
            self.random_seed,
            &self.metric,
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

                        kmeans_parallel(
                            &sub_vectors_m,
                            self.num_clusters,
                            self.kmeans_iterations,
                            self.tolerance,
                            self.random_seed + c as u64 + m as u64,
                            &self.metric,
                        )
                    })
                    .collect();
            });

        self.pq_codes = self.encode(&self.vectors);
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
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

    fn search_parallel(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let mut coarse_distances: Vec<(usize, f32)> = self
            .coarse_centroids
            .par_iter()
            .enumerate()
            .map(|(i, coarse_codeword)| (i, query_vector.distance(coarse_codeword, &self.metric)))
            .collect();
        coarse_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let sub_vec_dims = query_vector.indices.len() / self.num_subvectors;
        let scores = Mutex::new(Vec::with_capacity(self.vectors.len()));

        coarse_distances
            .par_iter()
            .take(k)
            .for_each(|&(coarse_index, _)| {
                let local_scores: Vec<(f32, usize)> = self
                    .coarse_codes
                    .par_iter()
                    .enumerate()
                    .filter(|&(_, &coarse_code)| coarse_code == coarse_index)
                    .map(|(n, _)| {
                        let distance: f32 = (0..self.num_subvectors)
                            .map(|m| {
                                let start_idx = m * sub_vec_dims;
                                let end_idx =
                                    ((m + 1) * sub_vec_dims).min(query_vector.indices.len());
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
                        (distance, n)
                    })
                    .collect();

                scores.lock().unwrap().extend(local_scores);
            });

        let scores = scores.into_inner().unwrap();

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
    use ordered_float::OrderedFloat;
    use rand::{thread_rng, Rng};

    use crate::test_utils::{get_complex_vectors, get_simple_vectors, is_in_actual_result};

    use super::*;

    #[test]
    fn test_serde() {
        let (data, _) = get_simple_vectors();
        let random_seed = 42;
        let num_subvectors = 2;
        let num_clusters = 3;
        let num_coarse_clusters = 2;
        let kmeans_iterations = 10;
        let mut index = IVFPQIndex::new(
            num_subvectors,
            num_clusters,
            num_coarse_clusters,
            kmeans_iterations,
            0.01,
            DistanceMetric::Euclidean,
            random_seed,
        );
        for vector in &data {
            index.add_vector(vector);
        }

        let bytes = bincode::serialize(&index).unwrap();
        let reconstructed: IVFPQIndex = bincode::deserialize(&bytes).unwrap();

        assert_eq!(index.vectors, reconstructed.vectors);
        assert_eq!(index.metric, reconstructed.metric);
        assert_eq!(index.random_seed, reconstructed.random_seed);
        assert_eq!(index.num_subvectors, reconstructed.num_subvectors);
        assert_eq!(index.num_clusters, reconstructed.num_clusters);
        assert_eq!(index.num_coarse_clusters, reconstructed.num_coarse_clusters);
        assert_eq!(index.coarse_centroids, reconstructed.coarse_centroids);
        assert_eq!(index.sub_quantizers, reconstructed.sub_quantizers);
        assert_eq!(index.coarse_centroids, reconstructed.coarse_centroids);
        assert_eq!(index.pq_codes, reconstructed.pq_codes);
        assert_eq!(index.kmeans_iterations, reconstructed.kmeans_iterations);
        assert_eq!(index.tolerance, reconstructed.tolerance);
    }

    #[test]
    fn test_search_parallel() {
        let random_seed = 42;
        let num_subvectors = 2;
        let num_clusters = 3;
        let num_coarse_clusters = 2;
        let kmeans_iterations = 10;
        let mut index = IVFPQIndex::new(
            num_subvectors,
            num_clusters,
            num_coarse_clusters,
            kmeans_iterations,
            0.01,
            DistanceMetric::Euclidean,
            random_seed,
        );

        let (data, query_vectors) = get_simple_vectors();

        for vector in &data {
            index.add_vector(vector);
        }
        index.build();

        let results = index.search_parallel(&query_vectors[0], 2);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
    }

    #[test]
    fn test_build_parallel() {
        let random_seed = 42;
        let num_subvectors = 2;
        let num_clusters = 3;
        let num_coarse_clusters = 2;
        let kmeans_iterations = 10;
        let mut index = IVFPQIndex::new(
            num_subvectors,
            num_clusters,
            num_coarse_clusters,
            kmeans_iterations,
            0.01,
            DistanceMetric::Euclidean,
            random_seed,
        );

        let (data, query_vectors) = get_simple_vectors();

        for vector in &data {
            index.add_vector(vector);
        }
        index.build_parallel();

        let results = index.search(&query_vectors[0], 2);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
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
        let kmeans_iterations = 10;
        let mut index = IVFPQIndex::new(
            num_subvectors,
            num_clusters,
            num_coarse_clusters,
            kmeans_iterations,
            0.01,
            DistanceMetric::Euclidean,
            random_seed,
        );

        let (data, query_vectors) = get_simple_vectors();

        for vector in &data {
            index.add_vector(vector);
        }
        index.build();

        let results = index.search(&query_vectors[0], 2);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
    }

    #[test]
    fn test_ivfpq_index_complex() {
        let random_seed = thread_rng().gen::<u64>();
        let num_subvectors = 2;
        let num_clusters = 10;
        let num_coarse_clusters = 40;
        let kmeans_iterations = 256;
        let mut index = IVFPQIndex::new(
            num_subvectors,
            num_clusters,
            num_coarse_clusters,
            kmeans_iterations,
            0.01,
            DistanceMetric::Euclidean,
            random_seed,
        );

        let (data, query_vector) = get_complex_vectors();
        for vector in &data {
            index.add_vector(vector);
        }

        index.build();

        let results = index.search(&query_vector, 2);
        assert!(is_in_actual_result(&data, &query_vector, &results));
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
