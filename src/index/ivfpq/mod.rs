use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    sync::Mutex,
};

use ordered_float::OrderedFloat;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};

use crate::{
    data::{vector::SparseVector, QueryResult},
    data_structures::min_heap::MinHeap,
    kmeans::kmeans,
};

use super::{DistanceMetric, IndexIdentifier, SparseIndex};

#[derive(Serialize, Deserialize, Clone, Debug)]
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
        vectors
            .par_iter()
            .map(|vec| {
                self.coarse_centroids
                    .iter()
                    .enumerate()
                    .map(|(i, coarse_codeword)| (i, vec.distance(coarse_codeword, &self.metric)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }

    fn encode(&self, vectors: &Vec<SparseVector>) -> Vec<Vec<usize>> {
        vectors
            .par_iter()
            .enumerate()
            .map(|(i, vec)| {
                let coarse_code = self.coarse_codes[i];
                let sub_vec_dims = vec.indices.len() / self.num_subvectors;
                let subvectors: Vec<SparseVector> = (0..self.num_subvectors)
                    .map(|m| {
                        let start_idx = m * sub_vec_dims;
                        let end_idx = ((m + 1) * sub_vec_dims).min(vec.indices.len());
                        SparseVector {
                            indices: vec.indices[start_idx..end_idx].to_vec(),
                            values: vec.values[start_idx..end_idx].to_vec(),
                        }
                    })
                    .collect();
                self.vector_quantize(&subvectors, coarse_code)
            })
            .collect()
    }

    fn vector_quantize(&self, vectors: &[SparseVector], coarse_code: usize) -> Vec<usize> {
        vectors
            .par_iter()
            .enumerate()
            .map(|(m, subvector)| {
                self.sub_quantizers[coarse_code][m]
                    .iter()
                    .enumerate()
                    .map(|(k, code)| (k, subvector.distance(code, &self.metric)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(k, _)| k)
                    .unwrap_or(0)
            })
            .collect()
    }
}

impl SparseIndex for IVFPQIndex {
    fn add_vector_before_build(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
    }

    fn add_vector(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());

        let mut min_distance = f32::MAX;
        let mut min_distance_code_index = 0;
        for (i, coarse_codeword) in self.coarse_centroids.iter().enumerate() {
            let distance = vector.distance(coarse_codeword, &self.metric);
            if distance < min_distance {
                min_distance = distance;
                min_distance_code_index = i;
            }
        }
        self.coarse_codes.push(min_distance_code_index);

        let sub_vec_dims = vector.indices.len() / self.num_subvectors;
        let remaining_dims = vector.indices.len() % self.num_subvectors;
        let mut subvectors: Vec<SparseVector> = Vec::new();

        for m in 0..self.num_subvectors {
            let start_idx = m * sub_vec_dims + m.min(remaining_dims);
            let end_idx = start_idx + sub_vec_dims + (m < remaining_dims) as usize;
            let indices = vector.indices[start_idx..end_idx].to_vec();
            let values = vector.values[start_idx..end_idx].to_vec();
            subvectors.push(SparseVector { indices, values });
        }

        let pq_code = self.vector_quantize(&subvectors, min_distance_code_index);
        self.pq_codes.push(pq_code);
    }

    fn remove_vector(&mut self, id: usize) -> Option<SparseVector> {
        if id <= self.vectors.len() {
            let removed_vector = self.vectors.remove(id);

            self.coarse_codes.remove(id);
            self.pq_codes.remove(id);

            Some(removed_vector)
        } else {
            None
        }
    }

    fn build(&mut self) {
        self.coarse_centroids = kmeans(
            &self.vectors,
            self.num_coarse_clusters,
            self.kmeans_iterations,
            self.tolerance,
            self.random_seed,
            &self.metric,
        );
        self.coarse_codes = self.encode_coarse(&self.vectors);

        let cluster_vectors: Vec<Mutex<Vec<SparseVector>>> = (0..self.num_coarse_clusters)
            .map(|_| Mutex::new(Vec::new()))
            .collect();

        self.vectors.par_iter().enumerate().for_each(|(i, vec)| {
            let code = self.coarse_codes[i];
            cluster_vectors[code].lock().unwrap().push(vec.clone());
        });

        let mut all_subvectors: Vec<Vec<Vec<SparseVector>>> =
            vec![vec![Vec::new(); self.num_subvectors]; self.num_coarse_clusters];

        // Pre-compute subvectors
        for (i, vec) in self.vectors.iter().enumerate() {
            let coarse_code = self.coarse_codes[i];
            let sub_vec_dims = vec.indices.len() / self.num_subvectors;
            let remaining_dims = vec.indices.len() % self.num_subvectors;

            for m in 0..self.num_subvectors {
                let start_idx = m * sub_vec_dims + m.min(remaining_dims);
                let end_idx = start_idx + sub_vec_dims + (m < remaining_dims) as usize;
                let subvec = SparseVector {
                    indices: vec.indices[start_idx..end_idx].to_vec(),
                    values: vec.values[start_idx..end_idx].to_vec(),
                };
                all_subvectors[coarse_code][m].push(subvec);
            }
        }

        self.sub_quantizers = (0..self.num_coarse_clusters * self.num_subvectors)
            .into_par_iter()
            .map(|index| {
                let c = index / self.num_subvectors;
                let m = index % self.num_subvectors;

                if all_subvectors[c][m].is_empty() {
                    vec![]
                } else {
                    kmeans(
                        &all_subvectors[c][m],
                        self.num_clusters,
                        self.kmeans_iterations,
                        self.tolerance,
                        self.random_seed + c as u64 + m as u64,
                        &self.metric,
                    )
                }
            })
            .collect::<Vec<_>>()
            .chunks(self.num_subvectors)
            .map(|chunk| chunk.to_vec())
            .collect();

        self.pq_codes = self.encode(&self.vectors);
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
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
        let mut writer = BufWriter::new(file);
        let index_type = IndexIdentifier::Ivfpq.to_u32();
        writer
            .write_all(&index_type.to_be_bytes())
            .expect("Failed to write metadata");
        bincode::serialize_into(&mut writer, &self).expect("Failed to serialize");
    }

    fn load_index(file: &File) -> Self {
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(0)).unwrap();

        let mut buffer = [0u8; 4];
        reader
            .read_exact(&mut buffer)
            .expect("Failed to read metadata");
        let index_type = u32::from_be_bytes(buffer);
        assert_eq!(index_type, IndexIdentifier::Ivfpq.to_u32());
        bincode::deserialize_from(&mut reader).unwrap()
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
            index.add_vector_before_build(vector);
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
    fn test_add_vector() {
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

        let (vectors, _) = get_simple_vectors();
        for vector in &vectors {
            index.add_vector_before_build(vector);
        }
        index.build();

        assert_eq!(index.vectors.len(), vectors.len());

        let new_vector = SparseVector {
            indices: vec![1, 3],
            values: vec![OrderedFloat(4.0), OrderedFloat(5.0)],
        };
        index.add_vector(&new_vector);

        assert_eq!(index.vectors.len(), vectors.len() + 1);
        assert_eq!(index.vectors[index.vectors.len() - 1], new_vector);

        let results = index.search(&new_vector, 2);

        println!("{:?}", results);
        assert_eq!(results[0].index, vectors.len());
        assert_eq!(results[1].index, 1);
    }

    #[test]
    fn test_remove_vector() {
        let mut index = IVFPQIndex::new(2, 4, 2, 10, 0.001, DistanceMetric::Cosine, 42);

        let (vectors, _) = get_simple_vectors();
        for vector in &vectors {
            index.add_vector_before_build(vector);
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
            index.add_vector_before_build(vector);
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
            index.add_vector_before_build(vector);
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
            index.add_vector_before_build(vector);
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
            index.add_vector_before_build(vector);
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
            index.add_vector_before_build(vector);
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
