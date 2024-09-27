use crate::{
    data::{vector::SparseVector, QueryResult},
    data_structures::min_heap::MinHeap,
    kmeans::kmeans,
};
use ordered_float::OrderedFloat;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    sync::Arc,
};

use super::{DistanceMetric, IndexIdentifier, SparseIndex};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PQIndex {
    /// Number of subvectors to divide the original vector into.
    /// Higher values increase compression but may reduce accuracy.
    num_subvectors: usize,
    /// Number of centroids per subvector.
    /// Higher values improve accuracy but increase memory usage and search
    /// time.
    num_clusters: usize,
    vectors: Vec<SparseVector>,
    codebooks: Vec<Vec<SparseVector>>,
    encoded_codes: Vec<Vec<usize>>,
    /// Number of iterations for k-means clustering when building codebooks.
    /// Higher values may improve codebook quality but increase build time.
    kmeans_iterations: usize,
    tolerance: f32,
    metric: DistanceMetric,
    random_seed: u64,
}

impl PQIndex {
    pub fn new(
        num_subvectors: usize,
        num_clusters: usize,
        kmeans_iterations: usize,
        tolerance: f32,
        metric: DistanceMetric,
        random_seed: u64,
    ) -> Self {
        PQIndex {
            num_subvectors,
            num_clusters,
            random_seed,
            kmeans_iterations,
            tolerance,
            metric,
            vectors: Vec::new(),
            codebooks: Vec::new(),
            encoded_codes: Vec::new(),
        }
    }

    fn vector_quantize(&self, vectors: &[SparseVector]) -> Vec<usize> {
        let mut codes: Vec<usize> = Vec::new();

        for (m, subvector) in vectors.iter().enumerate() {
            let mut min_distance = f32::MAX;
            let mut min_distance_code_index = 0;

            for (k, code) in self.codebooks[m].iter().enumerate() {
                // Finds the closest codeword to the subvector based on the distance metric.
                let distance = subvector.distance(code, &self.metric);
                if distance < min_distance {
                    min_distance = distance;
                    min_distance_code_index = k;
                }
            }

            codes.push(min_distance_code_index);
        }

        codes
    }

    fn vector_quantize_parallel(&self, vectors: &[SparseVector]) -> Vec<usize> {
        vectors
            .par_iter()
            .enumerate()
            .map(|(m, subvector)| {
                self.codebooks[m]
                    .par_iter()
                    .enumerate()
                    .map(|(k, code)| (k, subvector.distance(code, &self.metric)))
                    .min_by(|&(_, a), &(_, b)| {
                        a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(k, _)| k)
                    .unwrap()
            })
            .collect()
    }
}

impl SparseIndex for PQIndex {
    fn add_vector_before_build(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
    }

    fn add_vector(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());

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

        let encoded_code = self.vector_quantize(&subvectors);
        self.encoded_codes.push(encoded_code);
    }

    fn remove_vector(&mut self, id: usize) -> Option<SparseVector> {
        if id < self.vectors.len() {
            let removed_vector = self.vectors.remove(id);
            self.encoded_codes.remove(id);
            Some(removed_vector)
        } else {
            None
        }
    }

    fn build(&mut self) {
        let vectors = Arc::new(self.vectors.clone());
        self.codebooks = (0..self.num_subvectors)
            .into_par_iter()
            .map(|m| {
                let sub_vectors_m: Vec<SparseVector> = vectors
                    .par_iter()
                    .map(|vec| {
                        let sub_vec_dims = vec.indices.len() / self.num_subvectors;
                        let remaining_dims = vec.indices.len() % self.num_subvectors;
                        let start_idx = m * sub_vec_dims + m.min(remaining_dims);
                        let end_idx = start_idx + sub_vec_dims + (m < remaining_dims) as usize;
                        let indices = vec.indices[start_idx..end_idx].to_vec();
                        let values = vec.values[start_idx..end_idx].to_vec();
                        SparseVector { indices, values }
                    })
                    .collect();

                kmeans(
                    &sub_vectors_m,
                    self.num_clusters,
                    self.kmeans_iterations,
                    self.tolerance,
                    self.random_seed + m as u64, // Use different seeds for each subvector
                    &self.metric,
                )
            })
            .collect();

        self.encoded_codes = vectors
            .par_iter()
            .map(|vec| {
                let sub_vec_dims = vec.indices.len() / self.num_subvectors;
                let remaining_dims = vec.indices.len() % self.num_subvectors;
                let subvectors: Vec<SparseVector> = (0..self.num_subvectors)
                    .map(|m| {
                        let start_idx = m * sub_vec_dims + m.min(remaining_dims);
                        let end_idx = start_idx + sub_vec_dims + (m < remaining_dims) as usize;
                        let indices = vec.indices[start_idx..end_idx].to_vec();
                        let values = vec.values[start_idx..end_idx].to_vec();
                        SparseVector { indices, values }
                    })
                    .collect();

                self.vector_quantize_parallel(&subvectors)
            })
            .collect();
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let sub_vec_dims = query_vector.indices.len() / self.num_subvectors;
        let remaining_dims = query_vector.indices.len() % self.num_subvectors;

        let distance_tables: Vec<Vec<f32>> = (0..self.num_subvectors)
            .into_par_iter()
            .map(|m| {
                let start_idx = m * sub_vec_dims + m.min(remaining_dims);
                let end_idx = start_idx + sub_vec_dims + (m < remaining_dims) as usize;
                let query_sub_indices = query_vector.indices[start_idx..end_idx].to_vec();
                let query_sub_values = query_vector.values[start_idx..end_idx].to_vec();

                let query_sub = SparseVector {
                    indices: query_sub_indices,
                    values: query_sub_values,
                };

                self.codebooks[m]
                    .iter()
                    .map(|codeword| query_sub.distance(codeword, &self.metric))
                    .collect()
            })
            .collect();

        let distance_tables = Arc::new(distance_tables);
        let encoded_codes = Arc::new(self.encoded_codes.clone());

        let scores: Vec<(f32, usize)> = (0..self.encoded_codes.len())
            .into_par_iter()
            .map(|n| {
                let distance: f32 = encoded_codes[n]
                    .iter()
                    .enumerate()
                    .map(|(m, &code)| distance_tables[m][code])
                    .sum();
                (distance, n)
            })
            .collect();

        let mut heap: MinHeap<QueryResult> = MinHeap::new();
        for (score, index) in scores {
            if heap.len() < k || score > heap.peek().unwrap().score.into_inner() {
                heap.push(
                    QueryResult {
                        score: OrderedFloat(-score),
                        index,
                    },
                    OrderedFloat(-score),
                );
                if heap.len() > k {
                    heap.pop();
                }
            }
        }

        heap.into_sorted_vec()
            .into_iter()
            .map(|query_result| QueryResult {
                index: query_result.index,
                score: -query_result.score,
            })
            .collect()
    }

    fn save(&self, file: &mut File) {
        let mut writer = BufWriter::new(file);
        let index_type = IndexIdentifier::PQ.to_u32();
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
        assert_eq!(index_type, IndexIdentifier::PQ.to_u32());
        bincode::deserialize_from(&mut reader).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use crate::test_utils::{get_complex_vectors, get_simple_vectors, is_in_actual_result};

    use super::*;

    #[test]
    fn test_serde() {
        let (data, _) = get_simple_vectors();
        let num_subvectors = 4;
        let num_clusters = 4;
        let kmeans_iterations = 10;
        let mut index = PQIndex::new(
            num_subvectors,
            num_clusters,
            kmeans_iterations,
            0.01,
            DistanceMetric::Euclidean,
            42,
        );
        for vector in &data {
            index.add_vector_before_build(vector);
        }

        let bytes = bincode::serialize(&index).unwrap();
        let reconstructed: PQIndex = bincode::deserialize(&bytes).unwrap();

        assert_eq!(index.vectors, reconstructed.vectors);
        assert_eq!(index.metric, reconstructed.metric);
        assert_eq!(index.random_seed, reconstructed.random_seed);
        assert_eq!(index.num_subvectors, reconstructed.num_subvectors);
        assert_eq!(index.num_clusters, reconstructed.num_clusters);
        assert_eq!(index.codebooks, reconstructed.codebooks);
        assert_eq!(index.encoded_codes, reconstructed.encoded_codes);
        assert_eq!(index.kmeans_iterations, reconstructed.kmeans_iterations);
        assert_eq!(index.tolerance, reconstructed.tolerance);
    }

    #[test]
    fn test_add_vector() {
        let num_subvectors = 4;
        let num_clusters = 4;
        let kmeans_iterations = 10;
        let mut index = PQIndex::new(
            num_subvectors,
            num_clusters,
            kmeans_iterations,
            0.01,
            DistanceMetric::Euclidean,
            42,
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

        assert_eq!(results[0].index, vectors.len());
        assert_eq!(results[1].index, 1);
    }

    #[test]
    fn test_remove_vector() {
        let mut index = PQIndex::new(2, 4, 10, 0.001, DistanceMetric::Cosine, 42);

        let (vectors, _) = get_simple_vectors();

        for vector in &vectors {
            index.add_vector_before_build(vector);
        }

        index.build();

        assert_eq!(index.vectors.len(), 5);
        assert_eq!(index.encoded_codes.len(), 5);

        let result = index.remove_vector(1);
        assert_eq!(result, Some(vectors[1].clone()));
        assert_eq!(index.vectors.len(), 4);
        assert_eq!(index.encoded_codes.len(), 4);

        // Verify the remaining vectors.
        assert_eq!(index.vectors[0], vectors[0]);
        assert_eq!(index.vectors[1], vectors[2]); // Because vector with id 1 was removed.
    }

    #[test]
    fn test_remove_vector_out_of_bounds() {
        let mut index = PQIndex::new(2, 4, 10, 0.001, DistanceMetric::Cosine, 42);

        let (vectors, _) = get_simple_vectors();

        for vector in &vectors {
            index.add_vector_before_build(vector);
        }

        index.build();
        let result = index.remove_vector(10);
        assert!(result.is_none());
    }

    #[test]
    fn test_pq_index_simple() {
        let num_subvectors = 2;
        let num_clusters = 3;
        let kmeans_iterations = 10;
        let mut pq_index = PQIndex::new(
            num_subvectors,
            num_clusters,
            kmeans_iterations,
            0.01,
            DistanceMetric::Euclidean,
            42,
        );

        let (data, query_vectors) = get_simple_vectors();
        for vector in &data {
            pq_index.add_vector_before_build(vector);
        }

        pq_index.build();

        let results = pq_index.search(&query_vectors[0], 10);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
    }

    #[test]
    fn test_pq_index_complex() {
        let num_subvectors = 2;
        let num_clusters = 20;
        let kmeans_iterations = 256;
        let random_seed = 42;
        let mut index = PQIndex::new(
            num_subvectors,
            num_clusters,
            kmeans_iterations,
            0.01,
            DistanceMetric::Cosine,
            random_seed,
        );

        let (data, query_vector) = get_complex_vectors();
        for vector in &data {
            index.add_vector_before_build(vector);
        }

        index.build();

        let results = index.search(&query_vector, 10);
        assert!(is_in_actual_result(&data, &query_vector, &results));
    }
}
