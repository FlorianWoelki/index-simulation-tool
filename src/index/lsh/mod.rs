use std::{
    collections::HashSet,
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    sync::Mutex,
};

use minhash::minhash;
use ordered_float::OrderedFloat;
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use simhash::simhash;

use crate::{
    data::{vector::SparseVector, QueryResult},
    data_structures::min_heap::MinHeap,
};

use super::{DistanceMetric, IndexIdentifier, SparseIndex};

mod minhash;
mod simhash;

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub enum LSHHashType {
    MinHash,
    SimHash,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LSHIndex {
    /// Number of buckets in the hash table.
    /// Higher values can improve search speed but increase memory usage.
    num_buckets: usize,
    /// Number of hash functions used for each vector.
    /// Higher values increase accuracy but also increase computational cost
    /// and memory usage.
    num_hash_functions: usize,
    buckets: Vec<Vec<(usize, SparseVector)>>,
    vectors: Vec<SparseVector>,
    hash_type: LSHHashType,
    metric: DistanceMetric,
}

impl LSHIndex {
    pub fn new(
        num_buckets: usize,
        num_hash_functions: usize,
        hash_type: LSHHashType,
        metric: DistanceMetric,
    ) -> Self {
        assert!(
            num_hash_functions >= 2,
            "num_hash_functions must be at least 2"
        );
        LSHIndex {
            num_buckets,
            num_hash_functions,
            buckets: vec![Vec::new(); num_buckets],
            vectors: Vec::new(),
            hash_type,
            metric,
        }
    }

    /// Based on the MurmurHash3 algorithm.
    fn hash_bucket(&self, hash: u64) -> usize {
        let a: u64 = 0x9e3779b97f4a7c15;
        let b: u64 = 0xbb67ae8584caa73b;
        let c: u64 = 0x637c835768936735;
        let d: u64 = 0xf87694f3329e33d1;
        let hash = hash
            .wrapping_mul(a)
            .rotate_left(47)
            .wrapping_mul(b)
            .rotate_right(43)
            .wrapping_mul(c)
            .wrapping_add(d);
        (hash as usize) % self.num_buckets
    }

    fn hash(&self, vector: &SparseVector, i: usize) -> u64 {
        match self.hash_type {
            LSHHashType::MinHash => minhash(vector, i),
            LSHHashType::SimHash => simhash(vector, i),
        }
    }
}

impl SparseIndex for LSHIndex {
    fn add_vector_before_build(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
    }

    fn add_vector(&mut self, vector: &SparseVector) {
        let id = self.vectors.len();
        self.vectors.push(vector.clone());
        for i in 0..self.num_hash_functions {
            let hash = self.hash(&vector, i);
            let bucket_index = self.hash_bucket(hash);
            self.buckets[bucket_index].push((id, vector.clone()));
        }
    }

    fn remove_vector(&mut self, id: usize) -> Option<SparseVector> {
        if id >= self.vectors.len() {
            return None;
        }

        let vector = self.vectors.remove(id);
        for bucket in &mut self.buckets {
            bucket.retain(|&(vec_id, _)| vec_id != id);
        }

        for bucket in &mut self.buckets {
            for (vec_id, _) in bucket.iter_mut() {
                if *vec_id > id {
                    *vec_id -= 1;
                }
            }
        }

        Some(vector)
    }

    fn build(&mut self) {
        let mutex_buckets: Vec<Mutex<Vec<(usize, SparseVector)>>> = self
            .buckets
            .iter()
            .map(|bucket| Mutex::new(bucket.clone()))
            .collect();

        (0..self.vectors.len())
            .into_par_iter()
            .for_each(|vector_index| {
                let vector = &self.vectors[vector_index];

                for i in 0..self.num_hash_functions {
                    let hash = self.hash(vector, i);
                    let bucket_index = self.hash_bucket(hash);

                    let mut bucket = mutex_buckets[bucket_index].lock().unwrap();
                    bucket.push((vector_index, vector.clone()));
                }
            });

        self.buckets = mutex_buckets
            .into_iter()
            .map(|mutex| mutex.into_inner().unwrap())
            .collect();

        self.buckets
            .par_iter_mut()
            .for_each(|bucket| bucket.sort_by(|a, b| a.1.values.partial_cmp(&b.1.values).unwrap()));
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let candidate_set = Mutex::new(HashSet::new());

        (0..self.num_hash_functions).into_par_iter().for_each(|i| {
            let query_hash = self.hash(query_vector, i);
            let bucket_index = self.hash_bucket(query_hash);
            let bucket = &self.buckets[bucket_index];

            let mut local_candidates = HashSet::new();
            for (index, _) in bucket {
                local_candidates.insert(*index);
            }

            let mut global_candidates = candidate_set.lock().unwrap();
            global_candidates.extend(local_candidates);
        });

        let candidates = candidate_set.into_inner().unwrap();

        let results: Vec<_> = candidates
            .into_par_iter()
            .map(|index| {
                let vector = &self.vectors[index];
                let similarity = query_vector.distance(vector, &self.metric);
                (similarity, index)
            })
            .collect();

        let mut heap: MinHeap<QueryResult> = MinHeap::new();
        for (similarity, index) in results {
            if heap.len() < k || similarity > heap.peek().unwrap().score.into_inner() {
                heap.push(
                    QueryResult {
                        score: OrderedFloat(-similarity),
                        index,
                    },
                    OrderedFloat(-similarity),
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
        let index_type = IndexIdentifier::LSH.to_u32();
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
        assert_eq!(index_type, IndexIdentifier::LSH.to_u32());
        bincode::deserialize_from(&mut reader).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        data::vector::SparseVector,
        test_utils::{get_complex_vectors, get_simple_vectors, is_in_actual_result},
    };

    use super::*;

    #[test]
    fn test_add_vector() {
        let mut index = LSHIndex::new(4, 4, LSHHashType::MinHash, DistanceMetric::Cosine);

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
        let mut index = LSHIndex::new(4, 4, LSHHashType::MinHash, DistanceMetric::Cosine);

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

        index.build();

        let results = index.search(&query_vectors[0], 2);
        assert_eq!(results[0].index, 3);
        assert_eq!(results[1].index, 0);
    }

    #[test]
    fn test_serde() {
        let (data, _) = get_simple_vectors();
        let mut index = LSHIndex::new(4, 4, LSHHashType::MinHash, DistanceMetric::Cosine);
        for vector in &data {
            index.add_vector_before_build(vector);
        }

        let bytes = bincode::serialize(&index).unwrap();
        let reconstructed: LSHIndex = bincode::deserialize(&bytes).unwrap();

        assert_eq!(index.vectors, reconstructed.vectors);
        assert_eq!(index.metric, reconstructed.metric);
        assert_eq!(index.num_buckets, reconstructed.num_buckets);
        assert_eq!(index.num_hash_functions, reconstructed.num_hash_functions);
        assert_eq!(index.buckets, reconstructed.buckets);
        assert_eq!(index.hash_type, reconstructed.hash_type);
    }

    #[test]
    fn test_lsh_index_min_hash_simple() {
        let (data, query_vectors) = get_simple_vectors();

        let mut index = LSHIndex::new(4, 4, LSHHashType::MinHash, DistanceMetric::Cosine);
        for vector in &data {
            index.add_vector_before_build(vector);
        }
        index.build();

        let results = index.search(&query_vectors[0], 2);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
    }

    #[test]
    fn test_lsh_index_min_hash_complex() {
        let mut index = LSHIndex::new(10, 4, LSHHashType::MinHash, DistanceMetric::Cosine);

        let (data, query_vector) = get_complex_vectors();

        for vector in &data {
            index.add_vector_before_build(vector);
        }

        index.build();

        let results = index.search(&query_vector, 10);
        assert!(is_in_actual_result(&data, &query_vector, &results));
    }

    #[test]
    fn test_lsh_index_sim_hash_simple() {
        let (data, query_vectors) = get_simple_vectors();

        let mut index = LSHIndex::new(4, 4, LSHHashType::SimHash, DistanceMetric::Cosine);
        for vector in &data {
            index.add_vector_before_build(vector);
        }
        index.build();

        let results = index.search(&query_vectors[0], 2);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
    }

    #[test]
    fn test_lsh_index_sim_hash_complex() {
        let (data, query_vector) = get_complex_vectors();

        let mut index = LSHIndex::new(16, 8, LSHHashType::SimHash, DistanceMetric::Cosine);

        for vector in &data {
            index.add_vector_before_build(vector);
        }

        index.build();

        let results = index.search(&query_vector, 10);
        assert!(is_in_actual_result(&data, &query_vector, &results));
    }
}
