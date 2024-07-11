use std::{collections::HashSet, sync::Mutex};

use minhash::minhash;
use ordered_float::OrderedFloat;
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use simhash::simhash;

use crate::{
    data::{QueryResult, SparseVector},
    data_structures::min_heap::MinHeap,
};

use super::{DistanceMetric, SparseIndex};

mod minhash;
mod simhash;

pub enum LSHHashType {
    MinHash,
    SimHash,
}

pub struct LSHIndex {
    num_buckets: usize,
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
    fn add_vector(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
    }

    fn remove_vector(&mut self, id: usize) -> Option<SparseVector> {
        // TODO: Needs implementation
        None
    }

    fn build(&mut self) {
        for (id, vector) in self.vectors.iter().enumerate() {
            for i in 0..self.num_hash_functions {
                let hash = self.hash(vector, i);
                let bucket_index = self.hash_bucket(hash);
                self.buckets[bucket_index].push((id, vector.clone()));
            }
        }

        for bucket in &mut self.buckets {
            bucket.sort_by(|a, b| a.1.values.partial_cmp(&b.1.values).unwrap());
        }
    }

    fn build_parallel(&mut self) {
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
        let mut results: Vec<(f32, usize, SparseVector, usize)> = Vec::new();

        for i in 0..self.num_hash_functions {
            let query_hash = self.hash(query_vector, i);
            let bucket_index = self.hash_bucket(query_hash);
            let bucket = &self.buckets[bucket_index];

            for (index, vector) in bucket.iter() {
                let similarity = query_vector.distance(&vector, &self.metric);
                let mut found = false;

                for (existing_similarity, _, existing_vector, bucket_count) in &mut results {
                    if *existing_vector == *vector {
                        *existing_similarity += similarity;
                        *bucket_count += 1;
                        found = true;
                        break;
                    }
                }

                if !found {
                    results.push((similarity, *index, vector.clone(), 1));
                }
            }
        }

        for (similarity, _, _, bucket_count) in &mut results {
            *similarity /= *bucket_count as f32;
        }

        let mut heap: MinHeap<QueryResult> = MinHeap::new();
        for (score, index, _, _) in results.iter() {
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
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{get_complex_vectors, get_simple_vectors, is_in_actual_result};

    use super::*;

    #[test]
    fn test_search_parallel() {
        let (data, query_vectors) = get_simple_vectors();

        let mut index = LSHIndex::new(4, 4, LSHHashType::MinHash, DistanceMetric::Cosine);
        for vector in &data {
            index.add_vector(vector);
        }
        index.build();

        let results = index.search_parallel(&query_vectors[0], 2);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
    }

    #[test]
    fn test_build_parallel() {
        let (data, query_vectors) = get_simple_vectors();

        let mut index = LSHIndex::new(4, 4, LSHHashType::MinHash, DistanceMetric::Cosine);
        for vector in &data {
            index.add_vector(vector);
        }
        index.build_parallel();

        let results = index.search(&query_vectors[0], 2);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
    }

    #[test]
    fn test_lsh_index_min_hash_simple() {
        let (data, query_vectors) = get_simple_vectors();

        let mut index = LSHIndex::new(4, 4, LSHHashType::MinHash, DistanceMetric::Cosine);
        for vector in &data {
            index.add_vector(vector);
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
            index.add_vector(vector);
        }

        index.build();

        let results = index.search(&query_vector, 10);
        assert!(is_in_actual_result(&data, &query_vector, &results));

        assert!(true);
    }

    #[test]
    fn test_lsh_index_sim_hash_simple() {
        let (data, query_vectors) = get_simple_vectors();

        let mut index = LSHIndex::new(4, 4, LSHHashType::SimHash, DistanceMetric::Cosine);
        for vector in &data {
            index.add_vector(vector);
        }
        index.build();

        let results = index.search(&query_vectors[0], 2);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
    }

    #[test]
    fn test_lsh_index_sim_hash_complex() {
        let (data, query_vector) = get_complex_vectors();

        let mut index = LSHIndex::new(10, 4, LSHHashType::SimHash, DistanceMetric::Cosine);

        for vector in &data {
            index.add_vector(vector);
        }

        index.build();

        let results = index.search(&query_vector, 10);
        assert!(is_in_actual_result(&data, &query_vector, &results));
    }
}
