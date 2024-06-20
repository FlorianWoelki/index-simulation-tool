use std::hash::{DefaultHasher, Hash, Hasher};

use ordered_float::OrderedFloat;

use crate::{
    data::{QueryResult, SparseVector},
    data_structures::min_heap::MinHeap,
};

pub struct LSHIndex {
    num_buckets: usize,
    num_hash_functions: usize,
    buckets: Vec<Vec<(usize, SparseVector)>>,
    vectors: Vec<SparseVector>,
}

impl LSHIndex {
    pub fn new(num_buckets: usize, num_hash_functions: usize) -> Self {
        assert!(
            num_hash_functions >= 2,
            "num_hash_functions must be at least 2"
        );
        LSHIndex {
            num_buckets,
            num_hash_functions,
            buckets: vec![Vec::new(); num_buckets],
            vectors: Vec::new(),
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

    pub fn add_vector(&mut self, vector: &SparseVector) {
        for i in 0..self.num_hash_functions {
            let hash = self.minhash(vector, i);
            let bucket_index = self.hash_bucket(hash);
            self.buckets[bucket_index].push((self.vectors.len(), vector.clone()));
        }
        self.vectors.push(vector.clone());
    }

    pub fn build(&mut self) {
        for bucket in &mut self.buckets {
            bucket.sort_by(|a, b| a.1.values.partial_cmp(&b.1.values).unwrap());
        }
    }

    pub fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let mut results: Vec<(f32, usize, SparseVector, usize)> = Vec::new();

        for i in 0..self.num_hash_functions {
            let query_hash = self.minhash(query_vector, i);
            let bucket_index = self.hash_bucket(query_hash);
            let bucket = &self.buckets[bucket_index];

            for (index, vector) in bucket.iter() {
                let similarity = query_vector.cosine_similarity(&vector);
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
                        score: OrderedFloat(*score),
                    },
                    OrderedFloat(*score),
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
                score: OrderedFloat(query_result.score.into_inner()),
            })
            .collect()
    }

    fn minhash(&self, vector: &SparseVector, hash_function_index: usize) -> u64 {
        let mut hasher = DefaultHasher::new();
        let mut min_hash: u64 = u64::MAX;

        for (&index, &value) in vector.indices.iter().zip(vector.values.iter()) {
            let mut combined_hash = hash_function_index as u64;
            index.hash(&mut hasher);
            combined_hash = combined_hash.wrapping_mul(hasher.finish());
            value.hash(&mut hasher);
            combined_hash = combined_hash.wrapping_mul(hasher.finish());

            min_hash = min_hash.min(combined_hash);
            hasher = DefaultHasher::new();
            // let mut element_hash = HashSet::new();
            // element_hash.insert(index);
            // element_hash.insert(value.to_bits() as usize);

            // for item in element_hash {
            //     item.hash(&mut hasher);
            // }

            // let hash = hasher.finish() ^ (hash_function_index as u64);
            // min_hash = min_hash.min(hash);
            // hasher = DefaultHasher::new();
        }

        min_hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsh_index_with_min_hash_simple() {
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

        let mut index = LSHIndex::new(4, 4);
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

        assert!(false);
    }

    #[test]
    fn test_lsh_index_with_min_hash_complex() {
        let mut index = LSHIndex::new(10, 4);

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

        assert!(false);
    }
}
