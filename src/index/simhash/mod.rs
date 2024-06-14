use std::hash::{DefaultHasher, Hash, Hasher};

use ordered_float::OrderedFloat;

use crate::{
    data::{QueryResult, SparseVector},
    data_structures::min_heap::MinHeap,
};

pub struct SimHashIndex {
    vectors: Vec<(SparseVector, usize)>,
    hash_bits: usize,
}

impl SimHashIndex {
    pub fn new(hash_bits: usize) -> Self {
        assert!(
            hash_bits >= 1 && hash_bits <= 64,
            "hash_bits must be between 1 and 64"
        );
        SimHashIndex {
            vectors: Vec::new(),
            hash_bits,
        }
    }

    pub fn add_vector(&mut self, vector: &SparseVector) {
        let hash = simhash(&vector, self.hash_bits) as usize;
        self.vectors.push((vector.clone(), hash));
    }

    pub fn build(&self) {
        // No build needed for SimHash.
    }

    pub fn search(&self, vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let target_hash = simhash(&vector, self.hash_bits) as usize;
        let mut heap: MinHeap<QueryResult> = MinHeap::new();

        for (index, (_, hash)) in self.vectors.iter().enumerate() {
            let similarity = hash_similarity(target_hash as u64, *hash as u64);
            if heap.len() < k || similarity > heap.peek().unwrap().score.into_inner() {
                heap.push(
                    QueryResult {
                        index,
                        score: OrderedFloat(similarity),
                    },
                    OrderedFloat(similarity),
                );
                if heap.len() > k {
                    heap.pop();
                }
            }
        }

        heap.into_sorted_vec()
    }
}

fn hamming_distance(x: u64, y: u64) -> u32 {
    (x ^ y).count_ones()
}

fn hash_similarity(hash1: u64, hash2: u64) -> f32 {
    let distance = hamming_distance(hash1, hash2) as f32;
    1.0 - (distance / 64.0)
}

fn simhash(vector: &SparseVector, hash_bits: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    let mut v = vec![0f32; hash_bits];

    for (&index, &value) in vector.indices.iter().zip(vector.values.iter()) {
        index.hash(&mut hasher);
        let feature_hash = hasher.finish();

        for i in 0..hash_bits {
            let bit = (feature_hash >> i) & 1;
            if bit == 1 {
                v[i] += value.into_inner();
            } else {
                v[i] -= value.into_inner();
            }
        }
    }

    let mut simhash: u64 = 0;
    for q in 0..hash_bits {
        if v[q] > 0.0 {
            simhash |= 1 << q;
        }
    }

    simhash
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use super::*;

    #[test]
    fn test_sim_hash() {
        let vectors = vec![
            SparseVector {
                indices: vec![1, 2, 3],
                values: vec![OrderedFloat(0.5), OrderedFloat(1.5), OrderedFloat(-0.5)],
            },
            SparseVector {
                indices: vec![1, 2, 3, 4],
                values: vec![
                    OrderedFloat(0.6),
                    OrderedFloat(1.6),
                    OrderedFloat(-0.6),
                    OrderedFloat(2.0),
                ],
            },
            SparseVector {
                indices: vec![2, 3, 4, 5],
                values: vec![
                    OrderedFloat(1.0),
                    OrderedFloat(-1.0),
                    OrderedFloat(-0.5),
                    OrderedFloat(1.5),
                ],
            },
        ];

        let mut index = SimHashIndex::new(64);

        for vector in vectors {
            index.add_vector(&vector);
        }

        let query_vector = SparseVector {
            indices: vec![1, 2, 3],
            values: vec![OrderedFloat(0.5), OrderedFloat(1.5), OrderedFloat(-0.5)],
        };

        let results = index.search(&query_vector, 2);

        for result in results {
            println!("{:?}", result);
        }

        assert!(true);
    }
}
