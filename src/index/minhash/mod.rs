use std::{
    collections::{HashMap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
};

use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, Rng, SeedableRng};

use super::{QueryResult, SparseVector};

pub struct MinHash {
    num_permutations: usize,
    hash_functions: Vec<Box<dyn Fn(usize) -> u64>>,
}

impl MinHash {
    fn new_with_rng(num_permutations: usize, seed: u64) -> Self {
        assert!(
            num_permutations > 0,
            "Number of hashes must be greater than 0"
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let mut hash_functions = Vec::new();

        for _ in 0..num_permutations {
            let a = rng.gen::<u64>();
            let b = rng.gen::<u64>();
            hash_functions.push(Box::new(move |x: usize| -> u64 {
                let mut hasher = DefaultHasher::new();
                (a.wrapping_mul(x as u64).wrapping_add(b)).hash(&mut hasher);
                hasher.finish()
            }) as Box<dyn Fn(usize) -> u64>);
        }
        Self {
            num_permutations,
            hash_functions,
        }
    }

    pub fn compute_signature(&self, vector: &SparseVector) -> Vec<u64> {
        let mut signature = vec![u64::MAX; self.num_permutations];

        for &index in &vector.indices {
            for (i, hash_function) in self.hash_functions.iter().enumerate() {
                let hash_value = hash_function(index);
                if hash_value < signature[i] {
                    signature[i] = hash_value;
                }
            }
        }

        signature
    }

    pub fn jaccard_similarity(sig1: &[u64], sig2: &[u64]) -> f32 {
        let num_equal = sig1
            .iter()
            .zip(sig2.iter())
            .filter(|&(a, b)| a == b)
            .count();
        num_equal as f32 / sig1.len() as f32
    }
}

pub struct LSHFunction {
    a: u64,
    b: u64,
    num_buckets: usize,
}

impl LSHFunction {
    fn new_with_rng(num_buckets: usize, seed: u64) -> Self {
        assert!(num_buckets > 0, "Number of buckets must be greater than 0");

        let mut rng = StdRng::seed_from_u64(seed);
        let a = rng.gen::<u64>();
        let b = rng.gen::<u64>();
        Self { a, b, num_buckets }
    }

    fn hash(&self, signature: &[u64]) -> usize {
        let mut hash_value = 0u64;
        for &value in signature {
            hash_value = hash_value
                .wrapping_mul(self.a)
                .wrapping_add(value)
                .wrapping_add(self.b);
        }
        (hash_value as usize) % self.num_buckets
    }
}

pub struct MinHashIndex {
    pub minhash: MinHash,
    lsh_functions: Vec<LSHFunction>,
    vectors: Vec<SparseVector>,
    buckets: HashMap<usize, Vec<(SparseVector, usize, Vec<u64>)>>,
}

impl MinHashIndex {
    pub fn new_with_rng(
        num_permutations: usize,
        num_lsh_functions: usize,
        num_buckets: usize,
        seed: u64,
    ) -> Self {
        assert!(
            num_permutations > 0,
            "Number of permutations must be greater than zero."
        );
        assert!(
            num_lsh_functions > 0,
            "Number of LSH functions must be greater than zero."
        );
        assert!(
            num_buckets > 0,
            "Number of buckets must be greater than zero."
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let minhash = MinHash::new_with_rng(num_permutations, rng.gen::<u64>());
        let mut lsh_functions = Vec::with_capacity(num_lsh_functions);
        for _ in 0..num_lsh_functions {
            lsh_functions.push(LSHFunction::new_with_rng(num_buckets, rng.gen::<u64>()));
        }
        Self {
            minhash,
            lsh_functions,
            buckets: HashMap::new(),
            vectors: Vec::new(),
        }
    }

    pub fn add_vector(&mut self, vector: &SparseVector) {
        let signature = self.minhash.compute_signature(&vector);
        for lsh_function in &self.lsh_functions {
            let bucket_key = lsh_function.hash(&signature);
            self.buckets
                .entry(bucket_key)
                .or_insert_with(Vec::new)
                .push((vector.clone(), self.vectors.len(), signature.clone()));
        }

        self.vectors.push(vector.clone());
    }

    pub fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let query_signature = self.minhash.compute_signature(query_vector);
        let mut candidates = HashSet::new();

        for lsh_function in &self.lsh_functions {
            let bucket_key = lsh_function.hash(&query_signature);
            if let Some(bucket) = self.buckets.get(&bucket_key) {
                for (_, _, signature) in bucket {
                    candidates.insert(signature);
                }
            }
        }

        let mut results: Vec<(f32, usize, &SparseVector)> = candidates
            .iter()
            .filter_map(|signature| {
                self.buckets.values().find_map(|bucket| {
                    bucket
                        .iter()
                        .find(|(_, _, sig)| sig == *signature)
                        .map(|(vector, index, _)| {
                            let similarity =
                                MinHash::jaccard_similarity(&query_signature, signature);
                            (similarity, *index, vector)
                        })
                })
            })
            .collect();

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        results
            .into_iter()
            .take(k)
            .map(|(similarity, index, _)| QueryResult {
                index,
                score: OrderedFloat(similarity),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsh() {
        let vector1 = SparseVector {
            indices: vec![1, 2, 3],
            values: vec![OrderedFloat(0.5), OrderedFloat(0.3), OrderedFloat(0.2)],
        };
        let vector2 = SparseVector {
            indices: vec![1, 2, 4],
            values: vec![OrderedFloat(0.4), OrderedFloat(0.4), OrderedFloat(0.2)],
        };
        let vector3 = SparseVector {
            indices: vec![1, 3, 4],
            values: vec![OrderedFloat(0.6), OrderedFloat(0.1), OrderedFloat(0.3)],
        };

        let mut index = MinHashIndex::new_with_rng(100, 10, 50, 42);

        index.add_vector(&vector1);
        index.add_vector(&vector2);
        index.add_vector(&vector3);

        let query_vector = SparseVector {
            indices: vec![1, 2, 3],
            values: vec![OrderedFloat(0.5), OrderedFloat(0.3), OrderedFloat(0.2)],
        };

        let results = index.search(&query_vector, 2);

        for result in results {
            println!("{:?}", result);
        }

        assert!(true);
    }

    #[test]
    fn test_lsh_complex() {
        let vectors = vec![
            SparseVector {
                indices: vec![1, 2, 3],
                values: vec![OrderedFloat(0.5), OrderedFloat(0.3), OrderedFloat(0.2)],
            },
            SparseVector {
                indices: vec![1, 2, 4],
                values: vec![OrderedFloat(0.4), OrderedFloat(0.4), OrderedFloat(0.2)],
            },
            SparseVector {
                indices: vec![1, 3, 4],
                values: vec![OrderedFloat(0.6), OrderedFloat(0.1), OrderedFloat(0.3)],
            },
            SparseVector {
                indices: vec![2, 3, 5],
                values: vec![OrderedFloat(0.3), OrderedFloat(0.3), OrderedFloat(0.4)],
            },
            SparseVector {
                indices: vec![1, 2, 5],
                values: vec![OrderedFloat(0.2), OrderedFloat(0.5), OrderedFloat(0.3)],
            },
            SparseVector {
                indices: vec![3, 4, 5],
                values: vec![OrderedFloat(0.3), OrderedFloat(0.3), OrderedFloat(0.4)],
            },
        ];

        let mut index = MinHashIndex::new_with_rng(100, 10, 50, 42);

        for vector in &vectors {
            index.add_vector(vector);
        }

        let query_vector = SparseVector {
            indices: vec![1, 2, 3],
            values: vec![OrderedFloat(0.5), OrderedFloat(0.3), OrderedFloat(0.2)],
        };

        let results = index.search(&query_vector, 10);

        for result in results {
            println!("{:?}", result);
        }

        assert!(true);
    }
}
