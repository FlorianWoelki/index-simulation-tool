use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    sync::RwLock,
};

use rand::Rng;

use crate::data::HighDimVector;

use super::{neighbor::NeighborNode, DistanceMetric, Index};

pub struct LSHIndex {
    pub(super) vectors: Vec<HighDimVector>,
    pub(super) metric: DistanceMetric,
    pub(super) hash_functions: Vec<HashFunction>,
    pub(super) buckets: Vec<RwLock<HashMap<u64, Vec<usize>>>>,
    pub(super) n_items: usize,
    pub(super) n_hash_tables: usize,
    pub(super) n_hash_functions: usize,
}

impl Index for LSHIndex {
    fn new(metric: DistanceMetric) -> Self {
        let n_hash_tables = 5;
        let n_hash_functions = 5;
        LSHIndex {
            vectors: Vec::new(),
            metric,
            hash_functions: (0..n_hash_tables)
                .map(|_| HashFunction::new(n_hash_functions))
                .collect(),
            buckets: (0..n_hash_tables)
                .map(|_| RwLock::new(HashMap::new()))
                .collect(),
            n_items: 0,
            n_hash_tables,
            n_hash_functions,
        }
    }

    fn add_vector(&mut self, vector: HighDimVector) {
        let v_id = self.n_items;
        self.vectors.push(vector.clone());

        for (i, hash_function) in self.hash_functions.iter().enumerate() {
            let hash = hash_function.hash(&vector);
            let mut bucket = self.buckets[i].write().unwrap();
            bucket.entry(hash).or_default().push(v_id);
        }

        self.n_items += 1;
    }

    fn build(&mut self) {}

    fn search(&self, query_vector: &HighDimVector, k: usize) -> Vec<HighDimVector> {
        let mut candidates = HashSet::new();

        for (i, hash_function) in self.hash_functions.iter().enumerate() {
            let hash = hash_function.hash(query_vector);
            if let Some(bucket) = self.buckets[i].read().unwrap().get(&hash) {
                for &id in bucket {
                    candidates.insert(id);
                }
            }
        }

        let mut heap = BinaryHeap::new();

        for &id in &candidates {
            let a: &HighDimVector = &self.vectors[id];
            let distance = a.distance(query_vector, self.metric);
            heap.push(NeighborNode::new(id, distance));
            if heap.len() > k {
                heap.pop();
            }
        }

        let mut results = Vec::with_capacity(k);
        while let Some(neighbor) = heap.pop() {
            results.push(self.vectors[neighbor.id].clone());
        }

        results.reverse();
        results
    }
}

pub struct HashFunction {
    pub(super) projections: Vec<Vec<f32>>,
}

impl HashFunction {
    pub fn new(n_projections: usize) -> Self {
        let mut rng = rand::thread_rng();
        let projections = (0..n_projections)
            .map(|_| {
                (0..n_projections)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();

        HashFunction { projections }
    }

    pub fn hash(&self, vector: &HighDimVector) -> u64 {
        let mut hash = 0u64;
        for projections in &self.projections {
            let dot_product: f32 = vector
                .dimensions
                .iter()
                .zip(projections.iter())
                .map(|(&x, &y)| x * y)
                .sum();
            hash <<= 1;
            if dot_product >= 0.0 {
                hash |= 1;
            }
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adding_vector() {
        let mut index = LSHIndex::new(DistanceMetric::Euclidean);
        let v0 = HighDimVector::new(0, vec![1.0, 2.0, 3.0]);
        index.add_vector(v0.clone());

        assert_eq!(index.n_items, 1);
        assert_eq!(index.vectors.len(), 1);
        assert_eq!(index.vectors[0], v0);
    }

    #[test]
    fn test_search() {
        let mut index = LSHIndex::new(DistanceMetric::Euclidean);

        for i in 0..10 {
            let v = HighDimVector::new(i, vec![i as f32, (i + 1) as f32, (i + 2) as f32]);
            index.add_vector(v);
        }

        let query_vector = HighDimVector::new(99999, vec![10.0, 11.0, 12.0]);
        let result = index.search(&query_vector, 3);

        assert_eq!(result.len(), 3);
    }
}
