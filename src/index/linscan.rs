use std::collections::{BinaryHeap, HashMap};

use ordered_float::OrderedFloat;

use super::{DistanceMetric, SparseIndex, SparseVector};

#[derive(Debug)]
pub struct LinScanIndex {
    inverted_index: HashMap<usize, Vec<SparseVector>>,
    num_vectors: usize,
    metric: DistanceMetric,
}

impl SparseIndex for LinScanIndex {
    fn new(metric: DistanceMetric) -> Self
    where
        Self: Sized,
    {
        LinScanIndex {
            inverted_index: HashMap::new(),
            num_vectors: 0,
            metric,
        }
    }

    fn add_vector(&mut self, vector: &SparseVector) {
        for i in 0..vector.indices.len() {
            let index = vector.indices[i];
            let value = vector.values[i];
            self.inverted_index
                .entry(index)
                .or_default()
                .push(SparseVector {
                    indices: vec![self.num_vectors],
                    values: vec![value],
                });
        }

        self.num_vectors += 1;
    }

    fn build(&mut self) {
        // LinScan does not need to build an index
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<SparseVector> {
        let mut scores = Vec::with_capacity(self.num_vectors);
        scores.resize(self.num_vectors, 0f32);

        for i in 0..query_vector.indices.len() {
            let index = query_vector.indices[i];
            let value = query_vector.values[i];

            // TODO: Use DistanceMetric to calculate the score
            self.inverted_index
                .get(&index)
                .unwrap_or(&Vec::new())
                .iter()
                .for_each(|vector| {
                    scores[vector.indices[0] as usize] += (value * vector.values[0]).into_inner()
                });
        }

        let mut heap: BinaryHeap<SparseVector> = BinaryHeap::new();

        let mut threshold = f32::MIN;
        for (id, score) in scores.iter().enumerate() {
            if *score > threshold {
                heap.push(SparseVector {
                    indices: vec![id],
                    values: vec![OrderedFloat(*score)],
                });
                if heap.len() > k {
                    threshold = heap.pop().unwrap().values[0].into_inner();
                }
            }
        }

        let mut results = Vec::with_capacity(k);
        while let Some(entry) = heap.pop() {
            results.push(entry);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linscan() {
        let mut index = LinScanIndex::new(DistanceMetric::Cosine);
        let v0 = SparseVector {
            indices: vec![13, 5],
            values: vec![
                ordered_float::OrderedFloat(0.3),
                ordered_float::OrderedFloat(0.8),
            ],
        };
        let v1 = SparseVector {
            indices: vec![13, 5],
            values: vec![
                ordered_float::OrderedFloat(0.6),
                ordered_float::OrderedFloat(0.4),
            ],
        };
        let query_vector = SparseVector {
            indices: vec![13, 5],
            values: vec![
                ordered_float::OrderedFloat(0.5),
                ordered_float::OrderedFloat(0.5),
            ],
        };

        index.add_vector(&v0);
        index.add_vector(&v1);

        let result = index.search(&query_vector, 4);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].indices[0], 1);
        assert_eq!(result[1].indices[0], 0);

        assert_eq!(result[0].values[0].into_inner(), 0.5);
        assert_eq!(result[1].values[0].into_inner(), 0.55);
    }
}
