use std::collections::{BinaryHeap, HashMap};

use ordered_float::OrderedFloat;

use super::{DistanceMetric, QueryResult, SparseIndex, SparseVector};

#[derive(Debug)]
pub struct LinScanIndex {
    vectors: Vec<SparseVector>,
    inverted_index: HashMap<usize, Vec<(usize, f32)>>,
    metric: DistanceMetric,
}

impl SparseIndex for LinScanIndex {
    fn new(metric: DistanceMetric) -> Self
    where
        Self: Sized,
    {
        LinScanIndex {
            vectors: Vec::new(),
            inverted_index: HashMap::new(),
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
                .push((self.vectors.len(), value.into_inner()));
        }

        self.vectors.push(vector.clone());
    }

    fn build(&mut self) {
        // LinScan does not need to build an index
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let mut scores = vec![0.0; self.vectors.len()];

        for i in 0..query_vector.indices.len() {
            let index = query_vector.indices[i];
            let value = query_vector.values[i];

            if let Some(vectors) = self.inverted_index.get(&index) {
                for (vec_id, vec_value) in vectors.iter() {
                    // TODO: Use DistanceMetric to calculate the score
                    scores[*vec_id] += (value * vec_value).into_inner();
                }
            }
        }

        let mut heap: BinaryHeap<QueryResult> = BinaryHeap::new();

        for (id, score) in scores.iter().enumerate() {
            if *score > 0.0 {
                heap.push(QueryResult {
                    vector: self.vectors[id].clone(),
                    score: OrderedFloat(*score),
                });
                if heap.len() > k {
                    heap.pop();
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
        assert_eq!(result[0].score.into_inner(), 0.5);
        assert_eq!(result[1].score.into_inner(), 0.55);
    }
}
