use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
};

use ordered_float::OrderedFloat;

use super::{DistanceMetric, QueryResult, SparseIndex, SparseVector};

#[derive(Debug)]
pub struct LinScanIndex {
    vectors: Vec<SparseVector>,
    inverted_index: HashMap<usize, Vec<(usize, OrderedFloat<f32>)>>,
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
        for (index, value) in vector.indices.iter().zip(vector.values.iter()) {
            self.inverted_index
                .entry(*index)
                .or_default()
                .push((self.vectors.len(), *value));
        }

        self.vectors.push(vector.clone());
    }

    fn build(&mut self) {
        // LinScan does not need to build an index
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let mut scores = vec![0.0; self.vectors.len()];

        for (index, value) in query_vector.indices.iter().zip(query_vector.values.iter()) {
            if let Some(vectors) = self.inverted_index.get(index) {
                for (vec_id, vec_value) in vectors.iter() {
                    scores[*vec_id] += value.into_inner() * vec_value.into_inner();
                }
            }
        }

        let mut heap: BinaryHeap<Reverse<QueryResult>> = BinaryHeap::new();

        for (index, &score) in scores.iter().enumerate() {
            if heap.len() < k || score > heap.peek().unwrap().0.score.into_inner() {
                heap.push(Reverse(QueryResult {
                    //vector: self.vectors[index].clone(),
                    index,
                    score: OrderedFloat(score),
                }));
                if heap.len() > k {
                    heap.pop();
                }
            }
        }

        heap.into_sorted_vec().into_iter().map(|r| r.0).collect()
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
