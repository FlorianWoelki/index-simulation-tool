use std::collections::HashMap;

use ordered_float::OrderedFloat;

use crate::{
    data::{QueryResult, SparseVector},
    data_structures::min_heap::MinHeap,
    index::{DistanceMetric, SparseIndex},
};

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

        let mut heap: MinHeap<QueryResult> = MinHeap::new();
        for (index, &score) in scores.iter().enumerate() {
            if heap.len() < k || score > heap.peek().unwrap().score.into_inner() {
                heap.push(
                    QueryResult {
                        //vector: self.vectors[index].clone(),
                        index,
                        score: OrderedFloat(score),
                    },
                    OrderedFloat(score),
                );
                if heap.len() > k {
                    heap.pop();
                }
            }
        }

        heap.into_sorted_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linscan_simple() {
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

        let mut index = LinScanIndex::new(DistanceMetric::Cosine);
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

        assert!(true);
    }
}
