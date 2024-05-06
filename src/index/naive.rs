use crate::data::HighDimVector;

use super::{DistanceMetric, Index};

struct NaiveNode {
    index: usize,
    distance: f64,
}

pub struct NaiveIndex {
    vectors: Vec<HighDimVector>,
    indexed_vectors: Vec<HighDimVector>,
    metric: DistanceMetric,
}

impl Index for NaiveIndex {
    fn new(metric: DistanceMetric) -> Self {
        NaiveIndex {
            vectors: Vec::new(),
            indexed_vectors: Vec::new(),
            metric,
        }
    }

    fn add_vector(&mut self, vector: HighDimVector) {
        self.vectors.push(vector);
    }

    fn build(&mut self) {
        for vector in self.vectors.clone() {
            self.indexed_vectors.push(vector.clone());
        }
    }

    fn search(&self, query_vector: &HighDimVector, k: usize) -> Vec<HighDimVector> {
        let mut results = self
            .indexed_vectors
            .iter()
            .map(|vector| {
                let distance = query_vector.distance(vector, self.metric);
                NaiveNode {
                    index: vector.id,
                    distance,
                }
            })
            .collect::<Vec<NaiveNode>>();

        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
            .iter()
            .take(k)
            .map(|node| self.indexed_vectors[node.index].clone())
            .collect()
    }
}
