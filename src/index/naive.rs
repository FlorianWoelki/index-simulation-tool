use crate::{data::HighDimVector, query::QueryResult};

use super::{calculate_distance, DistanceMetric, Index};

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

    fn query(&self, query: &HighDimVector) -> QueryResult {
        let mut nearest = None;
        let mut min_distance = f64::MAX;

        for (i, vector) in self.vectors.iter().enumerate() {
            let distance = calculate_distance(vector, query, self.metric);
            if distance < min_distance {
                min_distance = distance;
                nearest = Some(i);
            }
        }

        QueryResult {
            index: nearest.unwrap_or(0),
            distance: min_distance,
        }
    }

    fn indexed_data(&self) -> &Vec<HighDimVector> {
        &self.indexed_vectors
    }
}
