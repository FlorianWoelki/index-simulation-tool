use crate::data::HighDimVector;

use super::{DistanceMetric, Index};

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

    fn indexed_data(&self) -> &Vec<HighDimVector> {
        &self.indexed_vectors
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }
}
