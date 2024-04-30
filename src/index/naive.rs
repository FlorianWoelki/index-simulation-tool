use crate::data::HighDimVector;

use super::{calculate_distance, DistanceMetric, Index};

pub struct NaiveIndex {
    vectors: Vec<HighDimVector>,
    metric: DistanceMetric,
}

impl Index for NaiveIndex {
    fn new(metric: DistanceMetric) -> Self {
        NaiveIndex {
            vectors: Vec::new(),
            metric,
        }
    }

    fn add_vector(&mut self, vector: HighDimVector) {
        self.vectors.push(vector);
    }

    fn find_nearest(&self, query: &HighDimVector) -> Option<&HighDimVector> {
        self.vectors.iter().min_by(|a, b| {
            calculate_distance(a, query, self.metric)
                .partial_cmp(&calculate_distance(b, query, self.metric))
                .unwrap()
        })
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &HighDimVector> + '_> {
        Box::new(self.vectors.iter())
    }
}
