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

#[cfg(test)]
mod tests {
    use super::*;

    fn create_vector(data: Vec<f64>) -> HighDimVector {
        HighDimVector::new(data)
    }

    #[test]
    fn test_empty_index() {
        let index = NaiveIndex::new(DistanceMetric::Euclidean);
        assert_eq!(index.indexed_data().len(), 0);
    }

    #[test]
    fn test_add_vector() {
        let mut index = NaiveIndex::new(DistanceMetric::Euclidean);
        let vector = create_vector(vec![1.0, 2.0]);

        index.add_vector(vector);
        assert_eq!(index.vectors.len(), 1);
        assert!(index.indexed_vectors.is_empty());
    }

    #[test]
    fn test_build_index() {
        let mut index = NaiveIndex::new(DistanceMetric::Euclidean);
        let v1 = create_vector(vec![1.0, 2.0]);
        let v2 = create_vector(vec![3.0, 4.0]);

        index.add_vector(v1.clone());
        index.add_vector(v2.clone());
        index.build();

        assert_eq!(index.indexed_vectors.len(), 2);
        assert_eq!(index.indexed_vectors[0].dimensions, v1.dimensions);
        assert_eq!(index.indexed_vectors[1].dimensions, v2.dimensions);
    }

    #[test]
    fn test_consistent_metric() {
        let index = NaiveIndex::new(DistanceMetric::Euclidean);
        assert_eq!(index.metric(), DistanceMetric::Euclidean);
    }

    #[test]
    fn test_indexed_data_immutable() {
        let mut index = NaiveIndex::new(DistanceMetric::Euclidean);
        let vector = create_vector(vec![1.0, 2.0]);

        index.add_vector(vector);
        index.build();

        let indexed_data = index.indexed_data().clone();
        assert_eq!(indexed_data.len(), 1);

        index.add_vector(create_vector(vec![7.0, 8.0]));
        assert_eq!(indexed_data.len(), 1);
    }
}
