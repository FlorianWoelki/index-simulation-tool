use crate::data::HighDimVector;

use super::{DistanceMetric, Index};

struct NaiveNode {
    index: usize,
    distance: f32,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_vector() {
        let mut index = NaiveIndex::new(DistanceMetric::Euclidean);
        let v0 = HighDimVector::new(0, vec![1.0, 2.0, 3.0]);

        index.add_vector(v0.clone());

        assert_eq!(index.vectors.len(), 1);
        assert_eq!(index.vectors[0], v0);
    }

    #[test]
    fn test_build() {
        let mut index = NaiveIndex::new(DistanceMetric::Euclidean);
        let v0 = HighDimVector::new(0, vec![1.0, 2.0, 3.0]);
        let v1 = HighDimVector::new(1, vec![4.0, 5.0, 6.0]);

        index.add_vector(v0.clone());
        index.add_vector(v1.clone());
        index.build();

        assert_eq!(index.indexed_vectors.len(), 2);
        assert_eq!(index.indexed_vectors[0], v0);
        assert_eq!(index.indexed_vectors[1], v1);
    }

    #[test]
    fn test_search() {
        let mut index = NaiveIndex::new(DistanceMetric::Euclidean);
        let v0 = HighDimVector::new(0, vec![1.0, 2.0]);
        let v1 = HighDimVector::new(1, vec![3.0, 4.0]);

        index.add_vector(v0.clone());
        index.add_vector(v1.clone());
        index.build();

        let query_vector = HighDimVector::new(2, vec![2.0, 3.0]);
        let results = index.search(&query_vector, 1);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], v0);
    }
}
