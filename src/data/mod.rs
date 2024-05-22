use std::collections::HashSet;

use ordered_float::OrderedFloat;

use crate::index::DistanceMetric;

pub mod generator_dense;
pub mod generator_sparse;

#[derive(Debug, Clone, PartialEq)]
pub struct HighDimVector {
    pub id: usize,
    pub dimensions: Vec<f64>,
}

impl HighDimVector {
    pub fn new(id: usize, dimensions: Vec<f64>) -> Self {
        HighDimVector { id, dimensions }
    }

    pub fn distance(&self, other: &HighDimVector, metric: DistanceMetric) -> f64 {
        match metric {
            DistanceMetric::Euclidean => self
                .dimensions
                .iter()
                .zip(other.dimensions.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt(),

            DistanceMetric::Manhattan => self
                .dimensions
                .iter()
                .zip(other.dimensions.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>(),

            DistanceMetric::Cosine => {
                let dot_product: f64 = self
                    .dimensions
                    .iter()
                    .zip(other.dimensions.iter())
                    .map(|(x, y)| x * y)
                    .sum();
                let norm_a: f64 = self
                    .dimensions
                    .iter()
                    .map(|x| x.powi(2))
                    .sum::<f64>()
                    .sqrt();
                let norm_b: f64 = other
                    .dimensions
                    .iter()
                    .map(|x| x.powi(2))
                    .sum::<f64>()
                    .sqrt();
                1.0 - dot_product / (norm_a * norm_b)
            }

            DistanceMetric::DotProduct => self
                .dimensions
                .iter()
                .zip(other.dimensions.iter())
                .map(|(x, y)| x * y)
                .sum::<f64>(),

            DistanceMetric::Jaccard => {
                let set1: HashSet<_> = self.dimensions.iter().cloned().map(OrderedFloat).collect();
                let set2: HashSet<_> = other.dimensions.iter().cloned().map(OrderedFloat).collect();

                let intersection_len = set1.intersection(&set2).count() as f64;
                let union_len = set1.union(&set2).count() as f64;

                let similarity = intersection_len / union_len;
                let distance = 1.0 - similarity;

                distance
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance() {
        let a = HighDimVector::new(1, vec![1.0, 2.0, 3.0]);
        let b = HighDimVector::new(2, vec![4.0, 5.0, 6.0]);

        assert_eq!(a.distance(&b, DistanceMetric::Euclidean), 5.196152422706632);
        assert_eq!(a.distance(&b, DistanceMetric::Manhattan), 9.0);
        assert_eq!(a.distance(&b, DistanceMetric::Cosine), 0.025368153802923787);
        assert_eq!(a.distance(&b, DistanceMetric::DotProduct), 32.0);
    }

    #[test]
    fn test_distance_jaccard() {
        let a = HighDimVector::new(1, vec![0.0, 1.0, 2.0, 5.0, 6.0]);
        let b = HighDimVector::new(2, vec![0.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0]);

        assert_eq!(a.distance(&b, DistanceMetric::Jaccard), 0.6666666666666667);
    }
}
