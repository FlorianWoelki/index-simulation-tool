use crate::index::DistanceMetric;

pub mod generator;

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
    }
}
