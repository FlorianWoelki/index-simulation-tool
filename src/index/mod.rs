use crate::{data::HighDimVector, query::QueryResult};

pub mod naive;

#[derive(Debug, Copy, Clone)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
}

pub trait Index {
    fn new(metric: DistanceMetric) -> Self
    where
        Self: Sized;
    fn add_vector(&mut self, vector: HighDimVector);
    fn build(&mut self);
    fn query(&self, query: &HighDimVector) -> QueryResult;
    fn indexed_data(&self) -> &Vec<HighDimVector>;
}

pub fn calculate_distance(a: &HighDimVector, b: &HighDimVector, metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => a
            .dimensions
            .iter()
            .zip(b.dimensions.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt(),

        DistanceMetric::Manhattan => a
            .dimensions
            .iter()
            .zip(b.dimensions.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>(),

        DistanceMetric::Cosine => {
            let dot_product: f64 = a
                .dimensions
                .iter()
                .zip(b.dimensions.iter())
                .map(|(x, y)| x * y)
                .sum();
            let norm_a: f64 = a.dimensions.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            let norm_b: f64 = b.dimensions.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            1.0 - dot_product / (norm_a * norm_b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_distance() {
        let a = HighDimVector::new(vec![1.0, 2.0, 3.0]);
        let b = HighDimVector::new(vec![4.0, 5.0, 6.0]);

        assert_eq!(
            calculate_distance(&a, &b, DistanceMetric::Euclidean),
            5.196152422706632
        );
        assert_eq!(calculate_distance(&a, &b, DistanceMetric::Manhattan), 9.0);
        assert_eq!(
            calculate_distance(&a, &b, DistanceMetric::Cosine),
            0.025368153802923787
        );
    }
}
