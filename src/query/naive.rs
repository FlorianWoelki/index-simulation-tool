use crate::{data::HighDimVector, index::DistanceMetric};

use super::{Query, QueryResult};

pub struct NaiveQuery {
    query_vector: HighDimVector,
    k: usize,
}

impl Query for NaiveQuery {
    fn new(query_vector: HighDimVector, k: usize) -> Self {
        NaiveQuery { query_vector, k }
    }

    fn execute(&self, data: &Vec<HighDimVector>, metric: DistanceMetric) -> Vec<QueryResult> {
        let mut results = data
            .iter()
            .enumerate()
            .map(|(index, vector)| {
                let distance = self.query_vector.distance(vector, metric);
                QueryResult { index, distance }
            })
            .collect::<Vec<QueryResult>>();

        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.into_iter().take(self.k).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_vector(data: Vec<f64>) -> HighDimVector {
        HighDimVector::new(data)
    }

    #[test]
    fn test_empty_data() {
        let query_vector = create_vector(vec![1.0, 2.0]);
        let data = vec![];
        let query = NaiveQuery::new(query_vector, 5);

        let results = query.execute(&data, DistanceMetric::Euclidean);

        assert!(results.is_empty());
    }

    #[test]
    fn test_single_data_point() {
        let query_vector = create_vector(vec![1.0, 2.0]);
        let first_vector = create_vector(vec![2.0, 3.0]);
        let data = vec![first_vector.clone()];
        let query = NaiveQuery::new(query_vector.clone(), 1);

        let results = query.execute(&data, DistanceMetric::Euclidean);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 0);

        let expected_distance = query_vector.distance(&first_vector, DistanceMetric::Euclidean);
        assert_eq!(results[0].distance, expected_distance);
    }

    #[test]
    fn test_multiple_data_points() {
        let query_vector = create_vector(vec![1.0, 2.0]);
        let first_vector = create_vector(vec![2.0, 3.0]);
        let second_vector = create_vector(vec![3.0, 4.0]);
        let data = vec![first_vector.clone(), second_vector.clone()];
        let query = NaiveQuery::new(query_vector.clone(), 2);

        let results = query.execute(&data, DistanceMetric::Euclidean);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].index, 0);
        assert_eq!(results[1].index, 1);

        let expected_distance = query_vector.distance(&first_vector, DistanceMetric::Euclidean);
        assert_eq!(results[0].distance, expected_distance);

        let expected_distance = query_vector.distance(&second_vector, DistanceMetric::Euclidean);
        assert_eq!(results[1].distance, expected_distance);
    }
}
