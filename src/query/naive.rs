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
