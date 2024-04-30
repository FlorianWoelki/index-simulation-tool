use std::collections::HashSet;

use crate::{data::HighDimVector, index::DistanceMetric};

use super::{Query, QueryResult};

struct Node {
    vector: HighDimVector,
    neighbors: Vec<usize>,
}

pub struct HNSWQuery {
    query_vector: HighDimVector,
    k: usize,
}

impl Query for HNSWQuery {
    fn new(query_vector: HighDimVector, k: usize) -> Self {
        HNSWQuery { query_vector, k }
    }

    fn execute(&self, data: &Vec<HighDimVector>, metric: DistanceMetric) -> Vec<QueryResult> {
        if data.is_empty() {
            return vec![];
        }

        let nodes: Vec<Node> = data
            .iter()
            .map(|vector| Node {
                vector: vector.clone(),
                neighbors: vec![],
            })
            .collect();

        let entry_point_index = 0;
        let mut current_best = &nodes[entry_point_index];
        let mut visited = HashSet::new();
        visited.insert(entry_point_index);

        loop {
            let mut found_better = false;

            for &neighbor_index in &current_best.neighbors {
                if !visited.contains(&neighbor_index) {
                    let neighbor = &nodes[neighbor_index];
                    let distance = self.query_vector.distance(&neighbor.vector, metric);

                    if distance < self.query_vector.distance(&current_best.vector, metric) {
                        current_best = neighbor;
                        visited.insert(neighbor_index);
                        found_better = true;
                    }
                }
            }

            if !found_better {
                break;
            }
        }

        let mut results: Vec<QueryResult> = current_best
            .neighbors
            .iter()
            .map(|&index| {
                let neighbor = &nodes[index];
                let distance = self.query_vector.distance(&neighbor.vector, metric);
                QueryResult { index, distance }
            })
            .collect();

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
        let query_vector = create_vector(vec![1.0, 2.0, 3.0]);
        let query = HNSWQuery::new(query_vector, 2);
        let data = vec![];

        let result = query.execute(&data, DistanceMetric::Euclidean);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_no_neighbors() {
        let query_vector = create_vector(vec![1.0, 2.0]);
        let data = vec![create_vector(vec![2.0, 3.0])];
        let query = HNSWQuery::new(query_vector, 1);

        let results = query.execute(&data, DistanceMetric::Euclidean);

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_single_element_with_self_neighbor() {
        let query_vector = create_vector(vec![1.0, 2.0]);
        let data = vec![query_vector.clone()];
        let query = HNSWQuery::new(query_vector, 1);

        let results = query.execute(&data, DistanceMetric::Euclidean);

        assert_eq!(results.len(), 0);
    }
}
