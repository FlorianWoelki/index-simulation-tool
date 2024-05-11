use rand::seq::SliceRandom;

use crate::{data::HighDimVector, index::DistanceMetric};

/// Performs k-means clustering on a set of high-dimensional vectors.
///
/// # Arguments
///
/// * `k` - The number of clusters to create.
/// * `epoch` - The number of iterations to perform.
/// * `nodes` - A reference to a vector of high-dimensional vectors to be clustered.
/// * `metric` - The distance metric to use for calculating distances between vectors.
/// * `rng` - A mutable reference to a random number generator, used for initializing the cluster vectors.
///
/// # Returns
///
/// A vector of indices representing the closest node to each cluster center.
pub fn kmeans(
    k: usize,
    epoch: usize,
    nodes: &Vec<HighDimVector>,
    metric: DistanceMetric,
    rng: &mut impl rand::Rng,
) -> Vec<usize> {
    if nodes.is_empty() {
        return vec![];
    }

    // Initializes the cluster centers by randomly selecting `k` nodes from the input vector.
    let mut centers: Vec<HighDimVector> = nodes.choose_multiple(rng, k).cloned().collect();
    let mut assignments = vec![0; nodes.len()];

    for _ in 0..epoch {
        for (i, node) in nodes.iter().enumerate() {
            let mut closest = usize::MAX;
            let mut closest_distance = f64::MAX;

            for (j, center) in centers.iter().enumerate() {
                let distance = node.distance(center, metric);

                if distance < closest_distance {
                    closest = j;
                    closest_distance = distance;
                }
            }

            assignments[i] = closest;
        }

        // Recalculate the cluster centers based on the new assignments.
        let mut new_centers: Vec<Vec<f64>> = vec![vec![0.0; centers[0].dimensions.len()]; k];
        let mut counts = vec![0; k];

        for (node, &cluster) in nodes.iter().zip(assignments.iter()) {
            for (i, dim) in node.dimensions.iter().enumerate() {
                new_centers[cluster][i] += dim;
            }
            counts[cluster] += 1;
        }

        for (i, center) in centers.iter_mut().enumerate() {
            for j in 0..center.dimensions.len() {
                center.dimensions[j] = new_centers[i][j] / counts[i] as f64;
            }
        }
    }

    // Find the closest node to each cluster center.
    let closest_node_indices = centers.iter().map(|center| {
        let mut closest_index = 0;
        let mut closest_distance = f64::MAX;

        nodes.iter().enumerate().for_each(|(i, node)| {
            let distance = node.distance(center, metric);
            if distance < closest_distance {
                closest_index = i;
                closest_distance = distance;
            }
        });

        closest_index
    });

    closest_node_indices.collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    #[test]
    fn test_single_node() {
        let nodes = vec![HighDimVector::new(0, vec![1.0, 2.0, 3.0])];
        let result = kmeans(
            1,
            10,
            &nodes,
            DistanceMetric::Euclidean,
            &mut rand::thread_rng(),
        );
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_multiple_nodes_multiple_clusters() {
        let nodes = vec![
            HighDimVector::new(0, vec![1.0, 2.0, 3.0]),
            HighDimVector::new(1, vec![4.0, 5.0, 6.0]),
            HighDimVector::new(2, vec![7.0, 8.0, 9.0]),
            HighDimVector::new(3, vec![10.0, 11.0, 12.0]),
        ];
        let result = kmeans(
            2,
            10,
            &nodes,
            DistanceMetric::Euclidean,
            &mut rand::thread_rng(),
        );
        assert!(result.len() == 2);
        assert!(result.contains(&0));
        assert!(result.contains(&2));
    }

    #[test]
    fn test_convergence() {
        let nodes = vec![
            HighDimVector::new(0, vec![1.0, 2.0, 3.0]),
            HighDimVector::new(1, vec![4.0, 5.0, 6.0]),
            HighDimVector::new(2, vec![7.0, 8.0, 9.0]),
            HighDimVector::new(3, vec![10.0, 11.0, 12.0]),
        ];
        let mut rng = StdRng::seed_from_u64(42);
        let result1 = kmeans(2, 10, &nodes, DistanceMetric::Euclidean, &mut rng);
        let result2 = kmeans(2, 100, &nodes, DistanceMetric::Euclidean, &mut rng);

        let unique_assignments = result1
            .iter()
            .zip(result2.iter())
            .filter(|&(a, b)| a != b)
            .collect::<HashSet<_>>();

        println!("{:?}", unique_assignments.len());
        assert!(unique_assignments.len() <= 2);
    }
}
