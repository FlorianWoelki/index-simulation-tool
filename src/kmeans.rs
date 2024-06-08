use rand::{seq::SliceRandom, Rng};

use std::collections::HashMap;

use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, SeedableRng};

use crate::data::SparseVector;

pub fn kmeans(
    vectors: Vec<SparseVector>,
    num_clusters: usize,
    iterations: usize,
    tolerance: f32,
    random_seed: u64,
) -> Vec<SparseVector> {
    let mut rng = StdRng::seed_from_u64(random_seed);
    let mut centroids = init_centroids(&vectors, num_clusters, &mut rng);
    let mut prev_centroids;
    let mut current_iterations = 0;

    loop {
        let mut clusters = vec![Vec::new(); num_clusters];
        for vector in &vectors {
            let mut min_distance = f32::MAX;
            let mut cluster_index = 0;
            for (i, centroid) in centroids.iter().enumerate() {
                let distance = vector.euclidean_distance(centroid);
                if distance < min_distance {
                    min_distance = distance;
                    cluster_index = i;
                }
            }
            clusters[cluster_index].push(vector.clone());
        }

        prev_centroids = centroids.clone();
        centroids = update_centroids(clusters, &vectors, &mut rng);

        let centroid_shift: f32 = centroids
            .iter()
            .zip(prev_centroids.iter())
            .map(|(c1, c2)| c1.euclidean_distance(c2))
            .sum();

        if centroid_shift <= tolerance || current_iterations >= iterations {
            break;
        }

        current_iterations += 1;
    }
    centroids
}

fn update_centroids(
    clusters: Vec<Vec<SparseVector>>,
    vectors: &[SparseVector],
    rng: &mut StdRng,
) -> Vec<SparseVector> {
    let mut centroids = Vec::new();
    for cluster in clusters {
        if cluster.is_empty() {
            let random_index = rng.gen_range(0..vectors.len());
            centroids.push(vectors[random_index].clone());
            /*centroids.push(SparseVector {
            indices: vec![],
            values: vec![],
            });*/
            continue;
        }

        let mut sum_indices = HashMap::new();
        for vector in cluster {
            for (i, &index) in vector.indices.iter().enumerate() {
                let value = vector.values[i].into_inner();
                sum_indices.entry(index).or_insert((0.0, 0)).0 += value;
                sum_indices.entry(index).or_insert((0.0, 0)).1 += 1;
            }
        }

        let mut centroid_indices = Vec::new();
        let mut centroid_values = Vec::new();

        for (index, (sum, count)) in sum_indices {
            centroid_indices.push(index);
            centroid_values.push(OrderedFloat(sum / count as f32));
        }

        let centroid = SparseVector {
            indices: centroid_indices,
            values: centroid_values,
        };
        centroids.push(centroid);
    }
    centroids
}

fn init_centroids(
    vectors: &Vec<SparseVector>,
    num_clusters: usize,
    rng: &mut StdRng,
) -> Vec<SparseVector> {
    let mut centroids = Vec::new();

    let mut indices: Vec<usize> = (0..vectors.len()).collect();
    indices.shuffle(rng);

    for &index in indices.iter().take(num_clusters) {
        centroids.push(vectors[index].clone());
    }
    centroids
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    use ordered_float::OrderedFloat;

    use crate::data::SparseVector;

    #[test]
    fn test_kmeans_basic() {
        let vectors = vec![
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(1.0)],
            },
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(2.0)],
            },
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(3.0)],
            },
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(10.0)],
            },
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(11.0)],
            },
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(12.0)],
            },
        ];

        let num_clusters = 2;
        let iterations = 10;
        let tolerance = 0.01;
        let random_seed = 42;

        let centroids = kmeans(
            vectors.clone(),
            num_clusters,
            iterations,
            tolerance,
            random_seed,
        );

        assert_eq!(centroids.len(), num_clusters);

        for centroid in &centroids {
            assert!(
                !centroid.indices.is_empty(),
                "Centroid indices should not be empty"
            );
            assert!(
                !centroid.values.is_empty(),
                "Centroid values should not be empty"
            );
        }

        // Check that centroids are close to some of the data points
        for centroid in &centroids {
            let mut min_distance = f32::MAX;
            for vector in &vectors {
                let distance = centroid.euclidean_distance(vector);
                if distance < min_distance {
                    min_distance = distance;
                }
            }

            assert!(
                min_distance < 1.0,
                "Centroid is not close to any data point"
            );
        }

        // Ensure no centroids are duplicated
        let unique_centroids: HashSet<_> = centroids.iter().collect();
        assert_eq!(
            unique_centroids.len(),
            centroids.len(),
            "Duplicate centroids found"
        );
    }

    #[test]
    fn test_kmeans_empty_clusters() {
        let vectors = vec![
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(1.0)],
            },
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(2.0)],
            },
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(3.0)],
            },
        ];

        let num_clusters = 5; // More clusters than points
        let iterations = 100;
        let tolerance = 0.01;
        let random_speed = 42;

        let centroids = kmeans(
            vectors.clone(),
            num_clusters,
            iterations,
            tolerance,
            random_speed,
        );

        assert_eq!(centroids.len(), num_clusters);

        for centroid in &centroids {
            assert!(
                !centroid.indices.is_empty() || !centroid.values.is_empty(),
                "Centroid should not be empty"
            );
        }

        // Check that each centroid is close to at least one data point
        for centroid in &centroids {
            let mut min_distance = f32::MAX;
            for vector in &vectors {
                let distance = centroid.euclidean_distance(vector);
                if distance < min_distance {
                    min_distance = distance;
                }
            }
            assert!(
                min_distance < 1.0,
                "Centroid is not close to any data point"
            );
        }
    }

    #[test]
    fn test_kmeans_clusters() {
        let vectors = vec![
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(1.0)],
            },
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(2.0)],
            },
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(3.0)],
            },
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(4.0)],
            },
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(5.0)],
            },
        ];

        let num_clusters = 3;
        let iterations = 100;
        let tolerance = 0.01;
        let random_seed = 42;

        let centroids = kmeans(
            vectors.clone(),
            num_clusters,
            iterations,
            tolerance,
            random_seed,
        );

        assert_eq!(centroids.len(), num_clusters);

        let mut clusters = vec![Vec::new(); num_clusters];
        for vector in &vectors {
            let mut min_distance = f32::MAX;
            let mut cluster_index = 0;
            for (i, centroid) in centroids.iter().enumerate() {
                let distance = vector.euclidean_distance(centroid);
                if distance < min_distance {
                    min_distance = distance;
                    cluster_index = i;
                }
            }
            clusters[cluster_index].push(vector.clone());
        }

        for cluster in &clusters {
            assert!(!cluster.is_empty(), "Cluster should not be empty");
        }

        // Check that each vector is assigned to exactly one cluster
        let mut assigned_vectors = 0;
        for cluster in &clusters {
            assigned_vectors += cluster.len();
        }
        assert_eq!(
            assigned_vectors,
            vectors.len(),
            "All vectors should be assigned to clusters"
        );
    }
}
