use rand::{seq::SliceRandom, Rng};

use std::collections::BTreeMap;

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
        centroids = update_centroids(&clusters, &vectors, &mut rng);

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
    clusters: &Vec<Vec<SparseVector>>,
    vectors: &[SparseVector],
    rng: &mut StdRng,
) -> Vec<SparseVector> {
    let mut centroids = Vec::new();
    for cluster in clusters {
        if cluster.is_empty() {
            //let random_index = rng.gen_range(0..vectors.len());
            //centroids.push(vectors[random_index].clone());
            centroids.push(SparseVector {
                indices: vec![],
                values: vec![],
            });
            continue;
        }

        // Using `BTreeMap` because ordering of the keys is important.
        let mut sum_indices = BTreeMap::new();
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
    use super::*;

    use ordered_float::OrderedFloat;

    use crate::data::SparseVector;

    #[test]
    fn test_kmeans_basic() {
        let vectors = vec![
            SparseVector {
                indices: vec![0, 1],
                values: vec![OrderedFloat(1.0), OrderedFloat(1.0)],
            },
            SparseVector {
                indices: vec![0, 1],
                values: vec![OrderedFloat(2.0), OrderedFloat(2.0)],
            },
            SparseVector {
                indices: vec![0, 1],
                values: vec![OrderedFloat(3.0), OrderedFloat(3.0)],
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

        // Check that the number of centroids is equal to num_clusters
        assert_eq!(centroids.len(), num_clusters);

        // Check that each centroid has the same number of indices and values
        for centroid in &centroids {
            assert_eq!(centroid.indices.len(), centroid.values.len());
        }

        // Check that the sum of the squared distances from each vector to its nearest centroid is minimized
        let mut total_distance = 0.0;
        for vector in &vectors {
            let mut min_distance = f32::MAX;
            for centroid in &centroids {
                let distance = vector.euclidean_distance(centroid);
                if distance < min_distance {
                    min_distance = distance;
                }
            }
            total_distance += min_distance * min_distance;
        }
        assert!(total_distance < 1.0);

        // Check that the centroids are not all the same
        let mut all_same = true;
        for i in 1..centroids.len() {
            if centroids[i] != centroids[0] {
                all_same = false;
                break;
            }
        }
        assert!(!all_same);
    }

    #[test]
    fn test_update_centroids() {
        let mut rng = StdRng::seed_from_u64(42);
        let clusters = vec![vec![
            SparseVector {
                indices: vec![0, 1],
                values: vec![OrderedFloat(1.0), OrderedFloat(1.0)],
            },
            SparseVector {
                indices: vec![0, 1],
                values: vec![OrderedFloat(2.0), OrderedFloat(2.0)],
            },
        ]];
        let vectors = vec![
            SparseVector {
                indices: vec![0, 1],
                values: vec![OrderedFloat(1.0), OrderedFloat(1.0)],
            },
            SparseVector {
                indices: vec![0, 1],
                values: vec![OrderedFloat(2.0), OrderedFloat(2.0)],
            },
        ];

        let centroids = update_centroids(&clusters, &vectors, &mut rng);

        // Check that the number of centroids is equal to the number of clusters
        assert_eq!(centroids.len(), clusters.len());

        // Check that each centroid has the same number of indices and values
        for centroid in &centroids {
            assert_eq!(centroid.indices.len(), centroid.values.len());
        }

        // Check that the centroid values are the average of the cluster values
        for (centroid, cluster) in centroids.iter().zip(clusters.iter()) {
            for (index, value) in centroid.indices.iter().zip(centroid.values.iter()) {
                let cluster_values: Vec<f32> = cluster
                    .iter()
                    .filter_map(|vector| {
                        Some(
                            vector.values[vector.indices.iter().position(|&i| i == *index)?]
                                .into_inner(),
                        )
                    })
                    .collect();
                let average: f32 = cluster_values.iter().sum::<f32>() / cluster_values.len() as f32;
                assert_eq!(*value, OrderedFloat(average));
            }
        }
    }

    #[test]
    fn test_init_centroids() {
        let mut rng = StdRng::seed_from_u64(42);
        let vectors = vec![
            SparseVector {
                indices: vec![0, 1],
                values: vec![OrderedFloat(1.0), OrderedFloat(1.0)],
            },
            SparseVector {
                indices: vec![0, 1],
                values: vec![OrderedFloat(2.0), OrderedFloat(2.0)],
            },
        ];
        let num_clusters = 2;

        let centroids = init_centroids(&vectors, num_clusters, &mut rng);

        // Check that the number of centroids is equal to num_clusters
        assert_eq!(centroids.len(), num_clusters);

        // Check that each centroid is one of the input vectors
        for centroid in &centroids {
            assert!(vectors.contains(centroid));
        }
    }
}
