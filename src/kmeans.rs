use rand::seq::SliceRandom;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use ordered_float::{Float, OrderedFloat};
use rand::{rngs::StdRng, SeedableRng};

use crate::{data::SparseVector, index::DistanceMetric};

pub fn kmeans(
    vectors: &Vec<SparseVector>,
    num_clusters: usize,
    iterations: usize,
    tolerance: f32,
    random_seed: u64,
    metric: &DistanceMetric,
) -> Vec<SparseVector> {
    let mut rng = StdRng::seed_from_u64(random_seed);
    if vectors.is_empty() {
        return vec![];
    }

    // Initializes the cluster centers by randomly selecting `k` nodes from the input vector.
    let mut centers: Vec<SparseVector> = vectors
        .choose_multiple(&mut rng, num_clusters)
        .cloned()
        .collect();
    let mut assignments = vec![0; vectors.len()];

    for _ in 0..iterations {
        for (i, node) in vectors.iter().enumerate() {
            let mut closest = usize::MAX;
            let mut closest_distance = f32::MAX;

            for (j, center) in centers.iter().enumerate() {
                let distance = node.distance(center, &metric);

                if distance < closest_distance {
                    closest = j;
                    closest_distance = distance;
                }
            }

            assignments[i] = closest;
        }

        // Recalculate the cluster centers based on the new assignments.
        let mut new_centers: Vec<BTreeMap<usize, f32>> = vec![BTreeMap::new(); num_clusters];
        let mut counts = vec![0; num_clusters];

        for (node, &cluster) in vectors.iter().zip(assignments.iter()) {
            for (index, &value) in node.indices.iter().zip(node.values.iter()) {
                *new_centers[cluster].entry(*index).or_insert(0.0) += value.0;
            }
            counts[cluster] += 1;
        }

        let mut max_change = 0.0;

        for (i, center) in centers.iter_mut().enumerate() {
            if counts[i] == 0 {
                // If a cluster is empty, reinitialize its center to a random vector
                *center = vectors.choose(&mut rng).unwrap().clone();
                continue;
            }

            let mut new_indices = Vec::new();
            let mut new_values = Vec::new();
            for (&index, &sum_value) in new_centers[i].iter() {
                new_indices.push(index);
                new_values.push(OrderedFloat(sum_value / counts[i] as f32));
            }

            let new_center = SparseVector {
                indices: new_indices,
                values: new_values,
            };

            let change = center.distance(&new_center, &metric);
            max_change = max_change.max(change);

            *center = new_center;
        }

        if max_change < tolerance {
            break;
        }
    }

    return centers;
}

pub fn kmeans_parallel(
    vectors: &Vec<SparseVector>,
    num_clusters: usize,
    iterations: usize,
    tolerance: f32,
    random_seed: u64,
    metric: &DistanceMetric,
) -> Vec<SparseVector> {
    let mut rng = StdRng::seed_from_u64(random_seed);
    if vectors.is_empty() {
        return vec![];
    }

    // Initializes the cluster centers by randomly selecting `k` nodes from the input vector.
    let mut centers: Vec<SparseVector> = vectors
        .choose_multiple(&mut rng, num_clusters)
        .cloned()
        .collect();
    let assignments = Mutex::new(vec![0; vectors.len()]);

    for _ in 0..iterations {
        vectors.par_iter().enumerate().for_each(|(i, node)| {
            let (closest, _) = centers
                .par_iter()
                .enumerate()
                .map(|(j, center)| {
                    let distance = node.distance(center, &metric);
                    (j, distance)
                })
                .reduce(
                    || (usize::MAX, f32::MAX),
                    |(min_j, min_dist), (j, dist)| {
                        if dist < min_dist {
                            (j, dist)
                        } else {
                            (min_j, min_dist)
                        }
                    },
                );

            assignments.lock().unwrap()[i] = closest;
        });

        // Recalculate the cluster centers based on the new assignments.
        let new_centers: Vec<Mutex<BTreeMap<usize, f32>>> = (0..num_clusters)
            .map(|_| Mutex::new(BTreeMap::new()))
            .collect();
        let counts = Mutex::new(vec![0; num_clusters]);

        vectors
            .par_iter()
            .zip(assignments.lock().unwrap().par_iter())
            .for_each(|(node, &cluster)| {
                let mut center = new_centers[cluster].lock().unwrap();
                for (index, &value) in node.indices.iter().zip(node.values.iter()) {
                    *center.entry(*index).or_insert(0.0) += value.0;
                }
                drop(center);

                let mut counts = counts.lock().unwrap();
                counts[cluster] += 1;
            });

        let max_change = Arc::new(Mutex::new(0.0));
        let rng = Arc::new(Mutex::new(rng.clone()));

        centers.par_iter_mut().enumerate().for_each(|(i, center)| {
            let counts = counts.lock().unwrap();

            if counts[i] == 0 {
                let mut rng = rng.lock().unwrap();
                *center = vectors.choose(&mut *rng).unwrap().clone();
                return;
            }

            let mut new_indices = Vec::new();
            let mut new_values = Vec::new();
            let new_center = new_centers[i].lock().unwrap();
            for (&index, &sum_value) in new_center.iter() {
                new_indices.push(index);
                new_values.push(OrderedFloat(sum_value / counts[i] as f32));
            }

            drop(counts);
            drop(new_center);

            let new_center = SparseVector {
                indices: new_indices,
                values: new_values,
            };

            let change = center.distance(&new_center, &metric);
            {
                let mut max_change = max_change.lock().unwrap();
                if change > *max_change {
                    *max_change = change;
                }
            }

            *center = new_center;
        });

        let max_change = *max_change.lock().unwrap();
        if max_change < tolerance {
            break;
        }
    }

    return centers;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sparse_vector(indices: Vec<usize>, values: Vec<f32>) -> SparseVector {
        SparseVector {
            indices,
            values: values.into_iter().map(OrderedFloat).collect(),
        }
    }

    #[test]
    fn test_kmeans_convergence_with_tolerance() {
        let vectors = vec![
            create_sparse_vector(vec![0, 1], vec![1.0, 2.0]),
            create_sparse_vector(vec![0, 1], vec![1.5, 1.8]),
            create_sparse_vector(vec![0, 1], vec![5.0, 8.0]),
            create_sparse_vector(vec![0, 1], vec![8.0, 8.0]),
        ];

        let num_clusters = 2;
        let iterations = 100;
        let tolerance = 0.1;
        let random_seed = 42;

        let centers = kmeans_parallel(
            &vectors,
            num_clusters,
            iterations,
            tolerance,
            random_seed,
            &DistanceMetric::Euclidean,
        );

        assert_eq!(centers.len(), num_clusters);

        assert!(centers
            .iter()
            .any(|c| c.indices == vec![0, 1]
                && c.values == vec![OrderedFloat(1.25), OrderedFloat(1.9)]));
        assert!(centers
            .iter()
            .any(|c| c.indices == vec![0, 1]
                && c.values == vec![OrderedFloat(6.5), OrderedFloat(8.0)]));
    }

    #[test]
    fn test_kmeans_no_convergence_with_high_tolerance() {
        let vectors = vec![
            create_sparse_vector(vec![0, 1], vec![1.0, 2.0]),
            create_sparse_vector(vec![0, 1], vec![1.5, 1.8]),
            create_sparse_vector(vec![0, 1], vec![5.0, 8.0]),
            create_sparse_vector(vec![0, 1], vec![8.0, 8.0]),
        ];

        let num_clusters = 2;
        let iterations = 10;
        let tolerance = 10.0;
        let random_seed = 42;

        let centers = kmeans(
            &vectors,
            num_clusters,
            iterations,
            tolerance,
            random_seed,
            &DistanceMetric::Euclidean,
        );
        assert_eq!(centers.len(), num_clusters);

        assert!(centers
            .iter()
            .any(|c| c.indices == vec![0, 1]
                && c.values != vec![OrderedFloat(1.25), OrderedFloat(1.9)]));
        assert!(centers
            .iter()
            .any(|c| c.indices == vec![0, 1]
                && c.values != vec![OrderedFloat(6.5), OrderedFloat(8.0)]));
    }
}
