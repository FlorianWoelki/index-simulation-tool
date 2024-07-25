use rand::seq::SliceRandom;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, RwLock},
};

use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, SeedableRng};

use crate::{data::SparseVector, index::DistanceMetric};

fn initialize_centers(
    vectors: &[SparseVector],
    num_clusters: usize,
    rng: &mut StdRng,
) -> Vec<SparseVector> {
    vectors
        .choose_multiple(rng, num_clusters)
        .cloned()
        .collect()
}

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
    let mut centers = initialize_centers(vectors, num_clusters, &mut rng);
    let assignments = Arc::new(RwLock::new(vec![0; vectors.len()]));

    for _ in 0..iterations {
        vectors.par_iter().enumerate().for_each(|(i, node)| {
            let (closest, _) = centers
                .par_iter()
                .enumerate()
                .map(|(j, center)| {
                    let distance = node.distance(center, metric);
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

            assignments.write().unwrap()[i] = closest;
        });

        // Recalculate the cluster centers based on the new assignments.
        let new_centers: Vec<Mutex<BTreeMap<usize, f32>>> = (0..num_clusters)
            .map(|_| Mutex::new(BTreeMap::new()))
            .collect();
        let counts = Arc::new(Mutex::new(vec![0; num_clusters]));

        vectors
            .par_iter()
            .zip(assignments.read().unwrap().par_iter())
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

        centers.par_iter_mut().enumerate().for_each(|(i, center)| {
            let counts = counts.lock().unwrap();

            if counts[i] == 0 {
                let mut rng = rng.clone();
                *center = vectors.choose(&mut rng).unwrap().clone();
                return;
            }

            let mut new_indices = Vec::new();
            let mut new_values = Vec::new();
            let new_center = new_centers[i].lock().unwrap();
            for (&index, &sum_value) in new_center.iter() {
                new_indices.push(index);
                new_values.push(OrderedFloat(sum_value / counts[i] as f32));
            }

            let new_center = SparseVector {
                indices: new_indices,
                values: new_values,
            };

            let change = center.distance(&new_center, metric);
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

    centers
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
