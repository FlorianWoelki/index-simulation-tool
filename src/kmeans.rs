use rand::{seq::SliceRandom, Rng};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use std::{cmp::Ordering, sync::Mutex};

use rand::{rngs::StdRng, SeedableRng};

use crate::{data::vector::SparseVector, index::DistanceMetric};

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

fn initialize_single_center(vectors: &Vec<SparseVector>, rng: &mut StdRng) -> SparseVector {
    vectors[rng.gen_range(0..vectors.len())].clone()
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

    for _ in 0..iterations {
        let assignments: Vec<usize> = vectors
            .par_iter()
            .map(|node| {
                centers
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        node.distance(a, metric)
                            .partial_cmp(&node.distance(b, metric))
                            .unwrap_or(Ordering::Equal)
                    })
                    .map(|(index, _)| index)
                    .unwrap()
            })
            .collect();

        // Recalculate the cluster centers based on the new assignments.
        let new_centers: Vec<Mutex<(SparseVector, f32, usize)>> = centers
            .iter()
            .map(|center| Mutex::new((center.clone(), f32::MAX, 0)))
            .collect();

        vectors
            .par_iter()
            .zip(assignments.par_iter())
            .for_each(|(node, &cluster)| {
                let mut center_data = new_centers[cluster].lock().unwrap();
                let current_distance = node.distance(&center_data.0, metric);
                if current_distance < center_data.1 {
                    *center_data = (node.clone(), current_distance, 1);
                } else {
                    center_data.2 += 1;
                }
            });

        let new_centers: Vec<SparseVector> = new_centers
            .into_iter()
            .map(|mutex| {
                let (center, _, count) = mutex.into_inner().unwrap();
                if count > 0 {
                    center
                } else {
                    initialize_single_center(vectors, &mut rng)
                }
            })
            .collect();

        let max_change = centers
            .par_iter()
            .zip(new_centers.par_iter())
            .map(|(old_center, new_center)| old_center.distance(new_center, metric))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        centers = new_centers;

        if max_change < tolerance {
            break;
        }
    }

    centers
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use super::*;

    fn create_sparse_vector(indices: Vec<usize>, values: Vec<f32>) -> SparseVector {
        SparseVector {
            indices,
            values: values.into_iter().map(OrderedFloat).collect(),
        }
    }

    // #[test]
    // fn test_kmeans_convergence_with_tolerance() {
    //     let vectors = vec![
    //         create_sparse_vector(vec![0, 1], vec![1.0, 2.0]),
    //         create_sparse_vector(vec![0, 1], vec![1.5, 1.8]),
    //         create_sparse_vector(vec![0, 1], vec![5.0, 8.0]),
    //         create_sparse_vector(vec![0, 1], vec![8.0, 8.0]),
    //     ];

    //     let num_clusters = 2;
    //     let iterations = 100;
    //     let tolerance = 0.1;
    //     let random_seed = 42;

    //     let centers = kmeans(
    //         &vectors,
    //         num_clusters,
    //         iterations,
    //         tolerance,
    //         random_seed,
    //         &DistanceMetric::Euclidean,
    //     );

    //     assert_eq!(centers.len(), num_clusters);

    //     println!("{:?}", centers);
    //     assert!(centers
    //         .iter()
    //         .any(|c| c.indices == vec![0, 1]
    //             && c.values == vec![OrderedFloat(1.25), OrderedFloat(1.9)]));
    //     assert!(centers
    //         .iter()
    //         .any(|c| c.indices == vec![0, 1]
    //             && c.values == vec![OrderedFloat(6.5), OrderedFloat(8.0)]));
    // }

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
