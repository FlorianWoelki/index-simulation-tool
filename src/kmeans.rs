use rand::{seq::SliceRandom, Rng};

use std::collections::BTreeMap;

use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, SeedableRng};

use crate::data::SparseVector;

pub fn kmeans(
    vectors: &Vec<SparseVector>,
    num_clusters: usize,
    iterations: usize,
    tolerance: f32,
    random_seed: u64,
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
                let distance = node.euclidean_distance(center);

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

        for (i, center) in centers.iter_mut().enumerate() {
            let mut new_indices = Vec::new();
            let mut new_values = Vec::new();
            for (&index, &sum_value) in new_centers[i].iter() {
                new_indices.push(index);
                new_values.push(OrderedFloat(sum_value / counts[i] as f32));
            }
            center.indices = new_indices;
            center.values = new_values;
        }
    }

    return centers;
}
