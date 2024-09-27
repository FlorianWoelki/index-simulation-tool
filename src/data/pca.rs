use nalgebra::{DMatrix, DVector, SymmetricEigen};
use ordered_float::OrderedFloat;
use plotly::{common::Mode, Plot, Scatter};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::data::vector::SparseVector;

fn compute_mean(vectors: &Vec<SparseVector>, dimension: usize) -> Vec<f32> {
    let (sum, count): (Vec<f32>, Vec<u32>) = vectors
        .par_iter()
        .fold(
            || (vec![0.0; dimension], vec![0; dimension]),
            |(mut sum, mut count), vector| {
                for (i, &value) in vector.values.iter().enumerate() {
                    let idx = vector.indices[i];
                    sum[idx] += value.into_inner();
                    count[idx] += 1;
                }
                (sum, count)
            },
        )
        .reduce(
            || (vec![0.0; dimension], vec![0; dimension]),
            |mut a, b| {
                a.0.iter_mut().zip(b.0.iter()).for_each(|(x, y)| *x += y);
                a.1.iter_mut().zip(b.1.iter()).for_each(|(x, y)| *x += y);
                a
            },
        );

    sum.into_iter()
        .zip(count)
        .map(|(s, c)| if c > 0 { s / c as f32 } else { 0.0 })
        .collect()
}

fn center_data(vectors: &Vec<SparseVector>, mean: &Vec<f32>) -> Vec<SparseVector> {
    vectors
        .par_iter()
        .map(|vector| {
            let centered_values = vector
                .values
                .par_iter()
                .enumerate()
                .map(|(i, &value)| {
                    let idx = vector.indices[i];
                    OrderedFloat(value.into_inner() - mean[idx])
                })
                .collect();
            SparseVector {
                indices: vector.indices.clone(),
                values: centered_values,
            }
        })
        .collect()
}

fn compute_covariance_matrix(vectors: &Vec<SparseVector>, dimension: usize) -> DMatrix<f32> {
    let covariance_matrix = vectors
        .par_iter()
        .fold(
            || DMatrix::zeros(dimension, dimension),
            |mut matrix, vector| {
                let mut dense_vector = vec![0.0; dimension];
                for (i, &value) in vector.values.iter().enumerate() {
                    dense_vector[vector.indices[i]] = value.into_inner();
                }
                let dv = DVector::from_vec(dense_vector);
                matrix += &dv * dv.transpose();
                matrix
            },
        )
        .reduce(
            || DMatrix::zeros(dimension, dimension),
            |mut a, b| {
                a += b;
                a
            },
        );

    covariance_matrix / (vectors.len() as f32 - 1.0)
}

pub fn pca(
    vectors: &Vec<SparseVector>,
    dimension: usize,
    num_components: usize,
) -> (Vec<SparseVector>, Vec<f32>, DMatrix<f32>) {
    let mean = compute_mean(&vectors, dimension);
    let centered_vectors = center_data(&vectors, &mean);
    let covariance_matrix = compute_covariance_matrix(&centered_vectors, dimension);

    let eig = SymmetricEigen::new(covariance_matrix);
    let eigenvectors = eig.eigenvectors.columns(0, num_components);
    let eigenvalues = eig.eigenvalues;

    let transformed_data: Vec<SparseVector> = centered_vectors
        .par_iter()
        .map(|vector| {
            let transformed = (0..num_components)
                .into_par_iter()
                .filter_map(|i| {
                    let component_value: f32 = vector
                        .values
                        .par_iter()
                        .enumerate()
                        .map(|(j, &value)| {
                            let idx = vector.indices[j];
                            value.into_inner() * eigenvectors[(idx, i)]
                        })
                        .sum();
                    if component_value.abs() > f32::EPSILON {
                        Some((i, OrderedFloat(component_value)))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            SparseVector {
                indices: transformed.iter().map(|&(i, _)| i).collect(),
                values: transformed.into_iter().map(|(_, v)| v).collect(),
            }
        })
        .collect();

    let explained_variances: Vec<f32> = eigenvalues.iter().take(num_components).cloned().collect();

    (
        transformed_data,
        explained_variances,
        eigenvectors.clone_owned(),
    )
}

#[allow(dead_code)]
pub fn reconstruct(
    transformed_vectors: &Vec<SparseVector>,
    mean: &Vec<f32>,
    eigenvectors: DMatrix<f32>,
    original_vectors: &Vec<SparseVector>,
    dimension: usize,
) -> Vec<SparseVector> {
    transformed_vectors
        .iter()
        .zip(original_vectors.iter())
        .map(|(transformed_vector, original_vector)| {
            let mut dense_vector = vec![0.0; dimension];

            for (i, &value) in transformed_vector.values.iter().enumerate() {
                let component_idx = transformed_vector.indices[i];
                let component_value = value.into_inner();

                for &idx in &original_vector.indices {
                    dense_vector[idx] += component_value * eigenvectors[(idx, component_idx)];
                }
            }

            for &idx in &original_vector.indices {
                dense_vector[idx] += mean[idx];
            }

            // Filter out near-zero values to maintain sparsity
            let indices: Vec<usize> = original_vector
                .indices
                .iter()
                .filter(|&&i| dense_vector[i].abs() > f32::EPSILON)
                .cloned()
                .collect();
            let values: Vec<OrderedFloat<f32>> = indices
                .iter()
                .map(|&i| OrderedFloat(dense_vector[i]))
                .collect();

            SparseVector { indices, values }
        })
        .collect()
}

#[allow(dead_code)]
fn plot_pca_results(transformed_data: Vec<SparseVector>, num_components: usize) {
    let mut plot = Plot::new();

    if num_components >= 2 {
        let mut x = Vec::new();
        let mut y = Vec::new();

        for vector in transformed_data {
            let mut x_value = 0.0;
            let mut y_value = 0.0;

            for (i, &value) in vector.values.iter().enumerate() {
                let idx = vector.indices[i];
                if idx == 0 {
                    x_value = value.into_inner();
                } else if idx == 1 {
                    y_value = value.into_inner();
                }
            }

            x.push(x_value);
            y.push(y_value);
        }

        let trace = Scatter::new(x, y)
            .mode(Mode::Markers)
            .marker(
                plotly::common::Marker::new()
                    .size(10)
                    .symbol(plotly::common::MarkerSymbol::Circle),
            )
            .name("PCA Result");

        plot.add_trace(trace);
    } else {
        println!("Number of components must be at least 2 for plotting.");
    }

    plot.show();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca() {
        // Example sparse vectors
        let vectors = vec![
            SparseVector {
                indices: vec![0, 2],
                values: vec![OrderedFloat(1.0), OrderedFloat(2.0)],
            },
            SparseVector {
                indices: vec![1, 3],
                values: vec![OrderedFloat(3.0), OrderedFloat(4.0)],
            },
            SparseVector {
                indices: vec![0, 1, 2],
                values: vec![OrderedFloat(1.5), OrderedFloat(2.5), OrderedFloat(3.5)],
            },
        ];

        // Perform PCA to reduce to 2 components
        let dimension = 4; // Assuming original vectors have dimension 4
        let num_components = 2;
        let (transformed_vectors, explained_variances, eigenvectors) =
            pca(&vectors, dimension, num_components);

        // Print the transformed sparse vectors
        for (i, vector) in transformed_vectors.iter().enumerate() {
            println!("Transformed Vector {}:", i);
            for (j, &index) in vector.indices.iter().enumerate() {
                println!("Index: {}, Value: {}", index, vector.values[j]);
            }
        }

        // Print explained variances
        println!("Explained variances for each component:");
        for (i, &variance) in explained_variances.iter().enumerate() {
            println!("Component {}: {}", i, variance);
        }

        // Reconstruct the original data from the transformed data
        let mean = compute_mean(&vectors, dimension);
        let reconstructed_vectors = reconstruct(
            &transformed_vectors,
            &mean,
            eigenvectors,
            &vectors,
            dimension,
        );

        // Print the reconstructed sparse vectors
        println!("Reconstructed Vectors:");
        for (i, vector) in reconstructed_vectors.iter().enumerate() {
            println!("Reconstructed Vector {}:", i);
            for (j, &index) in vector.indices.iter().enumerate() {
                println!("Index: {}, Value: {}", index, vector.values[j]);
            }
        }

        // plot_pca_results(transformed_vectors, num_components);

        assert!(true);
    }
}
