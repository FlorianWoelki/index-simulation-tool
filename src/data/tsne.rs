use nalgebra::DMatrix;
use ordered_float::OrderedFloat;
use plotly::{common::Mode, Plot, Scatter};
use rand::{rngs::StdRng, Rng};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

use super::SparseVector;

fn generate_normal(mean: f64, std_dev: f64, rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen();
    let u2: f64 = rng.gen();
    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    mean + z0 * std_dev
}

fn generate_structured_sparse_data(
    n_samples: usize,
    n_features: usize,
    rng: &mut StdRng,
) -> Vec<SparseVector> {
    let mut data = Vec::with_capacity(n_samples);
    let cluster_size = n_samples / 3;

    let cluster_centers = [0.0, 5.0, 10.0];
    let std_dev = 0.5;

    for &center in cluster_centers.iter() {
        for _ in 0..cluster_size {
            let mut indices = Vec::new();
            let mut values = Vec::new();
            for k in 0..n_features {
                let value = generate_normal(center, std_dev, rng);
                if value.abs() > 1e-5 {
                    indices.push(k);
                    values.push(OrderedFloat(value as f32));
                }
            }
            data.push(SparseVector { indices, values });
        }
    }

    data
}

fn tsne(
    data: &[SparseVector],
    n_components: usize,
    perplexity: f64,
    n_iter: usize,
) -> Vec<SparseVector> {
    let n_samples = data.len();
    let mut low_dim_data: Vec<SparseVector> = (0..n_samples)
        .into_par_iter()
        .map(|_| {
            let mut sv = SparseVector {
                indices: vec![],
                values: vec![],
            };
            for i in 0..n_components {
                sv.indices.push(i);
                sv.values
                    .push(OrderedFloat(rand::random::<f32>() * 2.0 - 1.0));
            }
            sv
        })
        .collect();

    let p = compute_pairwise_similarities(data, perplexity);

    for iter in 0..n_iter {
        let q = compute_low_dim_similarities(&low_dim_data);
        let gradient = compute_gradient(&p, &q, &low_dim_data);

        let learning_rate = 200.0 / (1.0 + iter as f64) as f32;
        low_dim_data
            .par_iter_mut()
            .zip(gradient.par_iter())
            .for_each(|(low_dim_vector, grad_vector)| {
                low_dim_vector.add_scaled(grad_vector, -learning_rate);
            });

        let kl_div = kl_divergence(&p, &q);
        println!("Iteration {}: KL divergence = {}", iter, kl_div);
    }

    low_dim_data
}

fn compute_pairwise_similarities(data: &[SparseVector], perplexity: f64) -> DMatrix<f64> {
    let n_samples = data.len();
    let mut p = DMatrix::zeros(n_samples, n_samples);

    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j {
                let dist = data[i].squared_distance(&data[j]) as f64;
                p[(i, j)] = (-dist / (2.0 * perplexity.powi(2))).exp();
            }
        }
    }

    // Symmetrize the matrix
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let avg = (p[(i, j)] + p[(j, i)]) / 2.0;
            p[(i, j)] = avg;
            p[(j, i)] = avg;
        }
    }

    // Normalize each row to sum to 1
    for i in 0..n_samples {
        let sum: f64 = p.row(i).sum();
        p.row_mut(i).iter_mut().for_each(|x| *x /= sum);
    }

    p
}

fn compute_low_dim_similarities(low_dim_data: &[SparseVector]) -> DMatrix<f64> {
    let n_samples = low_dim_data.len();
    let mut q = DMatrix::zeros(n_samples, n_samples);

    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j {
                let dist = low_dim_data[i].squared_distance(&low_dim_data[j]) as f64;
                q[(i, j)] = 1.0 / (1.0 + dist);
            }
        }
        // let sum: f64 = q.row(i).sum();
        // q.row_mut(i).iter_mut().for_each(|x| *x /= sum);
    }

    // Symmetrize the matrix
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let avg = (q[(i, j)] + q[(j, i)]) / 2.0;
            q[(i, j)] = avg;
            q[(j, i)] = avg;
        }
    }

    // Normalize each row to sum to 1
    for i in 0..n_samples {
        let sum: f64 = q.row(i).sum();
        q.row_mut(i).iter_mut().for_each(|x| *x /= sum);
    }

    q
}

fn compute_gradient(
    p: &DMatrix<f64>,
    q: &DMatrix<f64>,
    low_dim_data: &[SparseVector],
) -> Vec<SparseVector> {
    let n_samples = low_dim_data.len();
    let n_components = low_dim_data[0].indices.len();

    (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let mut gradient = SparseVector {
                indices: vec![],
                values: vec![],
            };

            for j in 0..n_samples {
                if i != j {
                    let mut diff = SparseVector {
                        indices: vec![],
                        values: vec![],
                    };
                    for k in 0..n_components {
                        let d = low_dim_data[i].values[k].0 - low_dim_data[j].values[k].0;
                        diff.indices.push(k);
                        diff.values.push(OrderedFloat(d));
                    }
                    let pq_diff = 4.0 * (p[(i, j)] - q[(i, j)]) * q[(i, j)];
                    gradient.add_scaled(&diff, pq_diff as f32);
                }
            }

            gradient
        })
        .collect()
}

fn kl_divergence(p: &DMatrix<f64>, q: &DMatrix<f64>) -> f64 {
    p.iter()
        .zip(q.iter())
        .map(|(&p_i, &q_i)| {
            if p_i > 0.0 && q_i > 0.0 {
                p_i * (p_i / q_i).ln()
            } else {
                0.0
            }
        })
        .sum()
}

fn plot_tsne_result(data: &[SparseVector]) {
    let mut plot = Plot::new();

    let cluster_size = data.len() / 3;
    let colors = vec!["red", "green", "blue"];

    for (i, color) in colors.iter().enumerate() {
        let start = i * cluster_size;
        let end = (i + 1) * cluster_size;

        let x: Vec<f64> = data[start..end]
            .iter()
            .map(|sv| sv.values[0].0 as f64)
            .collect();
        let y: Vec<f64> = data[start..end]
            .iter()
            .map(|sv| sv.values[1].0 as f64)
            .collect();

        let trace = Scatter::new(x, y)
            .name(&format!("Cluster {}", i + 1))
            .mode(Mode::Markers)
            .marker(
                plotly::common::Marker::new()
                    .color(*color)
                    .size(10)
                    .symbol(plotly::common::MarkerSymbol::Circle),
            );

        plot.add_trace(trace);
    }

    plot.show();
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn test_compute_gradient() {
        let low_dim_data = vec![
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(0.0)],
            },
            SparseVector {
                indices: vec![0],
                values: vec![OrderedFloat(1.0)],
            },
        ];
        let p = DMatrix::from_row_slice(2, 2, &[0.0, 0.5, 0.5, 0.0]);
        let q = DMatrix::from_row_slice(2, 2, &[0.0, 0.25, 0.25, 0.0]);

        let gradient = compute_gradient(&p, &q, &low_dim_data);

        assert_eq!(gradient.len(), 2);
        assert_eq!(gradient[0].indices, vec![0]);
        assert_eq!(gradient[1].indices, vec![0]);
        assert!(gradient[0].values[0].0 < 0.0);
        assert!(gradient[1].values[0].0 > 0.0);
    }

    #[test]
    fn test_kl_divergence() {
        let p = DMatrix::from_row_slice(2, 2, &[0.0, 0.5, 0.5, 0.0]);
        let q = DMatrix::from_row_slice(2, 2, &[0.0, 0.25, 0.25, 0.0]);

        let kl_div = kl_divergence(&p, &q);

        assert!(kl_div > 0.0);
    }

    // #[test]
    // fn test_tsne_simple() {
    //     let mut rng = StdRng::seed_from_u64(42);
    //     // Generate some random data
    //     let data = generate_structured_sparse_data(90, 10, &mut rng);

    //     // Run t-SNE
    //     let low_dim_data = tsne(&data, 2, 2.0, 200);

    //     // Print the low-dimensional representation
    //     println!("{:?}", low_dim_data);

    //     plot_tsne_result(&low_dim_data);

    //     assert!(false);
    // }

    // #[test]
    // fn test_tsne_complex() {
    //     let mut rng = StdRng::seed_from_u64(42);

    //     // Generate some random data
    //     let data = generate_structured_sparse_data(90, 100, &mut rng);

    //     // Run t-SNE
    //     let low_dim_data = tsne(&data, 2, 30.0, 1000);

    //     // Print the low-dimensional representation
    //     println!("{:?}", low_dim_data);

    //     plot_tsne_result(&low_dim_data);

    //     assert!(false);
    // }
}
