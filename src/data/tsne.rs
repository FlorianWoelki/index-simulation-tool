use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use plotly::{common::Mode, Plot, Scatter};
use rand::{rngs::StdRng, Rng};

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

    for (i, &center) in cluster_centers.iter().enumerate() {
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
        for (low_dim_vector, grad_vector) in low_dim_data.iter_mut().zip(gradient.iter()) {
            low_dim_vector.add_scaled(grad_vector, -learning_rate);
        }

        let kl_div = kl_divergence(&p, &q);
        println!("Iteration {}: KL divergence = {}", iter, kl_div);
    }

    low_dim_data
}

fn compute_pairwise_similarities(data: &[SparseVector], perplexity: f64) -> Array2<f64> {
    let n_samples = data.len();
    let mut p = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j {
                let dist = data[i].squared_distance(&data[j]) as f64;
                p[[i, j]] = (-dist / (2.0 * perplexity.powi(2))).exp();
            }
        }
        let sum: f64 = p.row(i).sum();
        p.row_mut(i).mapv_inplace(|x| x / sum);
    }

    p
}

fn compute_low_dim_similarities(low_dim_data: &[SparseVector]) -> Array2<f64> {
    let n_samples = low_dim_data.len();
    let mut q = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j {
                let dist = low_dim_data[i].squared_distance(&low_dim_data[j]) as f64;
                q[[i, j]] = 1.0 / (1.0 + dist);
            }
        }
        let sum: f64 = q.row(i).sum();
        q.row_mut(i).mapv_inplace(|x| x / sum);
    }

    q
}

fn compute_gradient(
    p: &Array2<f64>,
    q: &Array2<f64>,
    low_dim_data: &[SparseVector],
) -> Vec<SparseVector> {
    let n_samples = low_dim_data.len();
    let n_components = low_dim_data[0].indices.len();
    let mut gradient = vec![
        SparseVector {
            indices: vec![],
            values: vec![]
        };
        n_samples
    ];

    for i in 0..n_samples {
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
                let pq_diff = (p[[i, j]] - q[[i, j]]) as f32;
                gradient[i].add_scaled(&diff, 4.0 * pq_diff * q[[i, j]] as f32);
            }
        }
    }

    gradient
}

fn kl_divergence(p: &Array2<f64>, q: &Array2<f64>) -> f64 {
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

    // #[test]
    // fn test_tsne_simple() {
    //     let mut rng = StdRng::seed_from_u64(42);
    //     // Generate some random data
    //     let data = generate_structured_sparse_data(90, 10, &mut rng);

    //     // Run t-SNE
    //     let low_dim_data = tsne(&data, 2, 1.0, 100);

    //     // Print the low-dimensional representation
    //     println!("{:?}", low_dim_data);

    //     plot_tsne_result(&low_dim_data);

    //     assert!(false);
    // }

    #[test]
    fn test_tsne_simple2() {
        let mut rng = StdRng::seed_from_u64(42);
        // Generate some random data
        let data = generate_structured_sparse_data(150, 30, &mut rng);

        // Run t-SNE
        let low_dim_data = tsne(&data, 2, 10.0, 200);

        // Print the low-dimensional representation
        println!("{:?}", low_dim_data);

        plot_tsne_result(&low_dim_data);

        assert!(false);
    }

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
