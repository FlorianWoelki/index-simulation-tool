use ordered_float::OrderedFloat;
use rand::{
    distributions::{Bernoulli, Distribution, Uniform},
    thread_rng,
};

use super::SparseVector;

pub struct SparseDataGenerator {
    dim: usize,
    count: usize,
    range: (f32, f32),
    sparsity: f32,
    system: sysinfo::System,
}

impl SparseDataGenerator {
    /// Creates a new sparse data generator with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimensionality of each vector.
    /// * `count` - The number of vectors to generate.
    /// * `range` - The range of values for non-zero elements.
    /// * `sparsity` - The probability of an element being zero (sparsity factor). The higher the value, the sparser the data.
    ///
    /// # Returns
    ///
    /// A new instance of `SparseDataGenerator`.
    pub fn new(dim: usize, count: usize, range: (f32, f32), sparsity: f32) -> Self {
        SparseDataGenerator {
            dim,
            count,
            range,
            sparsity,
            system: sysinfo::System::new(),
        }
    }

    pub async fn generate(&mut self) -> Vec<SparseVector> {
        self.system.refresh_all();

        let mut handles = vec![];

        let chunks = self.system.cpus().len(); // Number of parallel tasks.
        let per_chunk = self.count / chunks;

        for _ in 0..chunks {
            let dim = self.dim;
            let range = self.range;
            let sparsity = self.sparsity;

            let handle = tokio::task::spawn_blocking(move || {
                let uniform_dist = Uniform::from(range.0..range.1);
                let bernoulli_dist = Bernoulli::new(sparsity as f64).unwrap();
                let mut data_chunk = Vec::with_capacity(dim);

                for _ in 0..per_chunk {
                    let mut indices = Vec::new();
                    let mut values = Vec::new();
                    for i in 0..dim {
                        if !bernoulli_dist.sample(&mut thread_rng()) {
                            indices.push(i);
                            values.push(OrderedFloat(uniform_dist.sample(&mut thread_rng())));
                        }
                    }

                    data_chunk.push(SparseVector { indices, values });
                }

                data_chunk
            });

            handles.push(handle);
        }

        let mut results = Vec::with_capacity(self.count);
        for handle in handles {
            let mut result = handle.await.unwrap();
            results.append(&mut result);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sparse_data_generation() {
        let count = 10;
        let dim = 100;
        let range = (0.0, 1.0);
        let sparsity = 0.5;
        let mut generator = SparseDataGenerator::new(dim, count, range, sparsity);
        let vectors = generator.generate().await;

        assert_eq!(vectors.len(), count);

        for vector in vectors {
            assert!(vector.indices.len() <= dim);
            assert_eq!(vector.indices.len(), vector.values.len());

            for value in vector.values {
                assert!(value.into_inner() >= range.0);
                assert!(value.into_inner() < range.1);
            }
        }
    }

    #[tokio::test]
    async fn test_sparsity() {
        let count = 10;
        let dim = 100;
        let range = (0.0, 1.0);
        let sparsity = 0.8;
        let mut generator = SparseDataGenerator::new(dim, count, range, sparsity);
        let vectors = generator.generate().await;

        assert_eq!(vectors.len(), count);

        let mut total_non_zero = 0;
        let mut total_elements = 0;

        for vector in vectors {
            let non_zero_count = vector.indices.len();

            total_non_zero += non_zero_count;
            total_elements += dim;
        }

        let actual_sparsity = 1.0 - (total_non_zero as f32 / total_elements as f32);
        assert!((actual_sparsity - sparsity).abs() < 0.05);
    }
}
