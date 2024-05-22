use rand::{
    distributions::{Bernoulli, Distribution, Uniform},
    thread_rng,
};

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

    pub async fn generate(&mut self) -> Vec<Vec<f32>> {
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
                    let mut inner_vec = Vec::with_capacity(dim);
                    for _ in 0..dim {
                        if bernoulli_dist.sample(&mut thread_rng()) {
                            inner_vec.push(0.0);
                        } else {
                            inner_vec.push(uniform_dist.sample(&mut thread_rng()));
                        }
                    }

                    data_chunk.push(inner_vec);
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
        let mut generator = SparseDataGenerator::new(5, 10, (0.0, 1.0), 0.5);
        let data = generator.generate().await;

        assert_eq!(data.len(), 10);
        assert_eq!(data[0].len(), 5);

        for vector in data {
            for &value in &vector {
                if value != 0.0 {
                    assert!(value >= 0.0 && value <= 1.0);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_sparse_data_generation_more_sparse() {
        let mut generator = SparseDataGenerator::new(5, 10, (0.0, 1.0), 0.9);
        let data = generator.generate().await;

        assert_eq!(data.len(), 10);
        assert_eq!(data[0].len(), 5);

        for vector in &data {
            for &value in vector {
                if value != 0.0 {
                    assert!(value >= 0.0 && value <= 1.0);
                }
            }
        }

        let mut non_zero_count = 0;
        let mut zero_count = 0;
        for vector in data {
            for &value in &vector {
                if value != 0.0 {
                    non_zero_count += 1;
                } else {
                    zero_count += 1;
                }
            }
        }

        assert!(non_zero_count < zero_count);
    }
}
