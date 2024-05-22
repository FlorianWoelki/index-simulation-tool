use rand::{
    distributions::{Distribution, Uniform},
    thread_rng,
};

pub struct DenseDataGenerator {
    dim: usize,
    count: usize,
    range: (f32, f32),
    system: sysinfo::System,
}

impl DenseDataGenerator {
    pub fn new(dim: usize, count: usize, range: (f32, f32)) -> Self {
        DenseDataGenerator {
            dim,
            count,
            range,
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

            let handle = tokio::task::spawn_blocking(move || {
                let uniform_dist = Uniform::from(range.0..range.1);
                let mut data_chunk = Vec::with_capacity(dim);

                for _ in 0..per_chunk {
                    let mut inner_vec = Vec::with_capacity(dim);
                    for _ in 0..dim {
                        inner_vec.push(uniform_dist.sample(&mut thread_rng()));
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
    async fn test_data_generation() {
        let mut generator = DenseDataGenerator::new(5, 10, (0.0, 1.0));
        let data = generator.generate().await;
        assert_eq!(data.len(), 10);
        assert_eq!(data[0].len(), 5);
        for vector in data {
            for &value in &vector {
                assert!(value >= 0.0 && value <= 1.0);
            }
        }
    }
}
