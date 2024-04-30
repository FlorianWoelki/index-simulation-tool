use rand::{
    distributions::{Distribution, Uniform},
    rngs::ThreadRng,
    thread_rng,
};

pub struct DataGenerator {
    dim: usize,
    count: usize,
    range: (f64, f64),
    rng: ThreadRng,
}

impl DataGenerator {
    pub fn new(dim: usize, count: usize, range: (f64, f64)) -> Self {
        DataGenerator {
            dim,
            count,
            range,
            rng: thread_rng(),
        }
    }

    pub fn generate(&mut self) -> Vec<Vec<f64>> {
        let uniform_dist = Uniform::from(self.range.0..self.range.1);
        (0..self.count)
            .map(|_| {
                (0..self.dim)
                    .map(|_| uniform_dist.sample(&mut self.rng))
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_generation() {
        let mut generator = DataGenerator::new(5, 10, (0.0, 1.0));
        let data = generator.generate();
        assert_eq!(data.len(), 10);
        assert_eq!(data[0].len(), 5);
        for vector in data {
            for &value in &vector {
                assert!(value >= 0.0 && value <= 1.0);
            }
        }
    }
}
