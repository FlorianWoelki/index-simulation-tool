use std::{
    collections::BinaryHeap,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    sync::Mutex,
};

use ordered_float::OrderedFloat;
use rand::{
    distributions::{Bernoulli, Distribution, Uniform},
    rngs::StdRng,
    SeedableRng,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::index::DistanceMetric;

use super::vector::SparseVector;

pub struct SparseDataGenerator {
    pub dim: usize,
    pub count: usize,
    range: (f32, f32),
    sparsity: f32,
    metric: DistanceMetric,
    system: sysinfo::System,
    seed: u64,
    query_count: Option<usize>,

    pub vectors: Vec<SparseVector>,
    pub query_vectors: Vec<SparseVector>,
    pub groundtruth: Vec<Vec<usize>>,
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
    /// * `metric` - The DistanceMetric that will be used to fetch the groundtruth vectors.
    /// * `query_count` - Optional number of query vectors to generate. Defaults to 10% of count if None.
    ///
    /// # Returns
    ///
    /// A new instance of `SparseDataGenerator`.
    pub fn new(
        dim: usize,
        count: usize,
        range: (f32, f32),
        sparsity: f32,
        metric: DistanceMetric,
        seed: u64,
        query_count: Option<usize>,
    ) -> Self {
        SparseDataGenerator {
            dim,
            count,
            range,
            sparsity,
            metric,
            system: sysinfo::System::new(),
            seed,
            query_count,
            vectors: vec![],
            query_vectors: vec![],
            groundtruth: vec![],
        }
    }

    /// Generates sparse vectors, query vectors, and ground truth nearest neighbors.
    /// Sets the vectors, query vectors, and groundtruth vectors inside the generator struct.
    pub async fn generate(&mut self) {
        self.system.refresh_all();

        let mut handles = vec![];

        let chunks = self.system.cpus().len(); // Number of parallel tasks.
        let per_chunk = self.count / chunks;

        for i in 0..chunks {
            let dim = self.dim;
            let range = self.range;
            let sparsity = self.sparsity;
            let seed = self.seed.wrapping_add(i as u64);

            let handle = tokio::task::spawn_blocking(move || {
                let mut rng = StdRng::seed_from_u64(seed);
                let uniform_dist = Uniform::from(range.0..range.1);
                let bernoulli_dist = Bernoulli::new(sparsity as f64).unwrap();
                let mut data_chunk = Vec::with_capacity(dim);

                for _ in 0..per_chunk {
                    let mut indices = Vec::new();
                    let mut values = Vec::new();
                    for i in 0..dim {
                        if !bernoulli_dist.sample(&mut rng) {
                            indices.push(i);
                            values.push(OrderedFloat(uniform_dist.sample(&mut rng)));
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

        let query_count = self.query_count.unwrap_or(self.count / 10);
        let query_vectors = self.generate_vectors(self.dim, query_count, self.range, self.sparsity);

        let mut groundtruth_vectors = Vec::with_capacity(query_vectors.len());
        for query_vector in &query_vectors {
            groundtruth_vectors.push(self.find_nearest_neighbors(&results, query_vector, 10));
        }

        self.vectors = results;
        self.query_vectors = query_vectors;
        self.groundtruth = groundtruth_vectors;
    }

    fn generate_vectors(
        &self,
        dim: usize,
        count: usize,
        range: (f32, f32),
        sparsity: f32,
    ) -> Vec<SparseVector> {
        let uniform_dist = Uniform::from(range.0..range.1);
        let bernoulli_dist = Bernoulli::new(sparsity as f64).unwrap();

        (0..count)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(1000 + i as u64));
                let mut indices = Vec::new();
                let mut values = Vec::new();
                for i in 0..dim {
                    if !bernoulli_dist.sample(&mut rng) {
                        indices.push(i);
                        values.push(OrderedFloat(uniform_dist.sample(&mut rng)));
                    }
                }

                SparseVector { indices, values }
            })
            .collect()
    }

    fn find_nearest_neighbors(
        &self,
        data: &[SparseVector],
        query: &SparseVector,
        k: usize,
    ) -> Vec<usize> {
        let heap = Mutex::new(BinaryHeap::new());

        data.par_iter().enumerate().for_each(|(i, vector)| {
            let distance = query.distance(vector, &self.metric);
            let mut heap = heap.lock().unwrap();
            if heap.len() < k {
                heap.push((OrderedFloat(distance), i));
            } else if let Some((OrderedFloat(max_distance), _)) = heap.peek() {
                if distance < *max_distance {
                    heap.pop();
                    heap.push((OrderedFloat(distance), i));
                }
            }
        });

        heap.into_inner()
            .unwrap()
            .into_sorted_vec()
            .into_iter()
            .map(|(_, v)| v)
            .collect::<Vec<_>>()
    }

    pub fn save_data(&self, filename: &str) -> std::io::Result<()> {
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);
        let serialized =
            bincode::serialize(&(&self.vectors, &self.query_vectors, &self.groundtruth))
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        writer.write_all(&serialized)?;
        Ok(())
    }

    pub fn load_data(
        &self,
        filename: &str,
    ) -> std::io::Result<(Vec<SparseVector>, Vec<SparseVector>, Vec<Vec<usize>>)> {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;
        let deserialized = bincode::deserialize(&buffer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(deserialized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sparse_data_generation() {
        let seed = 42;
        let count = 10;
        let dim = 100;
        let range = (0.0, 1.0);
        let sparsity = 0.5;
        let mut generator = SparseDataGenerator::new(
            dim,
            count,
            range,
            sparsity,
            DistanceMetric::Euclidean,
            seed,
            None,
        );
        generator.generate().await;

        assert_eq!(generator.vectors.len(), count);
        assert_eq!(generator.query_vectors.len(), count / 10);
        assert_eq!(generator.groundtruth.len(), generator.query_vectors.len());

        for vector in &generator.vectors {
            assert!(vector.indices.len() <= dim);
            assert_eq!(vector.indices.len(), vector.values.len());

            for value in &vector.values {
                assert!(value.into_inner() >= range.0);
                assert!(value.into_inner() < range.1);
            }
        }

        for vector in generator.query_vectors {
            assert!(vector.indices.len() <= dim);
            assert_eq!(vector.indices.len(), vector.values.len());

            for value in vector.values {
                assert!(value.into_inner() >= range.0);
                assert!(value.into_inner() < range.1);
            }
        }

        for groundtruth_set in generator.groundtruth {
            assert!(groundtruth_set.len() <= 10);
            for i in groundtruth_set {
                assert!(generator.vectors[i].indices.len() <= dim);
                assert_eq!(
                    generator.vectors[i].indices.len(),
                    generator.vectors[i].values.len()
                );

                for value in &generator.vectors[i].values {
                    assert!(value.into_inner() >= range.0);
                    assert!(value.into_inner() < range.1);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_custom_query_count() {
        let seed = 42;
        let count = 100;
        let dim = 100;
        let range = (0.0, 1.0);
        let sparsity = 0.5;
        let custom_query_count = 5;

        let mut generator = SparseDataGenerator::new(
            dim,
            count,
            range,
            sparsity,
            DistanceMetric::Euclidean,
            seed,
            Some(custom_query_count),
        );
        generator.generate().await;

        assert_eq!(generator.vectors.len(), count);
        assert_eq!(generator.query_vectors.len(), custom_query_count);
        assert_eq!(generator.groundtruth.len(), generator.query_vectors.len());
    }

    #[tokio::test]
    async fn test_sparsity() {
        let seed = 42;
        let count = 10;
        let dim = 100;
        let range = (0.0, 1.0);
        let sparsity = 0.8;
        let mut generator = SparseDataGenerator::new(
            dim,
            count,
            range,
            sparsity,
            DistanceMetric::Euclidean,
            seed,
            None,
        );
        generator.generate().await;

        assert_eq!(generator.vectors.len(), count);

        let mut total_non_zero = 0;
        let mut total_elements = 0;

        for vector in generator.vectors {
            let non_zero_count = vector.indices.len();

            total_non_zero += non_zero_count;
            total_elements += dim;
        }

        let actual_sparsity = 1.0 - (total_non_zero as f32 / total_elements as f32);
        assert!((actual_sparsity - sparsity).abs() < 0.05);
    }

    #[tokio::test]
    async fn test_groundtruth_generation() {
        let seed = 42;
        let count = 100;
        let dim = 50;
        let range = (0.0, 1.0);
        let sparsity = 0.5;
        let k = 10;
        let mut generator = SparseDataGenerator::new(
            dim,
            count,
            range,
            sparsity,
            DistanceMetric::Euclidean,
            seed,
            None,
        );
        generator.generate().await;

        assert_eq!(generator.query_vectors.len(), count / 10);
        assert_eq!(generator.groundtruth.len(), generator.query_vectors.len());

        for (query, groundtruth_set) in generator
            .query_vectors
            .iter()
            .zip(generator.groundtruth.iter())
        {
            let calculated_groundtruth =
                generator.find_nearest_neighbors(&generator.vectors, query, k);

            assert_eq!(groundtruth_set.len(), k);
            assert_eq!(groundtruth_set, &calculated_groundtruth);
        }
    }

    #[tokio::test]
    async fn test_manual_groundtruth() {
        let seed = 42;
        let vectors = vec![
            SparseVector {
                indices: vec![0, 2, 4],
                values: vec![OrderedFloat(0.1), OrderedFloat(0.2), OrderedFloat(0.3)],
            },
            SparseVector {
                indices: vec![1, 2, 3],
                values: vec![OrderedFloat(0.1), OrderedFloat(0.4), OrderedFloat(0.5)],
            },
            SparseVector {
                indices: vec![0, 3, 4],
                values: vec![OrderedFloat(0.6), OrderedFloat(0.7), OrderedFloat(0.8)],
            },
        ];

        let query_vector = SparseVector {
            indices: vec![0, 2, 4],
            values: vec![OrderedFloat(0.1), OrderedFloat(0.2), OrderedFloat(0.3)],
        };

        let expected_groundtruth = vec![0, 1, 2];

        let generator =
            SparseDataGenerator::new(0, 0, (0.0, 1.0), 0.0, DistanceMetric::Euclidean, seed, None);
        let groundtruth_vectors = generator.find_nearest_neighbors(&vectors, &query_vector, 3);

        assert_eq!(groundtruth_vectors, expected_groundtruth);
    }
}
