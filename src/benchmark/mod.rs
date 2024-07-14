use std::time::{Duration, Instant};

use serde::Serialize;

use crate::{data::SparseVector, index::SparseIndex};

pub mod logger;
pub mod measure_macro;
pub mod metrics;

/// Configuration for running benchmark on different dataset configurations.
///
/// This structure defines the range and step values for varying parameters
/// in benchmark tests, specifically focusing on the dimensions of the data
/// and the number of images used in each test set. It also specifies the
/// range of values each data point can hold.
/// Dimensions in a dataset refer to the number of attributes or features
/// each data point (e.g., image) has.
/// For example, a 3-dimensional vector might represent an RGB color value.
pub struct BenchmarkConfig {
    /// The starting number of dimensions for the test data vectors.
    pub start_dimensions: usize,
    /// The maximum number of dimensions to be tested.
    pub end_dimensions: usize,
    /// The increment in dimensions for each subsequent test after the
    /// initial start_dimensions.
    pub step_dimensions: usize,
    /// The starting number of data points to use in the dataset for the
    /// benchmarks.
    pub start_num_images: usize,
    /// The maximum number of data points to be tested.
    pub end_num_images: usize,
    /// The increment in the number of data points from one dataset to the
    /// next. This helps in assessing scalability and performance as the
    /// amount of data increases.
    pub step_num_images: usize,
    /// A tuple representing the inclusive minimum and maximum values that
    /// any single element in the data vectors can take. This is crucial for
    /// generating test data with realistic variability.
    pub value_range: (f32, f32),
}

impl BenchmarkConfig {
    pub fn new(
        dimensions_range: (usize, usize, usize),
        num_images_range: (usize, usize, usize),
        value_range: (f32, f32),
    ) -> Self {
        BenchmarkConfig {
            start_dimensions: dimensions_range.0,
            end_dimensions: dimensions_range.1,
            step_dimensions: dimensions_range.2,
            start_num_images: num_images_range.0,
            end_num_images: num_images_range.1,
            step_num_images: num_images_range.2,
            value_range,
        }
    }

    pub fn dataset_configurations(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (self.start_dimensions..=self.end_dimensions)
            .step_by(self.step_dimensions)
            .flat_map(|dimensions| {
                (self.start_num_images..=self.end_num_images)
                    .step_by(self.step_num_images)
                    .map(move |num_images| (dimensions, num_images))
            })
    }
}

#[derive(Serialize, Clone, Copy)]
pub struct BenchmarkResult {
    pub total_execution_time: Duration,
    pub index_execution_time: Duration,
    pub query_execution_time: Duration,
    pub queries_per_second: f32,
    pub dataset_size: usize,
    pub dataset_dimensionality: usize,
    pub scalability_factor: Option<f32>, // Optional because the first benchmark doesn't have a previous result to compare to.
}

pub struct Benchmark {
    index_type: Box<dyn SparseIndex>,
    query_vector: SparseVector,
    previous_benchmark_result: Option<BenchmarkResult>,
}

impl Benchmark {
    pub fn new(
        index_type: Box<dyn SparseIndex>,
        query_vector: SparseVector,
        previous_benchmark_result: Option<BenchmarkResult>,
    ) -> Self {
        Benchmark {
            index_type,
            query_vector,
            previous_benchmark_result,
        }
    }

    pub fn run(&mut self, dataset_size: usize, dimensions: usize, k: usize) -> BenchmarkResult {
        let start_time = Instant::now();

        // Builds the index.
        self.index_type.build();
        let index_execution_time = start_time.elapsed();

        // Perform the query.
        let _query_results = self.index_type.search(&self.query_vector, k);
        let query_execution_time = start_time.elapsed() - index_execution_time;

        let total_execution_time = start_time.elapsed();

        let queries_per_second = metrics::calculate_queries_per_second(query_execution_time);

        let scalability_factor = self
            .previous_benchmark_result
            .as_ref()
            .map(|previous_result| {
                metrics::calculate_scalability_factor(
                    (queries_per_second, dataset_size, dimensions),
                    previous_result,
                )
            });

        BenchmarkResult {
            total_execution_time,
            index_execution_time,
            query_execution_time,
            queries_per_second,
            scalability_factor,
            dataset_size,
            dataset_dimensionality: dimensions,
        }
    }
}
