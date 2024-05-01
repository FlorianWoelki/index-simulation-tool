use benchmark::{Benchmark, BenchmarkResult};
use data::{generator::DataGenerator, HighDimVector};
use index::{naive::NaiveIndex, DistanceMetric, Index};
use query::{naive::NaiveQuery, Query};

use crate::query::hnsw::HNSWQuery;

mod benchmark;
mod data;
mod index;
mod query;

fn main() {
    run_benchmark::<HNSWQuery>();
    run_benchmark::<NaiveQuery>();
}

/// Configuration for running benchmark on different dataset configurations.
///
/// This structure defines the range and step values for varying parameters
/// in benchmark tests, specifically focusing on the dimensions of the data
/// and the number of images used in each test set. It also specifies the
/// range of values each data point can hold.
/// Dimensions in a dataset refer to the number of attributes or features
/// each data point (e.g., image) has.
/// For example, a 3-dimensional vector might represent an RGB color value.
struct BenchmarkConfig {
    /// The starting number of dimensions for the test data vectors.
    start_dimensions: usize,
    /// The maximum number of dimensions to be tested.
    end_dimensions: usize,
    /// The increment in dimensions for each subsequent test after the
    /// initial start_dimensions.
    step_dimensions: usize,
    /// The starting number of data points to use in the dataset for the
    /// benchmarks.
    start_num_images: usize,
    /// The maximum number of data points to be tested.
    end_num_images: usize,
    /// The increment in the number of data points from one dataset to the
    /// next. This helps in assessing scalability and performance as the
    /// amount of data increases.
    step_num_images: usize,
    /// A tuple representing the inclusive minimum and maximum values that
    /// any single element in the data vectors can take. This is crucial for
    /// generating test data with realistic variability.
    value_range: (f64, f64),
}

impl BenchmarkConfig {
    fn new(
        dimensions_range: (usize, usize, usize),
        num_images_range: (usize, usize, usize),
        value_range: (f64, f64),
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

    fn dataset_configurations<'a>(&'a self) -> impl Iterator<Item = (usize, usize)> + 'a {
        (self.start_dimensions..=self.end_dimensions)
            .step_by(self.step_dimensions)
            .flat_map(|dimensions| {
                (self.start_num_images..=self.end_num_images)
                    .step_by(self.step_num_images)
                    .map(move |num_images| (dimensions, num_images))
            })
    }
}

fn run_benchmark<Q: Query + 'static>() {
    let benchmark_config =
        BenchmarkConfig::new((100, 100, 100), (100_000, 1_000_000, 100_000), (0.0, 255.0));

    let mut previous_benchmark_result = None;
    for config in benchmark_config.dataset_configurations() {
        let result =
            perform_single_benchmark::<Q>(&benchmark_config, previous_benchmark_result, config);
        previous_benchmark_result = Some(result);
    }
}

fn perform_single_benchmark<Q: Query + 'static>(
    config: &BenchmarkConfig,
    previous_benchmark_result: Option<BenchmarkResult>,
    (dimensions, num_images): (usize, usize),
) -> BenchmarkResult {
    println!(
        "Benchmarking with dimensions: {}, num_images: {}",
        dimensions, num_images
    );

    let generated_data = generate_data(config, dimensions, num_images);
    let index = add_vectors_to_index(&generated_data);
    let query = create_query::<Q>(config, dimensions);

    let mut benchmark = Benchmark::new(index, query, previous_benchmark_result);
    let result = benchmark.run(num_images, dimensions);

    print_benchmark_results(&result);
    result
}

fn generate_data(config: &BenchmarkConfig, dimensions: usize, num_images: usize) -> Vec<Vec<f64>> {
    println!("generating data...");
    let mut data_generator = DataGenerator::new(dimensions, num_images, config.value_range);
    let generated_data = data_generator.generate();
    println!("...done");
    generated_data
}

fn add_vectors_to_index(data: &Vec<Vec<f64>>) -> Box<dyn Index> {
    println!("adding vectors to the index data structure...");
    let mut index = Box::new(NaiveIndex::new(DistanceMetric::Euclidean));
    for d in data {
        index.add_vector(HighDimVector::new(d.clone()));
    }
    println!("...done");
    index
}

fn create_query<Q: Query>(config: &BenchmarkConfig, dimensions: usize) -> Box<Q> {
    let k = 2;
    let query_vector = HighDimVector::new(vec![128.0; dimensions]);
    Box::new(Q::new(query_vector, k))
}

fn print_benchmark_results(result: &BenchmarkResult) {
    println!("Total Execution time: {:?}", result.total_execution_time);
    println!("Index Execution time: {:?}", result.index_execution_time);
    println!("Query Execution time: {:?}", result.query_execution_time);
    println!("Queries per Second (QPS): {:?}", result.queries_per_second);
    println!("Scalability Factor: {:?}", result.scalability_factor);
    println!("Dataset Size: {:?}", result.dataset_size);
    println!(
        "Dataset Dimensionality: {:?}",
        result.dataset_dimensionality
    );
    println!("------------------------------------");
}
