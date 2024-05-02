use benchmark::{Benchmark, BenchmarkConfig, BenchmarkResult};
use data::{generator::DataGenerator, HighDimVector};
use index::{naive::NaiveIndex, DistanceMetric, Index};
use query::{naive::NaiveQuery, Query};

use clap::Parser;

use crate::query::hnsw::HNSWQuery;

mod benchmark;
mod data;
mod index;
mod query;

#[derive(Parser)]
struct Args {
    #[clap(long, short, action)]
    dimensions: Option<usize>,
    #[clap(long, short, action)]
    num_images: Option<usize>,
}

fn main() {
    let args = Args::parse();
    let dimensions = args.dimensions.unwrap_or(100);
    let num_images = args.num_images.unwrap_or(100_000);

    run_benchmark::<HNSWQuery>(dimensions, num_images);
    run_benchmark::<NaiveQuery>(dimensions, num_images);
}

fn run_benchmark<Q: Query + 'static>(dimensions: usize, num_images: usize) {
    let benchmark_config = BenchmarkConfig::new(
        (dimensions, 100, dimensions),
        (num_images, 1_000_000, num_images),
        (0.0, 255.0),
    );

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

fn create_query<Q: Query>(_config: &BenchmarkConfig, dimensions: usize) -> Box<Q> {
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
