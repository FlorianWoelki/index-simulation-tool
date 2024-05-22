use benchmark::{
    logger::BenchmarkLogger, metrics::DEFAULT_SCALABILITY_FACTOR, Benchmark, BenchmarkConfig,
    BenchmarkResult,
};
use data::{generator_dense::DenseDataGenerator, HighDimVector};
use index::{hnsw::HNSWIndex, naive::NaiveIndex, ssg::SSGIndex, DistanceMetric, Index};

use clap::Parser;
use sysinfo::Pid;

mod benchmark;
mod data;
mod index;
mod kmeans;

#[derive(Parser)]
struct Args {
    #[clap(long, short, action)]
    dimensions: Option<usize>,
    #[clap(long, short, action)]
    num_images: Option<usize>,
}

#[tokio::main]
async fn main() {
    let n = 1000;
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        samples.push(HighDimVector::new(i, vec![128.0 + i as f64; 3]));
    }
    let mut index = SSGIndex::new(DistanceMetric::Euclidean);
    for sample in samples {
        index.add_vector(sample);
    }
    index.build();
    let query = HighDimVector::new(999999999, vec![208.0; 3]);
    let result = index.search(&query, 5);
    println!("{:?}", result);

    /*let args = Args::parse();
    let dimensions = args.dimensions.unwrap_or(100);
    let num_images = args.num_images.unwrap_or(1000);

    //run_benchmark::<NaiveIndex>(dimensions, num_images).await;
    run_benchmark::<SSGIndex>(dimensions, num_images).await;*/
}

async fn run_benchmark<I: Index + 'static>(dimensions: usize, num_images: usize) {
    let benchmark_config = BenchmarkConfig::new(
        (dimensions, 100, dimensions),
        (num_images, 1_000_000, num_images),
        (0.0, 255.0),
    );
    let mut logger = BenchmarkLogger::new();

    let mut previous_benchmark_result = None;
    for config in benchmark_config.dataset_configurations() {
        measure_resources!({
            let result = perform_single_benchmark::<I>(
                &benchmark_config,
                &mut logger,
                previous_benchmark_result,
                config,
            )
            .await;
            previous_benchmark_result = Some(result);
        });
    }

    // TODO: Change file name to be more generic with a date.
    if let Err(e) = logger.write_to_csv("benchmark_results.csv") {
        eprintln!("Failed to write benchmark results to CSV: {}", e);
    }
}

async fn perform_single_benchmark<I: Index + 'static>(
    config: &BenchmarkConfig,
    logger: &mut BenchmarkLogger,
    previous_benchmark_result: Option<BenchmarkResult>,
    (dimensions, num_images): (usize, usize),
) -> BenchmarkResult {
    println!("------------------------------------");
    println!(
        "Benchmarking with dimensions: {}, num_images: {}",
        dimensions, num_images
    );

    let generated_data = generate_data(config, dimensions, num_images).await;
    let index = add_vectors_to_index(&generated_data);
    let query = create_query_vector(config, dimensions);

    let mut benchmark = Benchmark::new(index, query, previous_benchmark_result);
    let k = 5;
    let result = benchmark.run(num_images, dimensions, k);

    logger.add_record(&result);

    print_benchmark_results(&result);
    println!("------------------------------------");
    result
}

async fn generate_data(
    config: &BenchmarkConfig,
    dimensions: usize,
    num_images: usize,
) -> Vec<Vec<f64>> {
    let mut data_generator = DenseDataGenerator::new(dimensions, num_images, config.value_range);
    data_generator.generate().await
}

fn add_vectors_to_index(data: &[Vec<f64>]) -> Box<dyn Index> {
    let mut index = Box::new(SSGIndex::new(DistanceMetric::Euclidean));
    for (i, d) in data.iter().enumerate() {
        index.add_vector(HighDimVector::new(i, d.clone()));
    }
    index
}

fn create_query_vector(_config: &BenchmarkConfig, dimensions: usize) -> HighDimVector {
    HighDimVector::new(999999999, vec![128.0; dimensions])
}

fn print_benchmark_results(result: &BenchmarkResult) {
    println!("Total Execution time: {:?}", result.total_execution_time);
    println!("Index Execution time: {:?}", result.index_execution_time);
    println!("Query Execution time: {:?}", result.query_execution_time);
    println!("Queries per Second (QPS): {:?}", result.queries_per_second);
    println!(
        "Scalability Factor: {:?}",
        result
            .scalability_factor
            .unwrap_or(DEFAULT_SCALABILITY_FACTOR)
    );
    println!("Dataset Size: {:?}", result.dataset_size);
    println!(
        "Dataset Dimensionality: {:?}",
        result.dataset_dimensionality
    );
}
