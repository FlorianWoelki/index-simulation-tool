use std::collections::HashMap;

use data::{generator_dense::DenseDataGenerator, SparseVector};
use index::{linscan::LinScanIndex, minhash::MinHashIndex, DistanceMetric, SparseIndex};

use clap::Parser;
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use sysinfo::Pid;

mod benchmark;
mod data;
mod data_structures;
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
    let (groundtruth, vectors, query_vectors) = data::ms_marco::load_msmarco_dataset().unwrap();

    // let mut index = LinScanIndex::new(DistanceMetric::Euclidean);
    let seed = thread_rng().gen_range(0..1000);
    let mut index = MinHashIndex::new_with_rng(200, 20, 100, seed);
    let mut query_sparse_vectors = vec![];
    let mut vectors_sparse_vectors = vec![];
    for query_vector in query_vectors.iter() {
        let sparse_vector = SparseVector {
            indices: query_vector.0.clone(),
            values: query_vector
                .1
                .iter()
                .map(|&v| ordered_float::OrderedFloat(v))
                .collect(),
        };
        query_sparse_vectors.push(sparse_vector);
    }

    for vector in vectors.iter().take(15000) {
        let sparse_vector = SparseVector {
            indices: vector.0.clone(),
            values: vector
                .1
                .iter()
                .map(|&v| ordered_float::OrderedFloat(v))
                .collect(),
        };
        vectors_sparse_vectors.push(sparse_vector);
    }

    println!("Start adding vectors...");
    for (i, vector) in vectors_sparse_vectors.iter().enumerate() {
        index.add_vector(vector);
        println!("Vector {}", i)
    }
    println!("Done adding vectors...");

    let r = index.search(&query_sparse_vectors[0], 10);
    println!("{:?}", r);
    let r = index.search(
        &vectors_sparse_vectors[thread_rng().gen_range(0..15000)],
        10,
    );
    println!("{:?}", r);

    println!("gt: {:?}", groundtruth.0[0]);

    for (i, res) in r.iter().enumerate() {
        /*let gt_index = groundtruth.0[0][i];
        if res.index != gt_index as usize {
            println!("{}: expected: {}, got: {}", i, gt_index, res.index);
            }*/

        /*if (res.score - groundtruth.1[0][i]).abs() > 0.01 {
        println!(
            "{}: expected: {}, got: {}",
            i, groundtruth.1[0][i], res.score
        );
        }*/
    }

    /*let start = std::time::Instant::now();
    let mut hit = 0;
    for (i, query_vector) in query_sparse_vectors.iter().enumerate() {
        let result = index.search(query_vector, 10);

        let expected_indices = &groundtruth.0[i];
        let expected_scores = &groundtruth.1[i];

        for (j, res) in result.iter().enumerate() {
            let expected_score = expected_scores[j];
            if (res.score - expected_score).abs() > 0.01 {
                println!(
                    "{}({}). expected: {}, got: {}",
                    i, j, expected_score, res.score
                );
            }

            if res.id != expected_indices[j] as usize {
                println!(
                    "{}({}). expected: {}, got: {}",
                    i, j, expected_indices[j], res.id
                );
            } else {
                hit += 1;
            }
        }
    }
    let elapsed = start.elapsed();

    println!(
        "recall: {}",
        hit as f32 / (query_sparse_vectors.len() * 10) as f32
    );
    println!("elapsed: {:?}", elapsed);*/

    /*let (query_vectors, groundtruth, vectors) = data::sift::get_data().unwrap();

    let mut index = LSHIndex::new(DistanceMetric::Euclidean);
    for (i, vector) in vectors.iter().enumerate() {
        index.add_vector(HighDimVector::new(i, vector.to_vec()));
    }

    index.build();*/

    /*let query_id = 0;
    let query_vector = HighDimVector::new(query_id, query_vectors[query_id].to_vec());
    let baseline = &groundtruth[query_id];
    let k = 5;
    let result = index.search(&query_vector, k);
    for res in result {
        println!("{:?}", res.id);
    }
    println!("{:?}", &baseline[0..k]);*/

    /*let round = 100;
    let mut hit = 0;
    let mut rng = thread_rng();

    for _ in 0..round {
        let query_i = rng.gen_range(0..query_vectors.len());
        let query = HighDimVector::new(query_i, query_vectors[query_i].to_vec());

        let result = index.search(&query, 5);
        let top5_groundtruth = &groundtruth[query_i][0..5];
        for res in result {
            let id = res.id as i32;
            if top5_groundtruth.contains(&id) {
                hit += 1;
            }
        }
    }

    println!("recall: {}", hit as f32 / (round * 5) as f32);*/

    /*let n = 1000;
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        samples.push(HighDimVector::new(i, vec![128.0 + i as f32; 128]));
    }
    let mut index = LSHIndex::new(DistanceMetric::Euclidean);
    for sample in samples {
        index.add_vector(sample);
    }
    index.build();
    let query = HighDimVector::new(999999999, vec![208.0; 3]);
    let result = index.search(&query, 5);

    for (i, vec) in result.iter().enumerate() {
        println!("{} {:?}", i, vec.id);
        }*/

    /*let args = Args::parse();
    let dimensions = args.dimensions.unwrap_or(100);
    let num_images = args.num_images.unwrap_or(1000);

    //run_benchmark::<NaiveIndex>(dimensions, num_images).await;
    run_benchmark::<SSGIndex>(dimensions, num_images).await;*/
}

/*async fn run_benchmark<I: Index + 'static>(dimensions: usize, num_images: usize) {
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
}*/

/*async fn perform_single_benchmark<I: Index + 'static>(
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
) -> Vec<Vec<f32>> {
    let mut data_generator = DenseDataGenerator::new(dimensions, num_images, config.value_range);
    data_generator.generate().await
    }*/

/*fn add_vectors_to_index(data: &[Vec<f32>]) -> Box<dyn Index> {
let mut index = Box::new(SSGIndex::new(DistanceMetric::Euclidean));
for (i, d) in data.iter().enumerate() {
    index.add_vector(HighDimVector::new(i, d.clone()));
}
index
}*/

// fn create_query_vector(_config: &BenchmarkConfig, dimensions: usize) -> HighDimVector {
//     HighDimVector::new(999999999, vec![128.0; dimensions])
// }

// fn print_benchmark_results(result: &BenchmarkResult) {
//     println!("Total Execution time: {:?}", result.total_execution_time);
//     println!("Index Execution time: {:?}", result.index_execution_time);
//     println!("Query Execution time: {:?}", result.query_execution_time);
//     println!("Queries per Second (QPS): {:?}", result.queries_per_second);
//     println!(
//         "Scalability Factor: {:?}",
//         result
//             .scalability_factor
//             .unwrap_or(DEFAULT_SCALABILITY_FACTOR)
//     );
//     println!("Dataset Size: {:?}", result.dataset_size);
//     println!(
//         "Dataset Dimensionality: {:?}",
//         result.dataset_dimensionality
//     );
// }
