use std::{
    fs::{self, File, OpenOptions},
    time::{Duration, Instant},
};

use benchmark::{
    logger::BenchmarkLogger,
    macros::{measure_system::ResourceReport, measure_time},
    BenchmarkConfig, GenericBenchmarkResult, IndexBenchmarkResult,
};
use chrono::Local;
use data::{
    generator_sparse::SparseDataGenerator,
    plot::{plot_nearest_neighbor_distances, plot_sparsity_distribution},
    SparseVector,
};

use rand::{thread_rng, Rng};
use sysinfo::{Pid, System};

use clap::Parser;
use index::{annoy::AnnoyIndex, DistanceMetric, IndexType, SparseIndex};
use ordered_float::OrderedFloat;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

mod benchmark;
mod data;
mod data_structures;
mod index;
mod kmeans;
mod test_utils;

#[derive(Parser)]
struct Args {
    #[clap(long, short, action)]
    dimensions: Option<usize>,
    #[clap(long, short, action)]
    features: Option<usize>,
}

#[allow(dead_code)]
async fn plot_msmarco_dataset() {
    let (groundtruth, vectors, query_vectors) = data::ms_marco::load_msmarco_dataset().unwrap();
    let mut query_sparse_vectors = vec![];
    let mut groundtruth_sparse_vectors = vec![];

    // TODO: Parallelize this in the future.
    for (indices, values) in groundtruth.0.iter().zip(groundtruth.1.iter()) {
        let vector = SparseVector {
            indices: indices.iter().map(|i| *i as usize).collect(),
            values: values.iter().map(|v| OrderedFloat(*v)).collect(),
        };
        groundtruth_sparse_vectors.push(vector);
    }
    for (indices, values) in query_vectors.iter() {
        let sparse_vector = SparseVector {
            indices: indices.clone(),
            values: values
                .iter()
                .map(|&v| ordered_float::OrderedFloat(v))
                .collect(),
        };
        query_sparse_vectors.push(sparse_vector);
    }

    let vectors_sparse_vectors = vectors
        .par_iter()
        .map(|(indices, values)| {
            let sparse_vector = SparseVector {
                indices: indices.clone(),
                values: values
                    .iter()
                    .map(|&v| ordered_float::OrderedFloat(v))
                    .collect(),
            };
            sparse_vector
        })
        .collect::<Vec<SparseVector>>();

    plot_sparsity_distribution(&vectors_sparse_vectors).show();
    plot_nearest_neighbor_distances(
        &query_sparse_vectors,
        &groundtruth_sparse_vectors,
        &DistanceMetric::Cosine,
    )
    .show();
}

#[allow(dead_code)]
async fn plot_artificially_generated_data() {
    // Dimensionality:
    // - Text data: 10,000 - 100,000 features
    // - Image data: 1,000 - 1,000,000 pixels
    // - Sensor data: 50 - 500 features
    // Number of data points:
    // - Small dataset: 1,000 - 10,000 samples
    // - Medium dataset: 100,000 - 1,000,000 samples
    // - Large dataset: 1,000,000+ samples
    // Value range:
    // - Binary data: (0, 1)
    // - Normalized data: (-1, 1) or (0, 1)
    // - Count data: (0, 100) or (0, 1000) depending on the specific application
    // Sparsity:
    // - Text data (e.g., document-term matrices): 0.95 - 0.99
    // - Recommender systems: 0.99 - 0.9999
    // - Biological data (e.g., gene expression): 0.7 - 0.9
    // Distance metric:
    // - Text data: Cosine distance
    // - Binary sparse data: Jaccard distance
    // - High-dimensional sparse data: Manhattan distance
    let amount = 1000;
    let mut generator =
        SparseDataGenerator::new(10000, amount, (0.0, 1.0), 0.95, DistanceMetric::Cosine, 42);
    let (vectors, query_vectors, groundtruth) = generator.generate().await;

    // Get the first element of the groundtruth data
    let groundtruth_flat = groundtruth
        .iter()
        .map(|nn| nn[0].clone())
        .collect::<Vec<SparseVector>>();

    plot_sparsity_distribution(&vectors).show();
    plot_nearest_neighbor_distances(&query_vectors, &groundtruth_flat, &DistanceMetric::Cosine)
        .show();
}

fn set_num_threads(num_threads: Option<usize>) {
    // Initialize global pool with number of threads.
    num_threads.map(|nt| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build_global()
            .unwrap();
    });
}

#[tokio::main]
async fn main() {
    let num_threads = None; // Some(1) for serial
    set_num_threads(num_threads);
    println!(
        "Using {}/{} threads",
        rayon::current_num_threads(),
        rayon::max_num_threads()
    );
    let is_parallel = num_threads == None;
    println!("Executing in serial? {}", !is_parallel);

    let args = Args::parse();
    let dimensions = args.dimensions.unwrap_or(100);
    let amount = args.features.unwrap_or(1000);

    let mut rng = thread_rng();
    let distance_metric = DistanceMetric::Cosine;
    let benchmark_config = BenchmarkConfig::new(
        (dimensions, 100, dimensions),
        (amount, 1000, amount),
        (0.0, 1.0),
        0.90,
        distance_metric,
    );

    let mut index_logger: BenchmarkLogger<IndexBenchmarkResult> = BenchmarkLogger::new();
    let mut build_logger: BenchmarkLogger<GenericBenchmarkResult> = BenchmarkLogger::new();

    let current_date = Local::now().format("%Y-%m-%d").to_string();
    let dir_path = format!("index/{}", &current_date);
    fs::create_dir_all(&dir_path).expect("Failed to create directory");

    // let mut previous_benchmark_result = None;
    for (dimensions, amount) in benchmark_config.dataset_configurations() {
        println!("Generating data...");
        let (vectors, query_vectors, groundtruth) =
            generate_data(&benchmark_config, dimensions, amount).await;
        println!("...finished generating data");

        let total_index_start = Instant::now();
        let mut index = AnnoyIndex::new(20, 20, 40, distance_metric);

        for vector in &vectors {
            index.add_vector_before_build(&vector);
        }

        let (_, build_report) = measure_resources!({
            index.build();
        });

        build_logger.add_record(GenericBenchmarkResult {
            execution_time: build_report.execution_time.as_secs_f32(),
            dataset_dimensionality: dimensions,
            dataset_size: amount,
            consumed_cpu: build_report.final_cpu,
            consumed_memory: build_report.final_memory,
        });
        print_measurement_report(&build_report);

        // Benchmark for measuring searching.
        let (_, search_report) = measure_resources!({
            let result = index.search(&query_vectors[0], 5);

            println!("{:?}", vectors[result[0].index]);
            println!("{:?}", groundtruth[0][0]);
        });

        let total_index_duration = total_index_start.elapsed();

        print_measurement_report(&search_report);

        // Benchmark for measuring adding a vector to the index.
        let mut added_vectors = vec![
            SparseVector {
                indices: vec![],
                values: vec![]
            };
            100
        ];
        let mut total_add_duration = Duration::new(0, 0);
        for vector in &mut added_vectors {
            let random_vector = &vectors[rng.gen_range(0..vectors.len())];

            vector.indices = random_vector.indices.clone();
            vector.values = random_vector.values.clone();

            let add_vector_start = Instant::now();
            index.add_vector(&vector);
            let add_vector_duration = add_vector_start.elapsed();

            total_add_duration += add_vector_duration;
            println!("Add vector duration: {:?}", add_vector_duration);
        }

        let average_add_duration = total_add_duration / added_vectors.len() as u32;
        println!("Average vector adding time: {:?}", average_add_duration);

        // Benchmark for measuring removing a vector from the index.
        let mut total_remove_duration = Duration::new(0, 0);
        for _ in 0..added_vectors.len() {
            let remove_vector_start = Instant::now();
            index.remove_vector(vectors.len() + 1); // Always this because the vector gets removed from the array, therefore reducing the length.
            let remove_vector_duration = remove_vector_start.elapsed();

            total_remove_duration += remove_vector_duration;
            println!("Remove vector duration: {:?}", remove_vector_duration);
        }

        let average_remove_duration = total_add_duration / added_vectors.len() as u32;
        println!("Average vector adding time: {:?}", average_remove_duration);

        // TODO: Add benchmark for measuring application of dimensionality reduction techniques to data.

        // Benchmark for saving to disk.
        let (saved_file, total_save_duration) = measure_time!({
            save_index(
                &dir_path,
                format!("annoy_serial_{}", amount), // TODO: Modify to support parallel
                IndexType::Annoy(index),
            )
        });

        let index_disk_space = saved_file
            .metadata()
            .expect("Expected metadata in the file")
            .len() as f32
            / (1024.0 * 1024.0); // in mb

        // Benchmark loading time for index.
        let (_, total_load_duration) = measure_time!({
            AnnoyIndex::load_index(&saved_file);
        });

        index_logger.add_record(IndexBenchmarkResult {
            execution_time: total_index_duration.as_secs_f32(),
            index_loading_time: total_load_duration.as_secs_f32(),
            index_restoring_time: 0.0, // TODO;
            index_saving_time: total_save_duration.as_secs_f32(),
            queries_per_second: 0.0, // TODO;
            recall: 0.0,             // TODO;
            search_time: search_report.execution_time.as_secs_f32(),
            scalability_factor: None, // TODO:
            index_disk_space,
            dataset_dimensionality: dimensions,
            dataset_size: amount,
            build_time: build_report.execution_time.as_secs_f32(),
            add_vector_performance: average_add_duration.as_secs_f32(),
            remove_vector_performance: average_remove_duration.as_secs_f32(),
        });
    }

    build_logger
        .write_to_csv(format!("{}/annoy_build.csv", dir_path))
        .expect("Something went wrong while writing to csv");

    // let seed = thread_rng().gen_range(0..10000);
    // let mut annoy_index = AnnoyIndex::new(20, 20, 40, DistanceMetric::Cosine);
    // let mut simhash_index = LSHIndex::new(20, 4, LSHHashType::SimHash, DistanceMetric::Cosine);
    // let mut minhash_index = LSHIndex::new(20, 4, LSHHashType::MinHash, DistanceMetric::Cosine);
    // let mut pq_index = PQIndex::new(3, 50, 256, 0.01, DistanceMetric::Cosine, seed);
    // let mut ivfpq_index = IVFPQIndex::new(3, 100, 200, 256, 0.01, DistanceMetric::Cosine, seed);
    // let mut nsw_index = NSWIndex::new(200, 200, DistanceMetric::Cosine, seed);
    // let mut linscan_index = LinScanIndex::new(DistanceMetric::Euclidean);
    // let mut hnsw_index = HNSWIndex::new(0.5, 16, 200, 200, DistanceMetric::Cosine, seed);

    // println!("Start adding vectors...");
    // for (i, vector) in vectors.iter().enumerate() {
    //     linscan_index.add_vector(vector);
    // annoy_index.add_vector(vector);
    //     simhash_index.add_vector(vector);
    //     minhash_index.add_vector(vector);
    //     pq_index.add_vector(vector);
    //     ivfpq_index.add_vector(vector);
    //     hnsw_index.add_vector(vector);
    //     nsw_index.add_vector(vector);
    // }
    // println!("Done adding vectors...");

    // println!("Start building index...");
    // linscan_index.build_parallel();
    // annoy_index.build();
    // simhash_index.build_parallel();
    // minhash_index.build_parallel();
    // pq_index.build_parallel();
    // ivfpq_index.build_parallel();
    // hnsw_index.build_parallel();
    // nsw_index.build_parallel();
    // println!("Done building index...");

    // let linscan_result = linscan_index.search_parallel(&query_vectors[0], 10);
    // let pq_result = pq_index.search_parallel(&query_vectors[0], 10);
    // let ivfpq_result = ivfpq_index.search_parallel(&query_vectors[0], 10);
    // let simhash_result = simhash_index.search_parallel(&query_vectors[0], 10);
    // let minhash_result = minhash_index.search_parallel(&query_vectors[0], 10);
    // let hnsw_result = hnsw_index.search_parallel(&query_vectors[0], 10);
    // let nsw_result = nsw_index.search_parallel(&query_vectors[0], 10);
    // let annoy_result = annoy_index.search(&query_vectors[0], 10);

    // println!("l1: {:?}", vectors[linscan_result[0].index].indices);
    // println!("l2: {:?}", vectors[linscan_result[1].index].indices);
    // println!("h1: {:?}", vectors[hnsw_result[0].index].indices);
    // println!("h2: {:?}", vectors[hnsw_result[1].index].indices);
    // println!("n1: {:?}", vectors[nsw_result[0].index].indices);
    // println!("n2: {:?}", vectors[nsw_result[1].index].indices);
    // println!("p1: {:?}", vectors[pq_result[0].index].indices);
    // println!("p2: {:?}", vectors[pq_result[1].index].indices);
    // println!("s1: {:?}", vectors[simhash_result[0].index].indices);
    // println!("s2: {:?}", vectors[simhash_result[1].index].indices);
    // println!("i1: {:?}", vectors[ivfpq_result[0].index].indices);
    // println!("i2: {:?}", vectors[ivfpq_result[1].index].indices);
    // println!("a1: {:?}", vectors[annoy_result[0].index].indices);
    // println!("a2: {:?}", vectors[annoy_result[1].index].indices);
    // println!("m1: {:?}", vectors[minhash_result[0].index].indices);
    // println!("m2: {:?}", vectors[minhash_result[1].index].indices);
    // println!("gt: {:?}", groundtruth[0][0].indices);

    // for (i, res) in r.iter().enumerate() {
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
    // }

    /*let start = std::time::Instant::now();
    let mut hit = 0;
    for (i, query_vector) in query_sparse_vectors.iter().enumerate() {
        let result = index.search_parallel(query_vector, 10);

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

    index.build_parallel();*/
    /*let query_id = 0;
    let query_vector = HighDimVector::new(query_id, query_vectors[query_id].to_vec());
    let baseline = &groundtruth[query_id];
    let k = 5;
    let result = index.search_parallel(&query_vector, k);
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

        let result = index.search_parallel(&query, 5);
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
    index.build_parallel();
    let query = HighDimVector::new(999999999, vec![208.0; 3]);
    let result = index.search_parallel(&query, 5);

    for (i, vec) in result.iter().enumerate() {
        println!("{} {:?}", i, vec.id);
        }*/
    /*let args = Args::parse();
    let dimensions = args.dimensions.unwrap_or(100);
    let num_images = args.num_images.unwrap_or(1000);

    //run_benchmark::<NaiveIndex>(dimensions, num_images).await;
    run_benchmark::<SSGIndex>(dimensions, num_images).await;*/
}

fn print_measurement_report(report: &ResourceReport) {
    println!("\n--- Resource Consumption Report ---");
    println!("Initial Memory Usage: {:.2} MB", report.initial_memory);
    println!("Memory Consumed: {:.2} MB", report.final_memory);
    println!("Initial CPU Usage: {:.2}%", report.initial_cpu);
    println!("CPU Consumed: {:.2}%", report.final_cpu);
    println!("Execution Time: {:?}", report.execution_time);
    println!("-----------------------------------\n");
}

fn save_index(dir_path: &String, name: String, index: IndexType) -> File {
    let file_path = format!("{}/{}.ist", dir_path, name);
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .open(&file_path)
        .expect("Failed to open file");
    index.save(&mut file);
    file
}

async fn generate_data(
    config: &BenchmarkConfig,
    dimensions: usize,
    amount: usize,
) -> (Vec<SparseVector>, Vec<SparseVector>, Vec<Vec<SparseVector>>) {
    let seed = thread_rng().gen_range(0..10000);
    let mut generator = SparseDataGenerator::new(
        dimensions,
        amount,
        config.value_range,
        config.sparsity,
        config.distance_metric,
        seed,
    );
    let (vectors, query_vectors, groundtruth) = generator.generate().await;
    (vectors, query_vectors, groundtruth)
}

// async fn run_benchmark(
//     index: &mut IndexType,
//     distance_metric: DistanceMetric,
//     dimensions: usize,
//     num_images: usize,
// ) {
//     let benchmark_config = BenchmarkConfig::new(
//         (dimensions, 100, dimensions),
//         (num_images, 1_000_000, num_images),
//         (0.0, 255.0),
//         0.95,
//         distance_metric,
//     );
//     let mut logger = BenchmarkLogger::new();

//     let mut previous_benchmark_result = None;
//     for (dimensions, amount) in benchmark_config.dataset_configurations() {
//         let generated_data = generate_data(&benchmark_config, dimensions, amount).await;

//         measure_resources!({
//             let result = perform_single_benchmark(
//                 index,
//                 &benchmark_config,
//                 &mut logger,
//                 previous_benchmark_result,
//                 (dimensions, amount),
//             )
//             .await;
//             previous_benchmark_result = Some(result);
//         });
//     }

//     // TODO: Change file name to be more generic with a date.
//     if let Err(e) = logger.write_to_csv("benchmark_results.csv") {
//         eprintln!("Failed to write benchmark results to CSV: {}", e);
//     }
// }

// async fn perform_single_benchmark(
//     index: &mut IndexType,
//     config: &BenchmarkConfig,
//     logger: &mut BenchmarkLogger,
//     previous_benchmark_result: Option<BenchmarkResult>,
//     (dimensions, amount): (usize, usize),
// ) -> BenchmarkResult {
//     println!("------------------------------------");
//     println!(
//         "Benchmarking with number of vectors: {} and dimensions: {}",
//         amount, dimensions
//     );

//     let query_vector = SparseVector {
//         indices: vec![],
//         values: vec![],
//     };
//     let k = 5;
//     let result = measure_benchmark(
//         index,
//         &query_vector,
//         previous_benchmark_result,
//         amount,
//         dimensions,
//         k,
//     );

//     logger.add_record(&result);

//     print_benchmark_results(&result);
//     println!("------------------------------------");
//     result
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
