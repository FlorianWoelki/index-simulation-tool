use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    sync::{mpsc, Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use benchmark::{
    execute_with_timeout,
    logger::BenchmarkLogger,
    macros::{measure_system::ResourceReport, measure_time},
    metrics::{calculate_queries_per_second, calculate_recall, calculate_scalability_factor},
    BenchmarkConfig, GenericBenchmarkResult, IndexBenchmarkResult,
};
use chrono::Local;
use data::{
    generator_sparse::SparseDataGenerator,
    pca::pca,
    plot::{plot_nearest_neighbor_distances, plot_sparsity_distribution},
    vector::SparseVector,
};

use rand::{thread_rng, Rng};
use sysinfo::{Pid, System};

use clap::Parser;
use index::{
    annoy::AnnoyIndex,
    hnsw::HNSWIndex,
    ivfpq::IVFPQIndex,
    linscan::LinScanIndex,
    lsh::{LSHHashType, LSHIndex},
    nsw::NSWIndex,
    pq::PQIndex,
    DistanceMetric, IndexType, SparseIndex,
};
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
    #[clap(long, short, action)]
    index_type: String,
    #[clap(long, short, action)]
    reduction_technique: Option<String>,
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
    let dimensions = args.dimensions.unwrap_or(500);
    let amount = args.features.unwrap_or(500);

    let distance_metric = DistanceMetric::Cosine;

    let seed = thread_rng().gen_range(0..10000);
    let index_type_input = args.index_type.as_str();

    let mut rng = thread_rng();
    let benchmark_config = BenchmarkConfig::new(
        (dimensions, 500, dimensions),
        (amount, 500, amount),
        (0.0, 1.0),
        0.90,
        distance_metric,
    );

    let mut index_logger: BenchmarkLogger<IndexBenchmarkResult> = BenchmarkLogger::new();
    let mut build_logger: BenchmarkLogger<GenericBenchmarkResult> = BenchmarkLogger::new();

    let current_date = Local::now().format("%Y-%m-%d").to_string();
    let dir_path = format!("index/{}", &current_date);
    fs::create_dir_all(&dir_path).expect("Failed to create directory");

    let mut previous_benchmark_result = None;
    for (dimensions, amount) in benchmark_config.dataset_configurations() {
        let mut index: IndexType = match index_type_input {
            "hnsw" => IndexType::HNSW(HNSWIndex::new(0.5, 32, 32, 400, 200, distance_metric)),
            "lsh-simhash" => {
                IndexType::LSH(LSHIndex::new(20, 4, LSHHashType::SimHash, distance_metric))
            }
            "lsh-minhash" => {
                IndexType::LSH(LSHIndex::new(20, 4, LSHHashType::MinHash, distance_metric))
            }
            "pq" => IndexType::PQ(PQIndex::new(3, 50, 256, 0.01, distance_metric, seed)),
            "ivfpq" => IndexType::IVFPQ(IVFPQIndex::new(
                3,
                100,
                200,
                256,
                0.01,
                distance_metric,
                seed,
            )),
            "nsw" => IndexType::NSW(NSWIndex::new(32, 200, 200, distance_metric)),
            "linscan" => IndexType::LinScan(LinScanIndex::new(distance_metric)),
            "annoy" => IndexType::Annoy(AnnoyIndex::new(4, 20, 40, distance_metric)),
            _ => panic!("Unsupported index type"),
        };

        println!("\nGenerating data...");
        let (vectors, query_vectors, groundtruth) =
            generate_data(&benchmark_config, dimensions, amount).await;
        println!("...finished generating data");

        let timeout = Duration::from_secs(30);
        let vectors = if let Some(reduction_technique) = &args.reduction_technique {
            match reduction_technique.as_str() {
                "pca" => execute_with_timeout(
                    move || {
                        println!("\nTransforming data with reduction technique...",);
                        let (transformed_vectors, _, _) = pca(&vectors, dimensions, dimensions / 2);

                        // TODO: Transform query and groundtruth vectors as well.

                        transformed_vectors
                    },
                    timeout,
                ),
                _ => {
                    println!("Unsupported reduction technique: {}", reduction_technique);
                    None
                }
            }
        } else {
            Some(vectors)
        };

        let vectors = match vectors {
            Some(vectors) => vectors,
            None => {
                println!("Failed to process vectors due to timeout or unsupported operation");
                continue;
            }
        };

        let total_index_start = Instant::now();

        for vector in &vectors {
            index.add_vector_before_build(&vector);
        }

        let timeout = Duration::from_secs(30);

        let (tx, rx) = mpsc::channel();

        let index = Arc::new(Mutex::new(index));
        let index_clone = Arc::clone(&index);
        let query_vector = query_vectors[0].clone();

        // Spawns a new thread to have a timeout to cancel this thread.
        // TODO: Use execute_with_timeout function.
        thread::spawn(move || {
            let mut index = index_clone.lock().unwrap();
            let (_, build_report) = measure_resources!({
                index.build();
            });

            // Benchmark for measuring searching.
            let (_, search_report) = measure_resources!({
                index.search(&query_vector, 5);

                // println!("{:?}", vectors[result[0].index]);
                // println!("{:?}", groundtruth[0][0]);
            });

            tx.send((build_report, search_report)).unwrap();
        });

        let (build_report, search_report) = match rx.recv_timeout(timeout) {
            Ok((build_report, search_report)) => (Some(build_report), Some(search_report)),
            Err(_) => {
                println!("Build & Search index timed out");
                continue;
            }
        };

        let build_report = build_report.unwrap();
        let search_report = search_report.unwrap();

        let mut index = Arc::try_unwrap(index).unwrap().into_inner().unwrap();

        build_logger.add_record(GenericBenchmarkResult::from(
            &build_report,
            dimensions,
            amount,
        ));
        print_measurement_report(&build_report);

        print_measurement_report(&search_report);

        let total_index_duration = total_index_start.elapsed();

        // Calculate recall.
        let mut accumulated_recall = 0.0;
        for (i, query_vector) in query_vectors.iter().enumerate() {
            let groundtruth_vectors = &groundtruth[i];
            let k = 10;
            let results = index.search(query_vector, k);
            let search_results = results
                .iter()
                .map(|result| vectors[result.index].clone())
                .collect::<Vec<_>>();

            let iter_recall = calculate_recall(&search_results, groundtruth_vectors, k);
            println!("{}", iter_recall);
            accumulated_recall += iter_recall;
        }

        let recall = accumulated_recall / query_vectors.len() as f32;
        println!("Average recall: {:?}", recall);

        // Benchmark for measuring adding a vector to the index.
        println!("Measuring the addition of vectors to the index...");
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
        }

        let average_add_duration = total_add_duration / added_vectors.len() as u32;
        println!("Average vector adding time: {:?}", average_add_duration);
        println!("...finished\n");

        // Benchmark for measuring removing a vector from the index.
        println!("Measuring the removal of vectors from the index...");
        let mut total_remove_duration = Duration::new(0, 0);
        for _ in 0..added_vectors.len() {
            let remove_vector_start = Instant::now();
            index.remove_vector(vectors.len() + 1); // Always this because the vector gets removed from the array, therefore reducing the length.
            let remove_vector_duration = remove_vector_start.elapsed();

            total_remove_duration += remove_vector_duration;
        }

        let average_remove_duration = total_remove_duration / added_vectors.len() as u32;
        println!("Average vector removal time: {:?}", average_remove_duration);
        println!("...finished\n");

        // Benchmark for saving to disk.
        let (saved_file, total_save_duration) = measure_time!({
            save_index(
                &dir_path,
                format!("{}_serial_{}", index_type_input, amount), // TODO: Modify to support parallel
                &index,
            )
        });

        let index_disk_space = saved_file
            .metadata()
            .expect("Expected metadata in the file")
            .len() as f32
            / (1024.0 * 1024.0); // in mb

        // Benchmark loading time for index.
        let (_, total_load_duration) = measure_time!({
            IndexType::load_index(&saved_file);
        });

        let queries_per_second = calculate_queries_per_second(search_report.execution_time);
        let scalability_factor = previous_benchmark_result.as_ref().map(|previous_result| {
            calculate_scalability_factor((queries_per_second, amount, dimensions), previous_result)
        });
        let new_index_benchmark_result = IndexBenchmarkResult {
            execution_time: total_index_duration.as_secs_f32(),
            index_loading_time: total_load_duration.as_secs_f32(),
            index_saving_time: total_save_duration.as_secs_f32(),
            queries_per_second,
            recall,
            search_time: search_report.execution_time.as_secs_f32(),
            scalability_factor,
            index_disk_space,
            dataset_dimensionality: dimensions,
            dataset_size: amount,
            build_time: build_report.execution_time.as_secs_f32(),
            add_vector_performance: average_add_duration.as_secs_f32(),
            remove_vector_performance: average_remove_duration.as_secs_f32(),
        };

        previous_benchmark_result = Some(new_index_benchmark_result);
        index_logger.add_record(new_index_benchmark_result);
    }

    index_logger
        .write_to_csv(format!("{}/{}.csv", dir_path, index_type_input))
        .expect("Something went wrong while writing to csv");

    build_logger
        .write_to_csv(format!("{}/{}_build.csv", dir_path, index_type_input))
        .expect("Something went wrong while writing to csv");
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

fn save_index(dir_path: &String, name: String, index: &IndexType) -> File {
    let file_path = format!("{}/{}.ist", dir_path, name);
    let mut file = OpenOptions::new()
        .write(true)
        .read(true)
        .create(true)
        .open(&file_path)
        .expect("Failed to open file");
    index.save(&mut file);
    file.flush().expect("Issue with flushing file");
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
