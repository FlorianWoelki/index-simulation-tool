use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use benchmark::{
    execute_with_timeout,
    logger::BenchmarkLogger,
    macros::measure_system::ResourceReport,
    metrics::{calculate_queries_per_second, calculate_recall, calculate_scalability_factor},
    BenchmarkConfig, GenericBenchmarkResult, IndexBenchmarkResult,
};
use data::{
    generator_sparse::SparseDataGenerator,
    pca::pca,
    plot::{plot_nearest_neighbor_distances, plot_sparsity_distribution},
    read_groundtruth, read_sparse_vectors,
    vector::SparseVector,
};

use rand::{thread_rng, Rng};
use sysinfo::Pid;

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
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

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
    #[clap(long, short = 'm', action)]
    distance_metric: DistanceMetric,
    #[clap(long, short = 't', action)]
    dataset_type: String,
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

    plot_sparsity_distribution(
        &vectors_sparse_vectors,
        format!("amount: {}", vectors_sparse_vectors.len()).as_str(),
    )
    .show();
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
    let dim = 10000;
    let mut data_generator =
        SparseDataGenerator::new(dim, amount, (0.0, 1.0), 0.95, DistanceMetric::Cosine, 42);
    data_generator.generate().await;

    // Get the first element of the groundtruth data
    let groundtruth_flat = data_generator
        .groundtruth
        .iter()
        .map(|nn| data_generator.vectors[nn[0]].clone())
        .collect::<Vec<SparseVector>>();

    plot_sparsity_distribution(
        &data_generator.vectors,
        format!("dim: {}, amount: {}", dim, amount).as_str(),
    )
    .show();
    plot_nearest_neighbor_distances(
        &data_generator.query_vectors,
        &groundtruth_flat,
        &DistanceMetric::Cosine,
    )
    .show();
}

#[allow(dead_code)]
async fn plot_datasets(benchmark_config: &BenchmarkConfig, seeds: &[u64]) {
    for (i, (dimensions, amount)) in benchmark_config.dataset_configurations().enumerate() {
        let seed = seeds[i];
        let data_generator = generate_data(&benchmark_config, dimensions, amount, seed).await;

        plot_sparsity_distribution(
            &data_generator.vectors,
            format!("seed: {}, dim: {}, amount: {}", seed, dimensions, amount).as_str(),
        )
        .show();
    }
}

#[allow(dead_code)]
async fn generate_datasets(
    config: &BenchmarkConfig,
    seeds: &[u64],
    base_path: &str,
) -> Vec<String> {
    let mut file_paths = Vec::new();

    for (i, (dimensions, amount)) in config.dataset_configurations().enumerate() {
        let seed = seeds[i];
        let file_name = format!("dataset_{}_{}_{}.bin", dimensions, amount, seed);
        let file_path = format!("{}/{}", base_path, file_name);

        if Path::new(&file_path).exists() {
            println!("Dataset already exists: {}", file_path);
            file_paths.push(file_path);
            continue;
        }

        println!(
            "Generating dataset: dimensions={}, amount={}, seed={}",
            dimensions, amount, seed
        );
        let data_generator = generate_data(&config, dimensions, amount, seed).await;

        data_generator
            .save_data(&file_path)
            .expect("Failed to save dataset");
        println!("Saved dataset: {}", file_path);
        file_paths.push(file_path);
    }

    file_paths
}

fn create_index(index_type: &str, distance_metric: DistanceMetric, seed: u64) -> IndexType {
    match index_type {
        "hnsw" => IndexType::Hnsw(HNSWIndex::new(0.5, 16, 86, 400, 400, distance_metric)),
        "lsh-simhash" => {
            IndexType::Lsh(LSHIndex::new(32, 8, LSHHashType::SimHash, distance_metric))
        }
        "lsh-minhash" => {
            IndexType::Lsh(LSHIndex::new(32, 8, LSHHashType::MinHash, distance_metric))
        }
        "pq" => IndexType::Pq(PQIndex::new(3, 50, 256, 0.01, distance_metric, seed)),
        "ivfpq" => IndexType::Ivfpq(IVFPQIndex::new(
            6,
            256,
            16,
            500,
            0.01,
            distance_metric,
            seed,
        )),
        "nsw" => IndexType::Nsw(NSWIndex::new(32, 200, 200, distance_metric)),
        "linscan" => IndexType::LinScan(LinScanIndex::new(distance_metric)),
        "annoy" => IndexType::Annoy(AnnoyIndex::new(10, 20, 100, distance_metric)),
        _ => panic!("Unsupported index type"),
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let dataset_type = args.dataset_type;
    let distance_metric = args.distance_metric;
    let index_type_input = args.index_type.as_str();

    let mut index_logger: BenchmarkLogger<IndexBenchmarkResult> = BenchmarkLogger::new();
    let mut build_logger: BenchmarkLogger<GenericBenchmarkResult> = BenchmarkLogger::new();
    let base_path = format!("index/{}/{:?}", dataset_type, distance_metric);
    let dir_path = format!("{}/{}", &base_path, index_type_input);
    fs::create_dir_all(&dir_path).expect("Failed to create directory");

    if dataset_type == "real" {
        let vectors = read_sparse_vectors("./scripts/data.msgpack").unwrap();
        let query_vectors = read_sparse_vectors("./scripts/queries.msgpack").unwrap();
        let groundtruth = read_groundtruth("./scripts/groundtruth.msgpack").unwrap();

        let seed = 42;

        let mut index = create_index(&index_type_input, distance_metric, seed);
        let mut rng = thread_rng();

        let total_index_start = Instant::now();

        for vector in &vectors {
            index.add_vector_before_build(&vector);
        }

        let timeout = Duration::from_secs(5 * 60);

        let index = Arc::new(Mutex::new(index));
        let index_clone = Arc::clone(&index);
        let query_vector = query_vectors[0].clone();

        let result = execute_with_timeout(
            move || {
                let mut index = index_clone.lock().unwrap();
                let (_, build_report) = measure_resources!({
                    println!("\nBuilding the index...");
                    index.build();
                    println!("...finished build the index");
                });

                // Benchmark for measuring searching.
                let (_, search_report) = measure_resources!({
                    index.search(&query_vector, 10);
                });

                (build_report, search_report)
            },
            timeout,
        );

        let (build_report, search_report) = match result {
            Some((build_report, search_report)) => (build_report, search_report),
            None => {
                println!("Build & Search index timed out");
                build_logger.add_record(GenericBenchmarkResult::from(
                    &ResourceReport {
                        execution_time: Duration::new(0, 0),
                        final_cpu: 0.0,
                        final_memory: 0.0,
                        initial_cpu: 0.0,
                        initial_memory: 0.0,
                    },
                    0,
                    0,
                ));
                index_logger.add_record(IndexBenchmarkResult {
                    execution_time: 0.0,
                    index_loading_time: 0.0,
                    index_saving_time: 0.0,
                    queries_per_second: 0.0,
                    recall: 0.0,
                    search_time: 0.0,
                    scalability_factor: None,
                    index_disk_space: 0.0,
                    dataset_dimensionality: 0,
                    dataset_size: 0,
                    build_time: 0.0,
                    add_vector_performance: 0.0,
                    remove_vector_performance: 0.0,
                });
                return;
            }
        };

        let mut index = Arc::try_unwrap(index).unwrap().into_inner().unwrap();

        build_logger.add_record(GenericBenchmarkResult::from(&build_report, 0, 0));

        print_measurement_report(&build_report);
        print_measurement_report(&search_report);

        let total_index_duration = total_index_start.elapsed();

        // Calculate recall.
        let accumulated_recall = Mutex::new(0.0);
        query_vectors
            .par_iter()
            .enumerate()
            .for_each(|(i, query_vector)| {
                let groundtruth_vectors = &groundtruth[i]
                    .iter()
                    .map(|&i| vectors[i].clone())
                    .collect::<Vec<SparseVector>>();
                let k = 10;
                let results = index.search(query_vector, k);
                let search_results = results
                    .iter()
                    .map(|result| vectors[result.index].clone())
                    .collect::<Vec<_>>();

                let iter_recall = calculate_recall(&search_results, groundtruth_vectors, k);
                // println!("{}", iter_recall);
                let mut accumulated_recall = accumulated_recall.lock().unwrap();
                *accumulated_recall += iter_recall;
            });

        let recall = *accumulated_recall.lock().unwrap() / query_vectors.len() as f32;
        println!("Average recall: {:?}", recall);

        // Benchmark for measuring adding a vector to the index.
        println!("Measuring the addition of a vector to the index...");
        let mut added_vectors = vec![
            SparseVector {
                indices: vec![],
                values: vec![]
            };
            1
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
        println!("Measuring the removal of a vector from the index...");
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
        let file_name = format!("{}", index_type_input);
        let (saved_file, total_save_duration) =
            measure_time!({ save_index(&dir_path, file_name.clone(), &index,) });

        let index_disk_space = saved_file
            .metadata()
            .expect("Expected metadata in the file")
            .len() as f32
            / (1024.0 * 1024.0); // in mb

        // Benchmark loading time for index.
        let (_, total_load_duration) = measure_time!({
            IndexType::load_index(&saved_file);
        });

        println!("Measuring queries per second...");
        let num_queries = 25;
        let query_start = Instant::now();
        for query_vector in query_vectors.iter().take(num_queries) {
            index.search(query_vector, 10);
        }
        let query_duration = query_start.elapsed();
        let queries_per_second = calculate_queries_per_second(num_queries, query_duration);
        println!("Queries per second: {:.2}", queries_per_second);

        let new_index_benchmark_result = IndexBenchmarkResult {
            execution_time: total_index_duration.as_secs_f32(),
            index_loading_time: total_load_duration.as_secs_f32(),
            index_saving_time: total_save_duration.as_secs_f32(),
            queries_per_second,
            recall,
            search_time: search_report.execution_time.as_secs_f32(),
            scalability_factor: None,
            index_disk_space,
            dataset_dimensionality: 0,
            dataset_size: 0,
            build_time: build_report.execution_time.as_secs_f32(),
            add_vector_performance: average_add_duration.as_secs_f32(),
            remove_vector_performance: average_remove_duration.as_secs_f32(),
        };

        index_logger.add_record(new_index_benchmark_result);
    } else {
        let dimensions = args.dimensions.unwrap_or(10000);
        let amount = args.features.unwrap_or(1000);

        // let seed = thread_rng().gen_range(0..10000);
        let seeds = vec![42, 7890, 54321, 191098, 1521];

        let mut rng = thread_rng();
        let benchmark_config = BenchmarkConfig::new(
            (dimensions, 50000, dimensions),
            (amount, 5000, amount),
            (0.0, 1.0),
            0.96,
            distance_metric,
        );

        let datasets_dir = format!("{}/datasets", &base_path);
        fs::create_dir_all(&datasets_dir).expect("Failed to create datasets directory");

        let dataset_paths = generate_datasets(&benchmark_config, &seeds, &datasets_dir).await;

        let mut previous_benchmark_result = None;

        for (i, dataset_path) in dataset_paths.iter().enumerate() {
            let seed = seeds[i];
            let amount = benchmark_config.start_num_images * (i + 1);
            let dimensions = benchmark_config.start_dimensions * (i + 1);
            let file_name = format!("{}_{}_{}", index_type_input, amount, dimensions);

            let mut index = create_index(&index_type_input, distance_metric, seed);

            println!("\nLoading data...");
            let data_generator = SparseDataGenerator::new(
                dimensions,
                amount,
                (0.0, 1.0),
                0.96,
                distance_metric,
                seed,
            ); // Dummy values
            let (vectors, query_vectors, groundtruth) = data_generator
                .load_data(dataset_path)
                .expect("Failed to load dataset");
            println!("...finished loading data");

            let timeout = Duration::from_secs(5 * 60);
            let transformed_result = if let Some(reduction_technique) = &args.reduction_technique {
                match reduction_technique.as_str() {
                    "pca" => execute_with_timeout(
                        move || {
                            println!("\nTransforming data with reduction technique...",);
                            let (transformed_vectors, _, _) =
                                pca(&vectors, dimensions, dimensions / 2);
                            println!("✅ Transformed input vectors");
                            let (query_vectors, _, _) =
                                pca(&query_vectors, dimensions, dimensions / 2);
                            println!("✅ Transformed query vectors");

                            (transformed_vectors, query_vectors)
                        },
                        timeout,
                    ),
                    _ => {
                        println!("Unsupported reduction technique: {}", reduction_technique);
                        None
                    }
                }
            } else {
                Some((vectors, query_vectors))
            };

            let (vectors, query_vectors) = match transformed_result {
                Some((vectors, query_vectors)) => (vectors, query_vectors),
                None => {
                    println!("Failed to process vectors due to timeout or unsupported operation");
                    continue;
                }
            };

            let total_index_start = Instant::now();

            for vector in &vectors {
                index.add_vector_before_build(&vector);
            }

            let timeout = Duration::from_secs(5 * 60);

            let index = Arc::new(Mutex::new(index));
            let index_clone = Arc::clone(&index);
            let query_vector = query_vectors[0].clone();

            let result = execute_with_timeout(
                move || {
                    let mut index = index_clone.lock().unwrap();
                    let (_, build_report) = measure_resources!({
                        println!("\nBuilding the index...");
                        index.build();
                        println!("...finished build the index");
                    });

                    // Benchmark for measuring searching.
                    let (_, search_report) = measure_resources!({
                        index.search(&query_vector, 10);

                        // println!("{:?}", vectors[result[0].index]);
                        // println!("{:?}", groundtruth[0][0]);
                    });

                    (build_report, search_report)
                },
                timeout,
            );

            let (build_report, search_report) = match result {
                Some((build_report, search_report)) => (build_report, search_report),
                None => {
                    println!("Build & Search index timed out");
                    build_logger.add_record(GenericBenchmarkResult::from(
                        &ResourceReport {
                            execution_time: Duration::new(0, 0),
                            final_cpu: 0.0,
                            final_memory: 0.0,
                            initial_cpu: 0.0,
                            initial_memory: 0.0,
                        },
                        data_generator.dim,
                        data_generator.count,
                    ));
                    index_logger.add_record(IndexBenchmarkResult {
                        execution_time: 0.0,
                        index_loading_time: 0.0,
                        index_saving_time: 0.0,
                        queries_per_second: 0.0,
                        recall: 0.0,
                        search_time: 0.0,
                        scalability_factor: None,
                        index_disk_space: 0.0,
                        dataset_dimensionality: data_generator.dim,
                        dataset_size: data_generator.count,
                        build_time: 0.0,
                        add_vector_performance: 0.0,
                        remove_vector_performance: 0.0,
                    });
                    continue;
                }
            };

            let mut index = Arc::try_unwrap(index).unwrap().into_inner().unwrap();

            build_logger.add_record(GenericBenchmarkResult::from(
                &build_report,
                data_generator.dim,
                data_generator.count,
            ));

            print_measurement_report(&build_report);
            print_measurement_report(&search_report);

            let total_index_duration = total_index_start.elapsed();

            // Calculate recall.
            let accumulated_recall = Mutex::new(0.0);
            query_vectors
                .par_iter()
                .enumerate()
                .for_each(|(i, query_vector)| {
                    let groundtruth_vectors = &groundtruth[i]
                        .iter()
                        .map(|&i| vectors[i].clone())
                        .collect::<Vec<SparseVector>>();
                    let k = 10;
                    let results = index.search(query_vector, k);
                    let search_results = results
                        .iter()
                        .map(|result| vectors[result.index].clone())
                        .collect::<Vec<_>>();

                    let iter_recall = calculate_recall(&search_results, groundtruth_vectors, k);
                    // println!("{}", iter_recall);
                    let mut accumulated_recall = accumulated_recall.lock().unwrap();
                    *accumulated_recall += iter_recall;
                });

            let recall = *accumulated_recall.lock().unwrap() / query_vectors.len() as f32;
            println!("Average recall: {:?}", recall);

            // Benchmark for measuring adding a vector to the index.
            println!("Measuring the addition of a vector to the index...");
            let mut added_vectors = vec![
                SparseVector {
                    indices: vec![],
                    values: vec![]
                };
                1
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
            println!("Measuring the removal of a vector from the index...");
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
            let (saved_file, total_save_duration) =
                measure_time!({ save_index(&dir_path, file_name.clone(), &index,) });

            let index_disk_space = saved_file
                .metadata()
                .expect("Expected metadata in the file")
                .len() as f32
                / (1024.0 * 1024.0); // in mb

            // Benchmark loading time for index.
            let (_, total_load_duration) = measure_time!({
                IndexType::load_index(&saved_file);
            });

            println!("Measuring queries per second...");
            let num_queries = 25;
            let query_start = Instant::now();
            for query_vector in query_vectors.iter().take(num_queries) {
                index.search(query_vector, 10);
            }
            let query_duration = query_start.elapsed();
            let queries_per_second = calculate_queries_per_second(num_queries, query_duration);
            println!("Queries per second: {:.2}", queries_per_second);

            let scalability_factor = previous_benchmark_result.as_ref().map(|previous_result| {
                calculate_scalability_factor(
                    (queries_per_second, amount, dimensions),
                    previous_result,
                )
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
                dataset_dimensionality: data_generator.dim,
                dataset_size: data_generator.count,
                build_time: build_report.execution_time.as_secs_f32(),
                add_vector_performance: average_add_duration.as_secs_f32(),
                remove_vector_performance: average_remove_duration.as_secs_f32(),
            };

            previous_benchmark_result = Some(new_index_benchmark_result);
            index_logger.add_record(new_index_benchmark_result);
        }
    }

    let file_name = format!("{}", index_type_input);
    index_logger
        .write_to_csv(format!("{}/{}.csv", dir_path, file_name))
        .expect("Something went wrong while writing to csv");

    build_logger
        .write_to_csv(format!("{}/{}_build.csv", dir_path, file_name))
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
    seed: u64,
) -> SparseDataGenerator {
    let mut generator = SparseDataGenerator::new(
        dimensions,
        amount,
        config.value_range,
        config.sparsity,
        config.distance_metric,
        seed,
    );
    generator.generate().await;
    generator
}
