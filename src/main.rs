use std::{collections::HashMap, time::Instant};

use data::{
    generator_dense::DenseDataGenerator, generator_sparse::SparseDataGenerator, SparseVector,
};
use index::{
    annoy::AnnoyIndex,
    hnsw::HNSWIndex,
    ivfpq::IVFPQIndex,
    linscan::LinScanIndex,
    lsh::{LSHHashType, LSHIndex},
    nsw::NSWIndex,
    pq::PQIndex,
    DistanceMetric, SparseIndex,
};

use clap::Parser;
use ordered_float::Float;
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use sysinfo::Pid;

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
    num_images: Option<usize>,
}

#[tokio::main]
async fn main() {
    //let (groundtruth, vectors, query_vectors) = data::ms_marco::load_msmarco_dataset().unwrap();
    let amount = 200;
    let mut generator =
        SparseDataGenerator::new(100, amount, (0.0, 10.0), 0.9, DistanceMetric::Cosine);
    let (vectors, query_vectors, groundtruth) = generator.generate().await;

    let seed = thread_rng().gen_range(0..10000);
    let mut annoy_index = AnnoyIndex::new(20, 20, 40, DistanceMetric::Cosine);
    let mut simhash_index = LSHIndex::new(20, 4, LSHHashType::SimHash, DistanceMetric::Cosine);
    let mut minhash_index = LSHIndex::new(20, 4, LSHHashType::MinHash, DistanceMetric::Cosine);
    let mut pq_index = PQIndex::new(3, 50, 256, 0.01, DistanceMetric::Cosine, seed);
    let mut ivfpq_index = IVFPQIndex::new(3, 100, 200, 256, 0.01, DistanceMetric::Cosine, seed);
    let mut nsw_index = NSWIndex::new(200, 200, DistanceMetric::Cosine, seed);
    let mut linscan_index = LinScanIndex::new(DistanceMetric::Euclidean);
    let mut hnsw_index = HNSWIndex::new(0.5, 16, 200, 200, DistanceMetric::Cosine, seed);
    //let mut query_sparse_vectors = vec![];
    //let mut vectors_sparse_vectors = vec![];
    /*for query_vector in query_vectors.iter() {
    let sparse_vector = SparseVector {
        indices: query_vector.0.clone(),
        values: query_vector
            .1
            .iter()
            .map(|&v| ordered_float::OrderedFloat(v))
            .collect(),
    };
    query_sparse_vectors.push(sparse_vector);
    }*/
    /*for vector in vectors.iter().take(amount) {
    let sparse_vector = SparseVector {
        indices: vector.0.clone(),
        values: vector
            .1
            .iter()
            .map(|&v| ordered_float::OrderedFloat(v))
            .collect(),
    };
    vectors_sparse_vectors.push(sparse_vector);
    }*/
    println!("Start adding vectors...");
    for (i, vector) in vectors.iter().enumerate() {
        linscan_index.add_vector(vector);
        annoy_index.add_vector(vector);
        simhash_index.add_vector(vector);
        minhash_index.add_vector(vector);
        pq_index.add_vector(vector);
        ivfpq_index.add_vector(vector);
        hnsw_index.add_vector(vector);
        nsw_index.add_vector(vector);
        println!("Vector {}", i)
    }
    println!("Done adding vectors...");

    println!("Start building index...");
    linscan_index.build();
    annoy_index.build();
    simhash_index.build();
    minhash_index.build();
    pq_index.build();
    ivfpq_index.build();
    hnsw_index.build();
    nsw_index.build();
    println!("Done building index...");

    let linscan_result = linscan_index.search(&query_vectors[0], 10);
    let pq_result = pq_index.search(&query_vectors[0], 10);
    let ivfpq_result = ivfpq_index.search(&query_vectors[0], 10);
    let simhash_result = simhash_index.search(&query_vectors[0], 10);
    let minhash_result = minhash_index.search(&query_vectors[0], 10);
    let hnsw_result = hnsw_index.search(&query_vectors[0], 10);
    let nsw_result = nsw_index.search(&query_vectors[0], 10);
    let annoy_result = annoy_index.search(&query_vectors[0], 10);
    // println!("{:?}", r);
    //let r = index.search(&vectors[thread_rng().gen_range(0..amount)], 10);
    // println!("{:?}", r);

    println!("l1: {:?}", vectors[linscan_result[0].index].indices);
    println!("l2: {:?}", vectors[linscan_result[1].index].indices);
    println!("h1: {:?}", vectors[hnsw_result[0].index].indices);
    println!("h2: {:?}", vectors[hnsw_result[1].index].indices);
    println!("n1: {:?}", vectors[nsw_result[0].index].indices);
    println!("n2: {:?}", vectors[nsw_result[1].index].indices);
    println!("p1: {:?}", vectors[pq_result[0].index].indices);
    println!("p2: {:?}", vectors[pq_result[1].index].indices);
    println!("s1: {:?}", vectors[simhash_result[0].index].indices);
    println!("s2: {:?}", vectors[simhash_result[1].index].indices);
    println!("i1: {:?}", vectors[ivfpq_result[0].index].indices);
    println!("i2: {:?}", vectors[ivfpq_result[1].index].indices);
    println!("a1: {:?}", vectors[annoy_result[0].index].indices);
    println!("a2: {:?}", vectors[annoy_result[1].index].indices);
    println!("m1: {:?}", vectors[minhash_result[0].index].indices);
    println!("m2: {:?}", vectors[minhash_result[1].index].indices);
    println!("gt: {:?}", groundtruth[0][0].indices);

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
