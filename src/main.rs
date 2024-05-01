use benchmark::Benchmark;
use data::{generator::DataGenerator, HighDimVector};
use index::{naive::NaiveIndex, DistanceMetric, Index};
use query::{naive::NaiveQuery, Query};

use crate::query::hnsw::HNSWQuery;

mod benchmark;
mod data;
mod index;
mod query;

fn main() {
    // check_image_search();
    run_benchmark::<HNSWQuery>();
    run_benchmark::<NaiveQuery>();
}

fn run_benchmark<Q: Query + 'static>() {
    let dimensions = 100;
    let num_images = 100000;
    let range = (0.0, 255.0); // Range of pixel values for grayscale images.

    println!("generating data...");
    let mut data_generator = DataGenerator::new(dimensions, num_images, range);
    let generated_data = data_generator.generate();
    println!("...done");

    println!("adding vectors to the index data structure...");
    let mut index = Box::new(NaiveIndex::new(DistanceMetric::Euclidean));
    for d in generated_data {
        index.add_vector(HighDimVector::new(d));
    }
    println!("...done");

    let k = 2;
    let query_vector = HighDimVector::new(vec![128.0; dimensions]);
    let query = Box::new(Q::new(query_vector, k));

    println!("start benchmarking...");
    let mut benchmark = Benchmark::new(index, query);
    println!("...done");

    let result = benchmark.run();

    println!("Total Execution time: {:?}", result.total_execution_time);
    println!("Index Execution time: {:?}", result.index_execution_time);
    println!("Query Execution time: {:?}", result.query_execution_time);
    println!("Queries per Second (QPS): {:?}", result.queries_per_second);
    println!("------------------------------------");
}

fn check_image_search() {
    let dimensions = 100;
    let num_images = 1000;
    let range = (0.0, 255.0); // Range of pixel values for grayscale images.

    let mut data_generator = DataGenerator::new(dimensions, num_images, range);
    let image_data = data_generator.generate();
    let mut index = NaiveIndex::new(DistanceMetric::Euclidean);

    for vector in image_data {
        let high_dim_vector = HighDimVector::new(vector);
        index.add_vector(high_dim_vector);
    }

    index.build();

    // Creating a simulated query image with all pixels at intensity 128.
    let query_vector = HighDimVector::new(vec![128.0; dimensions]);

    let k = 10;
    let query = NaiveQuery::new(query_vector, k);
    let query_results = query.execute(index.indexed_data(), index.metric());

    for result in query_results {
        println!("Index: {}, Distance: {}", result.index, result.distance);
    }
}
