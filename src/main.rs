use benchmark::Benchmark;
use data::{generator::DataGenerator, HighDimVector};
use index::{naive::NaiveIndex, DistanceMetric, Index};
use query::{naive::NaiveQuery, Query};

mod benchmark;
mod data;
mod index;
mod query;

fn main() {
    // check_image_search();
    run_benchmark();
}

fn run_benchmark() {
    let dimensions = 100;
    let num_images = 100000;
    let range = (0.0, 255.0); // Range of pixel values for grayscale images.

    let mut data_generator = DataGenerator::new(dimensions, num_images, range);
    let generated_data = data_generator.generate();

    let mut index = Box::new(NaiveIndex::new(DistanceMetric::Euclidean));
    for d in generated_data {
        index.add_vector(HighDimVector::new(d));
    }

    let k = 10;
    let query_vector = HighDimVector::new(vec![128.0; dimensions]);
    let query = Box::new(NaiveQuery::new(query_vector, k));

    let mut benchmark = Benchmark::new(index, query);

    let result = benchmark.run();

    println!("Total Execution time: {:?}", result.total_execution_time);
    println!("Index Execution time: {:?}", result.index_execution_time);
    println!("Query Execution time: {:?}", result.query_execution_time);
    println!("Precision: {}", result.precision);
    println!("Recall: {}", result.recall);
    println!("F1 Score: {}", result.f1_score);
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
    let query_results = query.execute(index.indexed_data(), DistanceMetric::Euclidean);

    for result in query_results {
        println!("Index: {}, Distance: {}", result.index, result.distance);
    }
}
