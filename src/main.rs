use data::{generator::DataGenerator, HighDimVector};
use index::{naive::NaiveIndex, DistanceMetric, Index};
use query::range::range_query;

mod data;
mod index;
mod query;

fn main() {
    check_image_search();
    //check_text_search();
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

    // Creating a simulated query image with all pixels at intensity 128.
    let query_vector = HighDimVector::new(vec![128.0; dimensions]);

    /*if let Some(nearest_vector) = index.find_nearest(&query_vector) {
        println!("Nearest image vector found: {:?}", nearest_vector);
    } else {
        println!("No nearest image vector found.");
        }*/

    let image_query_range = 650.0;
    let image_results = range_query(&index, &query_vector, image_query_range);

    println!("Found {} images within range.", image_results.len());
    for (i, img) in image_results.iter().enumerate() {
        println!("Image {}: {:?}", i, img);
    }
}

fn check_text_search() {
    let dimensions = 300; // Typical dimensions for text embeddings like Word2Vec.
    let num_texts = 100;
    let range = (-1.0, 1.0); // Range for text embeddings, often normalized.

    let mut data_generator = DataGenerator::new(dimensions, num_texts, range);
    let text_data = data_generator.generate();
    let mut index = NaiveIndex::new(DistanceMetric::Cosine);

    for vector in text_data {
        let high_dim_vector = HighDimVector::new(vector);
        index.add_vector(high_dim_vector);
    }

    // Creating neutral text query.
    let query_vector = HighDimVector::new(vec![0.1; dimensions]);

    if let Some(nearest_vector) = index.find_nearest(&query_vector) {
        println!("Nearest text vector found: {:?}", nearest_vector);
    } else {
        println!("No nearest text vector found.");
    }
}
