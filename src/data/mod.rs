use std::{fs::File, io::Read};

use ordered_float::OrderedFloat;
use rmp_serde::Deserializer;
use serde::Deserialize;
use vector::SparseVector;

pub mod generator_sparse;
pub mod ms_marco;
pub mod pca;
pub mod plot;
pub mod tsne;
pub mod vector;

#[derive(Debug, PartialEq)]
pub struct QueryResult {
    //pub vector: SparseVector,
    pub index: usize,
    pub score: OrderedFloat<f32>,
}

impl Eq for QueryResult {}

impl PartialOrd for QueryResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for QueryResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

pub fn read_sparse_vectors(
    filename: &str,
) -> Result<Vec<SparseVector>, Box<dyn std::error::Error>> {
    let mut file = File::open(filename)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let mut deserializer = Deserializer::new(&buffer[..]);
    let sparse_vectors = Vec::<SparseVector>::deserialize(&mut deserializer)?;

    Ok(sparse_vectors)
}

pub fn read_groundtruth(filename: &str) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    let mut file = File::open(filename)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let mut deserializer = Deserializer::new(&buffer[..]);
    let groundtruth = Vec::<Vec<usize>>::deserialize(&mut deserializer)?;

    Ok(groundtruth)
}

#[cfg(test)]
mod tests {
    use rmp_serde::Serializer;
    use serde::Serialize;
    use tempfile::NamedTempFile;

    use super::*;

    fn create_test_file(vectors: &[SparseVector]) -> NamedTempFile {
        let file = NamedTempFile::new().unwrap();
        let mut serializer = Serializer::new(file.reopen().unwrap());
        vectors.serialize(&mut serializer).unwrap();
        file
    }

    #[test]
    fn test_read_sparse_vectors() {
        let vectors = vec![
            SparseVector {
                indices: vec![1, 3, 5],
                values: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            },
            SparseVector {
                indices: vec![0, 2, 4],
                values: vec![OrderedFloat(0.5), OrderedFloat(1.5), OrderedFloat(2.5)],
            },
        ];
        let file = create_test_file(&vectors);

        let result = read_sparse_vectors(file.path().to_str().unwrap());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vectors);
    }
}
