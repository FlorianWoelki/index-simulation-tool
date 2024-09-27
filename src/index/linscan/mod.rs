use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    sync::Mutex,
};

use ordered_float::OrderedFloat;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    data::{vector::SparseVector, QueryResult},
    data_structures::min_heap::MinHeap,
    index::DistanceMetric,
};

use super::{IndexIdentifier, SparseIndex};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LinScanIndex {
    vectors: Vec<SparseVector>,
    inverted_index: HashMap<usize, Vec<(usize, OrderedFloat<f32>)>>,
    metric: DistanceMetric,
}

impl LinScanIndex {
    pub fn new(metric: DistanceMetric) -> Self {
        LinScanIndex {
            vectors: Vec::new(),
            inverted_index: HashMap::new(),
            metric,
        }
    }
}

impl SparseIndex for LinScanIndex {
    fn add_vector_before_build(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
    }

    fn add_vector(&mut self, vector: &SparseVector) {
        for (index, value) in vector.indices.iter().zip(vector.values.iter()) {
            self.inverted_index
                .entry(*index)
                .or_default()
                .push((self.vectors.len(), *value));
        }

        self.vectors.push(vector.clone());
    }

    fn remove_vector(&mut self, id: usize) -> Option<SparseVector> {
        if id >= self.vectors.len() {
            return None;
        }

        let removed_vector = self.vectors.remove(id);

        // Update the inverted index.
        for index in &removed_vector.indices {
            if let Some(vectors) = self.inverted_index.get_mut(index) {
                vectors.retain(|(vec_id, _)| *vec_id != id);

                for (vec_id, _) in vectors.iter_mut() {
                    if *vec_id > id {
                        *vec_id -= 1;
                    }
                }

                if vectors.is_empty() {
                    self.inverted_index.remove(index);
                }
            }
        }

        // Update the inverted index for all vectors after the removed one.
        for index in id..self.vectors.len() {
            let vector = &self.vectors[index];
            for feature_index in &vector.indices {
                if let Some(vectors) = self.inverted_index.get_mut(feature_index) {
                    if let Some(pos) = vectors.iter().position(|(vec_id, _)| *vec_id == index + 1) {
                        vectors[pos].0 = index;
                    }
                }
            }
        }

        Some(removed_vector)
    }

    fn build(&mut self) {
        self.inverted_index.clear();
        let inverted_index = Mutex::new(HashMap::<usize, Vec<(usize, OrderedFloat<f32>)>>::new());

        self.vectors
            .par_iter()
            .enumerate()
            .for_each(|(vec_id, vector)| {
                for (index, value) in vector.indices.iter().zip(vector.values.iter()) {
                    let mut map = inverted_index.lock().unwrap();
                    map.entry(*index).or_default().push((vec_id, *value));
                }
            });

        self.inverted_index = inverted_index.into_inner().unwrap();
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let scores = Mutex::new(vec![0.0; self.vectors.len()]);

        for (index, value) in query_vector.indices.iter().zip(query_vector.values.iter()) {
            if let Some(vectors) = self.inverted_index.get(index) {
                vectors.par_iter().for_each(|(vec_id, vec_value)| {
                    scores.lock().unwrap()[*vec_id] += value.into_inner() * vec_value.into_inner();
                });
            }
        }

        let mut heap: MinHeap<QueryResult> = MinHeap::new();
        for (index, &score) in scores.lock().unwrap().iter().enumerate() {
            if heap.len() < k || score > heap.peek().unwrap().score.into_inner() {
                heap.push(
                    QueryResult {
                        //vector: self.vectors[index].clone(),
                        index,
                        score: OrderedFloat(score),
                    },
                    OrderedFloat(score),
                );
                if heap.len() > k {
                    heap.pop();
                }
            }
        }

        heap.into_sorted_vec()
    }

    fn save(&self, file: &mut File) {
        let mut writer = BufWriter::new(file);
        let index_type = IndexIdentifier::LinScan.to_u32();
        writer
            .write_all(&index_type.to_be_bytes())
            .expect("Failed to write metadata");
        bincode::serialize_into(&mut writer, &self).expect("Failed to serialize");
    }

    fn load_index(file: &File) -> Self {
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(0)).unwrap();

        let mut buffer = [0u8; 4];
        reader
            .read_exact(&mut buffer)
            .expect("Failed to read metadata");
        let index_type = u32::from_be_bytes(buffer);
        assert_eq!(index_type, IndexIdentifier::LinScan.to_u32());
        bincode::deserialize_from(&mut reader).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{get_complex_vectors, get_simple_vectors, is_in_actual_result};

    use super::*;

    #[test]
    fn test_serde() {
        let mut index = LinScanIndex::new(DistanceMetric::Cosine);
        let (data, _) = get_simple_vectors();
        for vector in &data {
            index.add_vector_before_build(vector);
        }
        index.build();

        let bytes = bincode::serialize(&index).unwrap();
        let reconstructed: LinScanIndex = bincode::deserialize(&bytes).unwrap();

        assert_eq!(index.vectors, reconstructed.vectors);
        assert_eq!(index.metric, reconstructed.metric);
        assert_eq!(index.inverted_index, reconstructed.inverted_index);
    }

    #[test]
    fn test_add_vector() {
        let mut index = LinScanIndex::new(DistanceMetric::Cosine);

        // Create test vectors
        let v1 = SparseVector {
            indices: vec![0, 2, 4],
            values: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
        };
        let v2 = SparseVector {
            indices: vec![1, 2, 3],
            values: vec![OrderedFloat(4.0), OrderedFloat(5.0), OrderedFloat(6.0)],
        };
        let v3 = SparseVector {
            indices: vec![0, 3, 4],
            values: vec![OrderedFloat(7.0), OrderedFloat(8.0), OrderedFloat(9.0)],
        };

        index.add_vector(&v1);
        index.add_vector(&v2);
        index.add_vector(&v3);

        assert_eq!(index.vectors.len(), 3);
        assert_eq!(index.vectors[0], v1);
        assert_eq!(index.vectors[1], v2);
        assert_eq!(index.vectors[2], v3);

        assert_eq!(index.inverted_index.len(), 5); // Should have entries for indices 0, 1, 2, 3, 4

        assert_eq!(
            index.inverted_index.get(&0).unwrap(),
            &vec![(0, OrderedFloat(1.0)), (2, OrderedFloat(7.0))]
        );
        assert_eq!(
            index.inverted_index.get(&1).unwrap(),
            &vec![(1, OrderedFloat(4.0))]
        );
        assert_eq!(
            index.inverted_index.get(&2).unwrap(),
            &vec![(0, OrderedFloat(2.0)), (1, OrderedFloat(5.0))]
        );
        assert_eq!(
            index.inverted_index.get(&3).unwrap(),
            &vec![(1, OrderedFloat(6.0)), (2, OrderedFloat(8.0))]
        );
        assert_eq!(
            index.inverted_index.get(&4).unwrap(),
            &vec![(0, OrderedFloat(3.0)), (2, OrderedFloat(9.0))]
        );
    }

    #[test]
    fn test_remove_vector() {
        let mut index = LinScanIndex::new(DistanceMetric::Cosine);

        // Add some test vectors
        let v1 = SparseVector {
            indices: vec![0, 2, 4],
            values: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
        };
        let v2 = SparseVector {
            indices: vec![1, 2, 3],
            values: vec![OrderedFloat(4.0), OrderedFloat(5.0), OrderedFloat(6.0)],
        };
        let v3 = SparseVector {
            indices: vec![0, 3, 4],
            values: vec![OrderedFloat(7.0), OrderedFloat(8.0), OrderedFloat(9.0)],
        };

        index.add_vector(&v1);
        index.add_vector(&v2);
        index.add_vector(&v3);

        let removed = index.remove_vector(1);
        assert_eq!(removed, Some(v2.clone()));

        assert_eq!(index.vectors.len(), 2);
        assert_eq!(index.vectors[0], v1);
        assert_eq!(index.vectors[1], v3);

        assert_eq!(index.inverted_index.len(), 4); // Should have entries for indices 0, 2, 3, 4

        assert_eq!(
            index.inverted_index.get(&0).unwrap(),
            &vec![(0, OrderedFloat(1.0)), (1, OrderedFloat(7.0))]
        );
        assert_eq!(
            index.inverted_index.get(&2).unwrap(),
            &vec![(0, OrderedFloat(2.0))]
        );
        assert_eq!(
            index.inverted_index.get(&3).unwrap(),
            &vec![(1, OrderedFloat(8.0))]
        );
        assert_eq!(
            index.inverted_index.get(&4).unwrap(),
            &vec![(0, OrderedFloat(3.0)), (1, OrderedFloat(9.0))]
        );

        assert!(!index.inverted_index.contains_key(&1));

        let non_existent = index.remove_vector(5);
        assert_eq!(non_existent, None);
    }

    #[test]
    fn test_linscan_simple() {
        let (data, query_vectors) = get_simple_vectors();

        let mut index = LinScanIndex::new(DistanceMetric::Cosine);
        for vector in &data {
            index.add_vector_before_build(vector);
        }
        index.build();

        let results = index.search(&query_vectors[0], 10);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
    }

    #[test]
    fn test_linscan_complex() {
        let (data, query_vector) = get_complex_vectors();

        let mut index = LinScanIndex::new(DistanceMetric::Cosine);
        for vector in &data {
            index.add_vector_before_build(vector);
        }
        index.build();

        let results = index.search(&query_vector, 2);
        println!("{:?}", results);
        assert!(is_in_actual_result(&data, &query_vector, &results));
    }
}
