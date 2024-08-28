use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
};

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::data::{QueryResult, SparseVector};

use super::{DistanceMetric, IndexIdentifier, SparseIndex};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NSWIndex {
    vectors: Vec<SparseVector>,
    /// Number of nearest neighbors to consider during index construction.
    /// Higher values improve recall but increase build time and memory usage.
    ef_construction: usize,
    /// Number of nearest neighbors to consider during search.
    /// Higher values improve recall but increase search time.
    ef_search: usize,
    metric: DistanceMetric,
    graph: HashMap<usize, Vec<usize>>,
    /// Number of bidirectional links created for every new element during
    /// construction. Controls the connectivity of the graph.
    /// Lower values lead to a more sparse graph, which can speed up the
    /// build time and reduce memory usage but might decrease the recall.
    /// Higher values increase the number of connections, potentially
    /// improving recall at the expense of increased memory usage and
    /// longer build times.
    m: usize,
}

impl NSWIndex {
    pub fn new(m: usize, ef_construction: usize, ef_search: usize, metric: DistanceMetric) -> Self {
        NSWIndex {
            vectors: Vec::new(),
            graph: HashMap::new(),
            ef_construction,
            ef_search,
            metric,
            m,
        }
    }

    fn search_graph(
        &self,
        query_vector: &SparseVector,
        entry_point: usize,
        ef: usize,
    ) -> Vec<(f32, usize)> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut nearest = BinaryHeap::new();

        let initial_dist = query_vector.distance(&self.vectors[entry_point], &self.metric);
        candidates.push(Reverse((OrderedFloat(initial_dist), entry_point)));
        nearest.push((OrderedFloat(initial_dist), entry_point));
        visited.insert(entry_point);

        while let Some(Reverse((dist, current))) = candidates.pop() {
            if nearest
                .peek()
                .map_or(false, |&(top_dist, _)| dist > top_dist)
                && nearest.len() >= ef
            {
                break;
            }

            if let Some(neighbors) = self.graph.get(&current) {
                for &neighbor in neighbors {
                    if !visited.insert(neighbor) {
                        continue;
                    }

                    let neighbor_dist =
                        query_vector.distance(&self.vectors[neighbor], &self.metric);
                    let neighbor_dist = OrderedFloat(neighbor_dist);

                    if nearest.len() < ef || neighbor_dist < nearest.peek().unwrap().0 {
                        candidates.push(Reverse((neighbor_dist, neighbor)));
                        nearest.push((neighbor_dist, neighbor));

                        if nearest.len() > ef {
                            nearest.pop();
                        }
                    }
                }
            }
        }

        nearest
            .into_sorted_vec()
            .into_iter()
            .map(|(dist, idx)| (dist.into_inner(), idx))
            .collect()
    }

    fn update_graph(&mut self, vector_index: usize, neighbor: usize) {
        self.graph.entry(vector_index).or_default().push(neighbor);
        self.graph.entry(neighbor).or_default().push(vector_index);
    }
}

impl SparseIndex for NSWIndex {
    fn add_vector_before_build(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
    }

    fn add_vector(&mut self, vector: &SparseVector) {
        let vector_index = self.vectors.len();
        self.vectors.push(vector.clone());

        if self.graph.is_empty() {
            self.graph.insert(vector_index, Vec::new());
            return;
        }

        let entry_point = *self.graph.keys().next().unwrap();
        let nearest = self.search_graph(vector, entry_point, self.ef_construction);

        for &(_, neighbor) in nearest.iter().take(self.m) {
            self.update_graph(vector_index, neighbor);
        }
    }

    fn remove_vector(&mut self, index: usize) -> Option<SparseVector> {
        if index >= self.vectors.len() {
            return None;
        }

        let removed_vector = self.vectors.swap_remove(index);
        self.graph.remove(&index);

        for connections in self.graph.values_mut() {
            connections.retain(|&x| x != index);
        }

        let last_index = self.vectors.len();
        if index != last_index {
            if let Some(connections) = self.graph.remove(&last_index) {
                self.graph.insert(index, connections);
            }

            for connections in self.graph.values_mut() {
                for connection in connections.iter_mut() {
                    if *connection == last_index {
                        *connection = index;
                    }
                }
            }
        }

        Some(removed_vector)
    }

    fn build(&mut self) {
        let vector_count = self.vectors.len();
        for vector_index in 0..vector_count {
            if self.graph.is_empty() {
                self.graph.insert(vector_index, Vec::new());
                continue;
            }

            let entry_point = *self.graph.keys().next().unwrap();
            let nearest = self.search_graph(
                &self.vectors[vector_index],
                entry_point,
                self.ef_construction,
            );

            for &(_, neighbor) in nearest.iter().take(self.m) {
                self.update_graph(vector_index, neighbor);
            }
        }
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        if let Some(&entry_point) = self.graph.keys().next() {
            let nearest = self.search_graph(query_vector, entry_point, k);
            nearest
                .into_iter()
                .map(|(distance, id)| QueryResult {
                    index: id,
                    score: OrderedFloat(distance),
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    fn save(&self, file: &mut File) {
        let mut writer = BufWriter::new(file);
        let index_type = IndexIdentifier::NSW.to_u32();
        writer
            .write_all(&index_type.to_be_bytes())
            .expect("Failed to write metdata");
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
        assert_eq!(index_type, IndexIdentifier::NSW.to_u32());
        bincode::deserialize_from(reader).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use crate::test_utils::{get_complex_vectors, get_simple_vectors, is_in_actual_result};

    use super::*;

    #[test]
    fn test_serde() {
        let (data, _) = get_simple_vectors();
        let mut index = NSWIndex::new(3, 200, 200, DistanceMetric::Euclidean);
        for vector in &data {
            index.add_vector_before_build(vector);
        }

        let bytes = bincode::serialize(&index).unwrap();
        let reconstructed: NSWIndex = bincode::deserialize(&bytes).unwrap();

        assert_eq!(index.vectors, reconstructed.vectors);
        assert_eq!(index.metric, reconstructed.metric);
        assert_eq!(index.m, reconstructed.m);
        assert_eq!(index.graph, reconstructed.graph);
        assert_eq!(index.ef_construction, reconstructed.ef_construction);
        assert_eq!(index.ef_search, reconstructed.ef_search);
    }

    #[test]
    fn test_add_vector() {
        let mut index = NSWIndex::new(8, 200, 200, DistanceMetric::Euclidean);

        let (vectors, _) = get_simple_vectors();
        for vector in &vectors {
            index.add_vector_before_build(vector);
        }
        index.build();

        assert_eq!(index.vectors.len(), vectors.len());

        let new_vector = SparseVector {
            indices: vec![1, 3],
            values: vec![OrderedFloat(4.0), OrderedFloat(5.0)],
        };
        index.add_vector(&new_vector);

        assert_eq!(index.vectors.len(), vectors.len() + 1);
        assert_eq!(index.vectors[index.vectors.len() - 1], new_vector);

        let results = index.search(&new_vector, 2);

        println!("{:?}", results);

        assert_eq!(results[0].index, vectors.len());
        assert_eq!(results[1].index, 1);
    }

    #[test]
    fn test_remove_vector() {
        let mut index = NSWIndex::new(8, 10, 10, DistanceMetric::Cosine);

        let mut vectors = vec![];
        for i in 0..5 {
            let vector = SparseVector {
                indices: vec![i],
                values: vec![OrderedFloat(1.0)],
            };
            vectors.push(vector);
        }

        for vector in &vectors {
            index.add_vector_before_build(vector);
        }

        index.build();

        let removed = index.remove_vector(2);

        assert_eq!(removed, Some(vectors[2].clone()));
        assert_eq!(index.vectors.len(), 4);

        assert_eq!(index.vectors[0], vectors[0].clone());
        assert_eq!(index.vectors[1], vectors[1].clone());
        assert_eq!(index.vectors[2], vectors[4].clone());
        assert_eq!(index.vectors[3], vectors[3].clone());

        // Check that the removed index is not present in any connections
        for (_, connections) in index.graph.iter() {
            assert!(!connections.contains(&4));
        }

        // Check that the last vector's connections have been updated
        if let Some(connections) = index.graph.get(&2) {
            for &neighbor in connections {
                assert!(index.graph.get(&neighbor).unwrap().contains(&2));
            }
        }
    }

    #[test]
    fn test_nsw_index_simple() {
        let mut index = NSWIndex::new(2, 50, 50, DistanceMetric::Euclidean);

        let (data, query_vectors) = get_simple_vectors();
        for vector in &data {
            index.add_vector_before_build(vector);
        }

        index.build();

        let results = index.search(&query_vectors[0], 2);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
    }

    #[test]
    fn test_nsw_index_complex() {
        let mut index = NSWIndex::new(16, 200, 200, DistanceMetric::Euclidean);

        let (data, query_vector) = get_complex_vectors();

        for vector in &data {
            index.add_vector_before_build(vector);
        }

        index.build();

        let results = index.search(&query_vector, 2);
        assert!(is_in_actual_result(&data, &query_vector, &results));
    }
}
