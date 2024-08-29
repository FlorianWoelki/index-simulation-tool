use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
};

use ordered_float::OrderedFloat;
use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::data::{vector::SparseVector, QueryResult};

use super::{DistanceMetric, IndexIdentifier, SparseIndex};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HNSWIndex {
    vectors: Vec<SparseVector>,
    /// Factor controlling the distribution of levels in the hierarchical graph.
    /// Values closer to 1 result in more levels, potentially improving search speed.
    /// at the cost of increased memory usage. Typical values range from 0.3 to 0.5.
    level_distribution_factor: f32,
    /// Maximum number of layers in the graph.
    /// Higher values can improve search speed for large datasets, but increase
    /// memory usage.
    max_layers: usize,
    /// Number of nearest neighbors to consider during index construction.
    /// Higher values improve recall but increase build time and memory usage.
    ef_construction: usize,
    /// Number of nearest neighbors to consider during search.
    /// Higher values improve recall but increase search time.
    ef_search: usize,
    metric: DistanceMetric,
    entry_point: Option<usize>,
    max_level: usize,
    graph: HashMap<usize, HashMap<usize, Vec<usize>>>,
    element_levels: HashMap<usize, usize>,
    /// Number of bidirectional links created for every new element during
    /// construction. Controls the connectivity of the graph.
    /// Lower values lead to a more sparse graph, which can speed up the
    /// build time and reduce memory usage but might decrease the recall.
    /// Higher values increase the number of connections, potentially
    /// improving recall at the expense of increased memory usage and
    /// longer build times.
    m: usize,
}

impl HNSWIndex {
    pub fn new(
        level_distribution_factor: f32,
        max_layers: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: DistanceMetric,
    ) -> Self {
        HNSWIndex {
            vectors: Vec::new(),
            graph: HashMap::new(),
            element_levels: HashMap::new(),
            entry_point: None,
            level_distribution_factor,
            max_layers,
            max_level: 0,
            ef_construction,
            ef_search,
            metric,
            m,
        }
    }

    fn search_layer(
        &self,
        query_vector: &SparseVector,
        entry_point: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(f32, usize)> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        candidates.push(Reverse((
            OrderedFloat(query_vector.distance(&self.vectors[entry_point], &self.metric)),
            entry_point,
        )));
        let mut nearest: Vec<(f32, usize)> = Vec::new();
        let mut unique_indices = HashSet::new();

        while !candidates.is_empty() {
            let (dist, current) = candidates.pop().unwrap().0;

            let should_terminate =
                !nearest.is_empty() && dist.into_inner() > nearest[0].0 && nearest.len() >= ef;

            if should_terminate {
                break;
            }

            if unique_indices.insert(current) {
                if nearest.len() < ef {
                    nearest.push((dist.into_inner(), current));
                    nearest.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                } else if dist.into_inner() < nearest[ef - 1].0 {
                    nearest.pop();
                    nearest.push((dist.into_inner(), current));
                    nearest.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                }
            }

            if let Some(neighbors) = self
                .graph
                .get(&current)
                .and_then(|layers| layers.get(&layer))
            {
                let new_candidates: Vec<_> = neighbors
                    .par_iter()
                    .filter(|&&neighbor| !visited.contains(&neighbor))
                    .map(|&neighbor| {
                        let neighbor_dist =
                            query_vector.distance(&self.vectors[neighbor], &self.metric);
                        (neighbor, neighbor_dist)
                    })
                    .collect();

                new_candidates
                    .into_iter()
                    .for_each(|(neighbor, neighbor_dist)| {
                        visited.insert(neighbor);
                        candidates.push(Reverse((OrderedFloat(neighbor_dist), neighbor)))
                    })
            }
        }

        nearest
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut level = 0;
        while rng.gen::<f32>() < self.level_distribution_factor && level < self.max_layers {
            level += 1;
        }
        level
    }

    fn update_graph(&mut self, vector_index: usize, neighbor: usize, layer: usize) {
        self.graph
            .entry(vector_index)
            .or_default()
            .entry(layer)
            .or_default()
            .push(neighbor);
        self.graph
            .entry(neighbor)
            .or_default()
            .entry(layer)
            .or_default()
            .push(vector_index);
    }
}

impl SparseIndex for HNSWIndex {
    fn add_vector_before_build(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
    }

    fn add_vector(&mut self, vector: &SparseVector) {
        let vector_index = self.vectors.len();
        self.vectors.push(vector.clone());

        if self.graph.is_empty() {
            self.entry_point = Some(vector_index);
            self.graph.insert(vector_index, HashMap::new());
            self.element_levels.insert(vector_index, 0);
            return;
        }

        let level = self.random_level();
        self.element_levels.insert(vector_index, level);

        let mut current_node = self.entry_point.unwrap();
        for layer in (0..=self.max_level.min(level)).rev() {
            let nearest = self.search_layer(&vector, current_node, self.ef_construction, layer);
            for &(_, neighbor) in nearest.iter().take(self.m) {
                self.update_graph(vector_index, neighbor, layer);
            }

            if layer > 0 {
                current_node = nearest[0].1;
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(vector_index);
        }
    }

    fn remove_vector(&mut self, index: usize) -> Option<SparseVector> {
        if index >= self.vectors.len() {
            return None;
        }

        // Remove the vector from the vectors list
        let removed_vector = self.vectors.swap_remove(index);

        // Update the graph and element_levels
        if let Some(level) = self.element_levels.remove(&index) {
            // Collect all the neighbors that need updating
            let mut neighbors_to_update = Vec::new();
            if let Some(layers) = self.graph.get(&index) {
                for (&layer, neighbors) in layers.iter().take(level + 1) {
                    for &neighbor in neighbors {
                        neighbors_to_update.push((layer, neighbor));
                    }
                }
            }

            // Update the neighbors
            for (layer, neighbor) in neighbors_to_update {
                if let Some(neighbor_connections) = self
                    .graph
                    .get_mut(&neighbor)
                    .and_then(|layers| layers.get_mut(&layer))
                {
                    neighbor_connections.retain(|&x| x != index);
                }
            }

            // Remove the index from the graph
            self.graph.remove(&index);

            // Update max_level if necessary
            if level == self.max_level {
                self.max_level = self.element_levels.values().max().cloned().unwrap_or(0);
            }
        }

        // Update entry point if necessary
        if Some(index) == self.entry_point {
            self.entry_point = self.graph.keys().next().cloned();
        }

        // Update indices for the swapped element
        let last_index = self.vectors.len();
        if index != last_index {
            if let Some(swapped_level) = self.element_levels.remove(&last_index) {
                self.element_levels.insert(index, swapped_level);

                // Update max_level if the swapped element was at the highest level
                if swapped_level > self.max_level {
                    self.max_level = swapped_level;
                }
            }

            if let Some(layers) = self.graph.remove(&last_index) {
                self.graph.insert(index, layers);
            }

            // Update references to the swapped element in other nodes
            for (_, layers) in self.graph.iter_mut() {
                for (_, connections) in layers.iter_mut() {
                    for connection in connections.iter_mut() {
                        if *connection == last_index {
                            *connection = index;
                        }
                    }
                }
            }

            // Update entry point if it was the last element
            if Some(last_index) == self.entry_point {
                self.entry_point = Some(index);
            }
        }

        Some(removed_vector)
    }

    fn build(&mut self) {
        let vector_count = self.vectors.len();
        for vector_index in 0..vector_count {
            if self.graph.is_empty() {
                self.entry_point = Some(vector_index);
                self.graph.insert(vector_index, HashMap::new());
                self.element_levels.insert(vector_index, 0);
                continue;
            }

            let level = self.random_level();
            self.element_levels.insert(vector_index, level);

            let mut current_node = self.entry_point.unwrap();
            for layer in (0..=self.max_level.min(level)).rev() {
                let nearest = {
                    let vector = &self.vectors[vector_index];
                    self.search_layer(&vector, current_node, self.ef_construction, layer)
                };

                for &(_, neighbor) in nearest.iter().take(self.m) {
                    self.update_graph(vector_index, neighbor, layer);
                }

                if layer > 0 {
                    current_node = nearest[0].1;
                }
            }

            if level > self.max_level {
                self.max_level = level;
                self.entry_point = Some(vector_index);
            }
        }
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        if let Some(entry_point) = self.entry_point {
            let mut current_node = entry_point;
            for layer in (0..=self.max_level).rev() {
                current_node = self.search_layer(query_vector, current_node, 1, layer)[0].1;
            }

            let nearest = self.search_layer(query_vector, current_node, k, 0);
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
        let index_type = IndexIdentifier::HNSW.to_u32();
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
        assert_eq!(index_type, IndexIdentifier::HNSW.to_u32());
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
        let mut index = HNSWIndex::new(0.5, 32, 3, 200, 200, DistanceMetric::Euclidean);
        for vector in &data {
            index.add_vector_before_build(vector);
        }

        let bytes = bincode::serialize(&index).unwrap();
        let reconstructed: HNSWIndex = bincode::deserialize(&bytes).unwrap();

        assert_eq!(index.vectors, reconstructed.vectors);
        assert_eq!(index.metric, reconstructed.metric);
        assert_eq!(index.element_levels, reconstructed.element_levels);
        assert_eq!(index.entry_point, reconstructed.entry_point);
        assert_eq!(index.m, reconstructed.m);
        assert_eq!(index.graph, reconstructed.graph);
        assert_eq!(
            index.level_distribution_factor,
            reconstructed.level_distribution_factor
        );
        assert_eq!(index.max_layers, reconstructed.max_layers);
        assert_eq!(index.ef_construction, reconstructed.ef_construction);
        assert_eq!(index.ef_search, reconstructed.ef_search);
    }

    #[test]
    fn test_add_vector() {
        let mut index = HNSWIndex::new(0.5, 32, 3, 200, 200, DistanceMetric::Euclidean);

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

        assert_eq!(results[0].index, vectors.len());
        assert_eq!(results[1].index, 1);
    }

    #[test]
    fn test_remove_vector() {
        let mut index = HNSWIndex::new(0.5, 32, 8, 10, 10, DistanceMetric::Cosine);

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

        // Check if the graph structure is updated correctly
        for (node_id, layers) in &index.graph {
            for (_, connections) in layers {
                // Ensure no connections point to the old last index (4)
                assert!(!connections.contains(&4));

                // Check if connections are within the valid range (0-3)
                for &connection in connections {
                    assert!(connection < 4);
                }

                // If this is the swapped node (2), check its connections
                if *node_id == 2 {
                    assert!(!connections.is_empty());
                }
            }
        }
    }

    #[test]
    fn test_hnsw_index_simple() {
        let mut index = HNSWIndex::new(0.5, 32, 2, 50, 50, DistanceMetric::Euclidean);

        let (data, query_vectors) = get_simple_vectors();
        for vector in &data {
            index.add_vector_before_build(vector);
        }

        index.build();

        let results = index.search(&query_vectors[0], 2);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
    }

    #[test]
    fn test_hnsw_index_complex() {
        let mut index = HNSWIndex::new(0.5, 32, 16, 200, 200, DistanceMetric::Euclidean);

        let (data, query_vector) = get_complex_vectors();

        for vector in &data {
            index.add_vector_before_build(vector);
        }

        index.build();

        let results = index.search(&query_vector, 2);
        assert!(is_in_actual_result(&data, &query_vector, &results));
    }
}
