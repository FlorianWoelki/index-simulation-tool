use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    sync::{Arc, Mutex},
};

use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, seq::IteratorRandom, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::data::{QueryResult, SparseVector};

use super::{DistanceMetric, IndexIdentifier, SparseIndex};

#[derive(Serialize, Deserialize)]
pub struct NSWIndex {
    vectors: Vec<SparseVector>,
    graph: HashMap<usize, HashSet<usize>>,
    /// Number of nearest neighbors to consider during index construction.
    /// Higher values improve recall but increase build time and memory usage.
    ef_construction: usize,
    /// Number of nearest neighbors to consider during search.
    /// Higher values improve recall but increase search time.
    ef_search: usize,
    metric: DistanceMetric,
    random_seed: u64,
}

impl NSWIndex {
    pub fn new(
        ef_construction: usize,
        ef_search: usize,
        metric: DistanceMetric,
        random_seed: u64,
    ) -> Self {
        NSWIndex {
            vectors: Vec::new(),
            graph: HashMap::new(),
            ef_construction,
            ef_search,
            metric,
            random_seed,
        }
    }

    fn knn_search(
        &self,
        query: &SparseVector,
        m: usize,
        k: usize,
        graph: &HashMap<usize, HashSet<usize>>,
    ) -> Vec<usize> {
        let result = Arc::new(Mutex::new(HashSet::new()));
        let candidates = Arc::new(Mutex::new(HashSet::new()));
        let visited_set = Arc::new(Mutex::new(HashSet::new()));

        (0..m).into_par_iter().for_each(|_| {
            if let Some(entry_point) = self.get_random_entry_point(graph) {
                candidates.lock().unwrap().insert(entry_point);
            }

            loop {
                let c = {
                    let candidates_guard = candidates.lock().unwrap();
                    if candidates_guard.is_empty() {
                        break;
                    }
                    *candidates_guard
                        .iter()
                        .min_by(|&&x, &&y| {
                            query
                                .distance(&self.vectors[x], &self.metric)
                                .partial_cmp(&query.distance(&self.vectors[y], &self.metric))
                                .unwrap()
                        })
                        .unwrap()
                };

                candidates.lock().unwrap().remove(&c);

                if !visited_set.lock().unwrap().insert(c) {
                    continue;
                }

                {
                    let mut result_guard = result.lock().unwrap();
                    result_guard.insert(c);

                    if result_guard.len() > k {
                        let d1 = query.distance(&self.vectors[c], &self.metric);
                        let d2 = result_guard
                            .iter()
                            .map(|&x| query.distance(&self.vectors[x], &self.metric))
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap();

                        if d1 > d2 {
                            result_guard.remove(&c);
                        }
                    }
                }

                if let Some(neighbors) = self.graph.get(&c) {
                    let mut candidates_guard = candidates.lock().unwrap();
                    let visited_set_guard = visited_set.lock().unwrap();
                    for &e in neighbors {
                        if !visited_set_guard.contains(&e) {
                            candidates_guard.insert(e);
                        }
                    }
                }
            }
        });

        let mut result_vec: Vec<usize> = result.lock().unwrap().iter().cloned().collect();
        result_vec.sort_by(|&x, &y| {
            query
                .distance(&self.vectors[x], &self.metric)
                .partial_cmp(&query.distance(&self.vectors[y], &self.metric))
                .unwrap()
        });
        result_vec.truncate(k);
        result_vec
    }

    fn get_random_entry_point(&self, graph: &HashMap<usize, HashSet<usize>>) -> Option<usize> {
        let mut rng = StdRng::seed_from_u64(self.random_seed);
        graph.keys().choose(&mut rng).cloned()
    }
}

impl SparseIndex for NSWIndex {
    fn add_vector_before_build(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
    }

    fn add_vector(&mut self, vector: &SparseVector) {
        let vector_index = self.vectors.len();
        self.vectors.push(vector.clone());

        let new_node_neighbors = if self.graph.is_empty() {
            HashSet::new()
        } else {
            let neighbors =
                self.knn_search(vector, self.ef_construction, self.ef_search, &self.graph);
            neighbors.iter().cloned().collect()
        };

        self.graph.insert(vector_index, new_node_neighbors.clone());

        for &neighbor in &new_node_neighbors {
            self.graph.get_mut(&neighbor).unwrap().insert(vector_index);
        }
    }

    fn remove_vector(&mut self, id: usize) -> Option<SparseVector> {
        if id >= self.vectors.len() {
            return None;
        }

        let removed_vector = self.vectors.remove(id);
        self.graph.remove(&id);

        // Updates the graph: remove references to the deleted id and shift ids
        let mut new_graph = HashMap::new();
        for (&k, v) in self.graph.iter() {
            let mut new_set = HashSet::new();
            for &neighbor in v {
                if neighbor != id {
                    if neighbor > id {
                        new_set.insert(neighbor - 1);
                    } else {
                        new_set.insert(neighbor);
                    }
                }
            }

            let new_key = if k > id { k - 1 } else { k };
            new_graph.insert(new_key, new_set);
        }
        self.graph = new_graph;

        Some(removed_vector)
    }

    fn build(&mut self) {
        let graph = Arc::new(Mutex::new(HashMap::new()));
        let vectors = Arc::new(self.vectors.clone());

        (0..vectors.len()).into_iter().for_each(|i| {
            let neighbors = if i == 0 {
                HashSet::new()
            } else {
                let current_graph = graph.lock().unwrap().clone();
                self.knn_search(
                    &vectors[i],
                    self.ef_construction,
                    self.ef_search,
                    &current_graph,
                )
                .into_iter()
                .collect()
            };

            let mut current_graph = graph.lock().unwrap();
            current_graph.insert(i, neighbors.clone());
            for &neighbor in &neighbors {
                current_graph.entry(neighbor).or_default().insert(i);
            }
        });

        self.graph = Arc::try_unwrap(graph).unwrap().into_inner().unwrap();
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let nearest_neighbors = self.knn_search(query_vector, self.graph.len(), k, &self.graph);
        nearest_neighbors
            .into_par_iter()
            .map(|idx| QueryResult {
                index: idx,
                score: OrderedFloat(query_vector.distance(&self.vectors[idx], &self.metric)),
            })
            .collect()
    }

    fn save(&self, file: &mut File) {
        let mut writer = BufWriter::new(file);
        let index_type = IndexIdentifier::NSW.to_u32();
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
        let random_seed = 42;
        let mut index = NSWIndex::new(5, 3, DistanceMetric::Euclidean, random_seed);
        for vector in &data {
            index.add_vector_before_build(vector);
        }

        let bytes = bincode::serialize(&index).unwrap();
        let reconstructed: NSWIndex = bincode::deserialize(&bytes).unwrap();

        assert_eq!(index.vectors, reconstructed.vectors);
        assert_eq!(index.metric, reconstructed.metric);
        assert_eq!(index.graph, reconstructed.graph);
        assert_eq!(index.ef_construction, reconstructed.ef_construction);
        assert_eq!(index.ef_search, reconstructed.ef_search);
        assert_eq!(index.random_seed, reconstructed.random_seed);
    }

    #[test]
    fn test_add_vector() {
        let random_seed = 42;
        let mut index = NSWIndex::new(5, 3, DistanceMetric::Euclidean, random_seed);

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
        let mut index = NSWIndex::new(5, 3, DistanceMetric::Cosine, 42);

        for i in 0..5 {
            let vector = SparseVector {
                indices: vec![i],
                values: vec![OrderedFloat(1.0)],
            };
            index.add_vector_before_build(&vector);
        }

        index.build();

        assert_eq!(index.vectors.len(), 5);
        assert_eq!(index.graph.len(), 5);

        let result = index.remove_vector(2);
        assert_eq!(
            result,
            Some(SparseVector {
                indices: vec![2],
                values: vec![OrderedFloat(1.0)]
            })
        );
        assert_eq!(index.vectors.len(), 4);
        assert_eq!(index.graph.len(), 4);

        assert!(index.graph.contains_key(&0));
        assert!(index.graph.contains_key(&1));
        assert!(index.graph.contains_key(&2)); // This was previously 3
        assert!(index.graph.contains_key(&3)); // This was previously 4
        assert!(!index.graph.contains_key(&4));

        for neighbors in index.graph.values() {
            assert!(!neighbors.contains(&4)); // Shifted out of bounds
        }

        let result = index.remove_vector(10);
        assert!(result.is_none());
    }

    #[test]
    fn test_remove_vector_out_of_bounds() {
        let mut index = NSWIndex::new(5, 3, DistanceMetric::Cosine, 42);

        for i in 0..5 {
            let vector = SparseVector {
                indices: vec![i],
                values: vec![OrderedFloat(1.0)],
            };
            index.add_vector_before_build(&vector);
        }

        index.build();

        let result = index.remove_vector(10);
        assert!(result.is_none());
        assert_eq!(index.vectors.len(), 5);
        assert_eq!(index.graph.len(), 5);
    }

    #[test]
    fn test_remove_vector_last() {
        let mut index = NSWIndex::new(5, 3, DistanceMetric::Cosine, 42);

        for i in 0..5 {
            let vector = SparseVector {
                indices: vec![i],
                values: vec![OrderedFloat(1.0)],
            };
            index.add_vector_before_build(&vector);
        }

        index.build();

        assert_eq!(index.vectors.len(), 5);
        assert_eq!(index.graph.len(), 5);

        let result = index.remove_vector(3);
        assert_eq!(
            result,
            Some(SparseVector {
                indices: vec![3],
                values: vec![OrderedFloat(1.0)]
            })
        );
        assert_eq!(index.vectors.len(), 4);
        assert_eq!(index.graph.len(), 4);

        // Verify that no neighbors reference the removed index
        for neighbors in index.graph.values() {
            assert!(!neighbors.contains(&4));
        }
    }

    #[test]
    fn test_nsw_index_simple() {
        let mut index = NSWIndex::new(10, 5, DistanceMetric::Euclidean, 42);

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
        let mut index = NSWIndex::new(200, 200, DistanceMetric::Euclidean, 42);

        let (data, query_vector) = get_complex_vectors();

        for vector in &data {
            index.add_vector_before_build(vector);
        }

        index.build();

        let results = index.search(&query_vector, 10);
        assert!(is_in_actual_result(&data, &query_vector, &results));
    }
}
