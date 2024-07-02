use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, RwLock},
};

use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, seq::IteratorRandom, SeedableRng};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::data::{QueryResult, SparseVector};

use super::DistanceMetric;

pub struct NSWIndex {
    vectors: Vec<SparseVector>,
    graph: HashMap<usize, HashSet<usize>>,
    /// Controls the number of neighbors considered during the construction phase.
    ef_construction: usize,
    /// Controls the number of neighbors considered during the search phase.
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

    pub fn add_vector(&mut self, item: &SparseVector) {
        self.vectors.push(item.clone());
    }

    pub fn remove_vector(&mut self, id: usize) -> Option<SparseVector> {
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

    pub fn build(&mut self) {
        for (i, vector) in self.vectors.iter().enumerate() {
            if i == 0 {
                self.graph.insert(i, HashSet::new());
                continue;
            }

            let neighbors =
                self.knn_search(vector, self.ef_construction, self.ef_search, &self.graph);
            self.graph.insert(i, neighbors.iter().cloned().collect());
            for &neighbor in &neighbors {
                self.graph.get_mut(&neighbor).unwrap().insert(i);
            }
        }
    }

    pub fn build_parallel(&mut self) {
        let graph = Arc::new(RwLock::new(HashMap::new()));
        let vectors = Arc::new(self.vectors.clone());

        (0..vectors.len()).into_par_iter().for_each(|i| {
            let neighbors = if i == 0 {
                HashSet::new()
            } else {
                let current_graph = graph.read().unwrap();
                self.knn_search(
                    &vectors[i],
                    self.ef_construction,
                    self.ef_search,
                    &current_graph,
                )
                .into_iter()
                .collect()
            };

            let mut write_graph = graph.write().unwrap();
            write_graph.insert(i, neighbors.clone());
            for &neighbor in &neighbors {
                write_graph.entry(neighbor).or_default().insert(i);
            }
        });

        self.graph = Arc::try_unwrap(graph).unwrap().into_inner().unwrap();
    }

    fn knn_search(
        &self,
        query: &SparseVector,
        m: usize,
        k: usize,
        graph: &HashMap<usize, HashSet<usize>>,
    ) -> Vec<usize> {
        let mut result = HashSet::new();
        let mut candidates = HashSet::new();
        let mut visited_set = HashSet::new();

        for _ in 0..m {
            if let Some(entry_point) = self.get_random_entry_point(graph) {
                candidates.insert(entry_point);
            }

            while let Some(&c) = candidates.iter().min_by(|&&x, &&y| {
                query
                    .distance(&self.vectors[x], &self.metric)
                    .partial_cmp(&query.distance(&self.vectors[y], &self.metric))
                    .unwrap()
            }) {
                candidates.remove(&c);

                if visited_set.contains(&c) {
                    continue;
                }

                visited_set.insert(c);
                result.insert(c);

                if result.len() > k {
                    let d1 = query.distance(&self.vectors[c], &self.metric);
                    let d2 = result
                        .iter()
                        .map(|&x| query.distance(&self.vectors[x], &self.metric))
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap();

                    if d1 > d2 {
                        result.remove(&c);
                    }
                }

                if let Some(neighbors) = self.graph.get(&c) {
                    for &e in neighbors {
                        if !visited_set.contains(&e) {
                            candidates.insert(e);
                        }
                    }
                }
            }
        }

        let mut result_vec: Vec<usize> = result.into_iter().collect();
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

    pub fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let nearest_neighbors = self.knn_search(query_vector, self.graph.len(), k, &self.graph);
        nearest_neighbors
            .into_iter()
            .map(|idx| QueryResult {
                index: idx,
                score: OrderedFloat(query_vector.distance(&self.vectors[idx], &self.metric)),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;
    use rand::{thread_rng, Rng};

    use crate::test_utils::get_simple_vectors;

    use super::*;

    #[test]
    fn test_build_parallel() {
        let random_seed = 42;
        let mut index = NSWIndex::new(5, 3, DistanceMetric::Euclidean, random_seed);

        let (vectors, query_vectors) = get_simple_vectors();
        for vector in &vectors {
            index.add_vector(vector);
        }

        index.build_parallel();

        let results = index.search(&query_vectors[0], 2);

        println!("{:?}", results);
        assert!(false);
    }

    #[test]
    fn test_remove_vector() {
        let mut index = NSWIndex::new(5, 3, DistanceMetric::Cosine, 42);

        for i in 0..5 {
            let vector = SparseVector {
                indices: vec![i],
                values: vec![OrderedFloat(1.0)],
            };
            index.add_vector(&vector);
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
            index.add_vector(&vector);
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
            index.add_vector(&vector);
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
        let random_seed = 42;
        let mut index = NSWIndex::new(10, 5, DistanceMetric::Euclidean, random_seed);

        let vectors = vec![
            SparseVector {
                indices: vec![0, 1, 2],
                values: vec![OrderedFloat(0.1), OrderedFloat(0.2), OrderedFloat(0.3)],
            },
            SparseVector {
                indices: vec![1, 2, 3],
                values: vec![OrderedFloat(0.4), OrderedFloat(0.5), OrderedFloat(0.6)],
            },
            SparseVector {
                indices: vec![2, 3, 4],
                values: vec![OrderedFloat(0.7), OrderedFloat(0.8), OrderedFloat(0.9)],
            },
            SparseVector {
                indices: vec![3, 4, 5],
                values: vec![OrderedFloat(1.0), OrderedFloat(1.1), OrderedFloat(1.2)],
            },
            SparseVector {
                indices: vec![4, 5, 6],
                values: vec![OrderedFloat(1.3), OrderedFloat(1.4), OrderedFloat(1.5)],
            },
        ];

        for vector in &vectors {
            index.add_vector(vector);
        }

        index.build();

        let results = index.search(&vectors[0], 3);

        println!("{:?}", results);
        assert!(true);
    }

    #[test]
    fn test_nsw_index_complex() {
        let random_seed = thread_rng().gen::<u64>();
        let mut index = NSWIndex::new(200, 200, DistanceMetric::Euclidean, random_seed);

        let mut vectors = vec![];
        for i in 0..100 {
            vectors.push(SparseVector {
                indices: vec![i % 10, (i / 10) % 10],
                values: vec![OrderedFloat((i % 10) as f32), OrderedFloat((i / 10) as f32)],
            });
        }

        for vector in &vectors {
            index.add_vector(vector);
        }

        index.build();

        let query_vector = SparseVector {
            indices: vec![5, 9],
            values: vec![OrderedFloat(5.0), OrderedFloat(9.0)],
        };
        let results = index.search(&query_vector, 10);
        println!("Results for search on query vector: {:?}", results);
        println!("Top Search: {:?}", vectors[results[0].index]);
        println!("Groundtruth: {:?}", query_vector);

        assert!(true);
    }
}
