use std::collections::{HashMap, HashSet};

use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, seq::IteratorRandom, SeedableRng};

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

    pub fn build(&mut self) {
        for (i, vector) in self.vectors.iter().enumerate() {
            if i == 0 {
                self.graph.insert(i, HashSet::new());
                continue;
            }

            let neighbors = self.knn_search(vector, self.ef_construction, self.ef_search);
            self.graph.insert(i, neighbors.iter().cloned().collect());
            for &neighbor in &neighbors {
                self.graph.get_mut(&neighbor).unwrap().insert(i);
            }
        }
    }

    fn knn_search(&self, query: &SparseVector, m: usize, k: usize) -> Vec<usize> {
        let mut result = HashSet::new();
        let mut candidates = HashSet::new();
        let mut visited_set = HashSet::new();

        for _ in 0..m {
            if let Some(entry_point) = self.get_random_entry_point() {
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

    fn get_random_entry_point(&self) -> Option<usize> {
        let mut rng = StdRng::seed_from_u64(self.random_seed);
        self.graph.keys().choose(&mut rng).cloned()
    }

    pub fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let nearest_neighbors = self.knn_search(query_vector, self.graph.len(), k);
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

    use super::*;

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

        assert!(false);
    }
}
