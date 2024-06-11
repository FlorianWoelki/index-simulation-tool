use std::collections::{BinaryHeap, HashMap};

use node::Node;
use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::data::{QueryResult, SparseVector};

use super::neighbor::NeighborNode;

mod node;

pub struct HNSWIndex {
    vectors: Vec<SparseVector>,
    nodes: HashMap<usize, Node>,
    /// Controls the probability distribution for assigning the layer of each node in the graph.
    level_distribution_factor: f32,
    /// Determines the maximum layer that a node can have in the graph.
    max_layers: usize,
    /// Controls the number of neighbors considered during the construction phase.
    ef_construction: usize,
    /// Controls the number of neighbors considered during the search phase.
    ef_search: usize,
    random_seed: u64,
}

impl HNSWIndex {
    pub fn new(
        level_distribution_factor: f32,
        max_layers: usize,
        ef_construction: usize,
        ef_search: usize,
        random_seed: u64,
    ) -> Self {
        HNSWIndex {
            vectors: Vec::new(),
            nodes: HashMap::new(),
            max_layers,
            level_distribution_factor,
            ef_construction,
            ef_search,
            random_seed,
        }
    }

    pub fn add_vector(&mut self, item: &SparseVector) {
        self.vectors.push(item.clone());
    }

    pub fn build(&mut self) {
        for (i, vector) in self.vectors.iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(self.random_seed);
            let mut layer = 0;
            while rng.gen::<f32>() < self.level_distribution_factor.powi(layer as i32)
                && layer < self.max_layers
            {
                layer += 1;
            }

            let new_node = Node {
                id: i,
                connections: vec![Vec::new(); self.max_layers + 1],
                vector: vector.clone(),
                layer,
            };

            self.nodes.insert(i, new_node);
        }

        let nodes: Vec<Node> = self.nodes.values().cloned().collect();
        for node in nodes {
            for layer in (0..=node.layer).rev() {
                self.connect_new_node(&node, layer);
            }
        }
    }

    fn connect_new_node(&mut self, new_node: &Node, layer: usize) {
        let neighbors: BinaryHeap<NeighborNode> = self
            .nodes
            .values()
            .filter(|&node| node.layer >= layer && node.id != new_node.id)
            .map(|node| {
                let distance = new_node.vector.euclidean_distance(&node.vector);
                NeighborNode {
                    id: node.id,
                    distance: OrderedFloat(distance),
                }
            })
            .collect();

        let neighbors_to_add: Vec<NeighborNode> = neighbors
            .into_sorted_vec()
            .into_iter()
            .take(self.ef_construction)
            .collect();

        for neighbor in neighbors_to_add {
            self.nodes.get_mut(&new_node.id).unwrap().connections[layer].push(neighbor.id);
            self.nodes.get_mut(&neighbor.id).unwrap().connections[layer].push(new_node.id);
        }
    }

    pub fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let entry_point = StdRng::seed_from_u64(self.random_seed).gen_range(0..self.nodes.len());
        let mut current_node = &self.nodes[&entry_point];

        for layer in (0..self.max_layers).rev() {
            current_node = self.find_closest_node(current_node, query_vector, layer);
        }

        self.knn_search(query_vector, current_node, k)
    }

    fn find_closest_node<'a>(
        &'a self,
        start_node: &'a Node,
        query_vector: &SparseVector,
        layer: usize,
    ) -> &'a Node {
        let mut current_node = start_node;
        let mut closest_distance = query_vector.euclidean_distance(&current_node.vector);

        loop {
            let mut updated = false;

            for &neighbor_id in &current_node.connections[layer] {
                let neighbor_node = &self.nodes[&neighbor_id];
                let distance = query_vector.euclidean_distance(&neighbor_node.vector);
                if distance < closest_distance {
                    closest_distance = distance;
                    current_node = neighbor_node;
                    updated = true;
                }
            }

            if !updated {
                break;
            }
        }

        current_node
    }

    fn knn_search(
        &self,
        query_vector: &SparseVector,
        entry_node: &Node,
        k: usize,
    ) -> Vec<QueryResult> {
        let mut top_k: BinaryHeap<NeighborNode> = BinaryHeap::new();
        let mut visited: HashMap<usize, bool> = HashMap::new();
        let mut candidates: BinaryHeap<NeighborNode> = BinaryHeap::new();

        let entry_distance = query_vector.euclidean_distance(&entry_node.vector);
        candidates.push(NeighborNode {
            id: entry_node.id,
            distance: OrderedFloat(entry_distance),
        });
        top_k.push(NeighborNode {
            id: entry_node.id,
            distance: OrderedFloat(entry_distance),
        });
        visited.insert(entry_node.id, true);

        while let Some(candidate) = candidates.pop() {
            for layer in (0..=self.max_layers).rev() {
                for &neighbor_id in &self.nodes[&candidate.id].connections[layer] {
                    if visited.contains_key(&neighbor_id) {
                        continue;
                    }

                    let neighbor_node = &self.nodes[&neighbor_id];
                    let distance = query_vector.euclidean_distance(&neighbor_node.vector);
                    if top_k.len() < k || distance < -top_k.peek().unwrap().distance.into_inner() {
                        candidates.push(NeighborNode {
                            id: neighbor_id,
                            distance: OrderedFloat(distance),
                        });
                        top_k.push(NeighborNode {
                            id: neighbor_id,
                            distance: OrderedFloat(distance),
                        });

                        if top_k.len() > k {
                            top_k.pop();
                        }
                    }

                    visited.insert(neighbor_id, true);
                }

                if candidates.len() > self.ef_search {
                    break;
                }
            }
        }

        top_k
            .into_sorted_vec()
            .into_iter()
            .map(|neighbor| QueryResult {
                index: neighbor.id,
                score: neighbor.distance,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;
    use rand::thread_rng;

    use super::*;

    #[test]
    fn test_hnsw_index_simple() {
        let random_seed = 42;
        let mut index = HNSWIndex::new(1.0 / 3.0, 16, 200, 200, random_seed);

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
    fn test_hnsw_index_complex() {
        let random_seed = thread_rng().gen::<u64>();
        let mut index = HNSWIndex::new(1.0 / 3.0, 16, 200, 200, random_seed);

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
