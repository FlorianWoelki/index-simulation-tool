use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    sync::{Arc, Mutex},
};

use node::Node;
use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::data::{QueryResult, SparseVector};

use super::{neighbor::NeighborNode, DistanceMetric, SparseIndex};

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
    metric: DistanceMetric,
    random_seed: u64,
}

impl HNSWIndex {
    pub fn new(
        level_distribution_factor: f32,
        max_layers: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: DistanceMetric,
        random_seed: u64,
    ) -> Self {
        HNSWIndex {
            vectors: Vec::new(),
            nodes: HashMap::new(),
            max_layers,
            level_distribution_factor,
            ef_construction,
            ef_search,
            metric,
            random_seed,
        }
    }

    fn connect_new_node(&mut self, new_node: &Node, layer: usize) {
        let neighbors: BinaryHeap<NeighborNode> = self
            .nodes
            .values()
            .filter(|&node| node.layer >= layer && node.id != new_node.id)
            .map(|node| {
                let distance = new_node.vector.distance(&node.vector, &self.metric);
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

    fn knn_search_parallel(
        &self,
        query_vector: &SparseVector,
        entry_node: &Node,
        k: usize,
    ) -> Vec<QueryResult> {
        let top_k = Arc::new(Mutex::new(BinaryHeap::new()));
        let visited = Arc::new(Mutex::new(HashMap::new()));
        let candidates = Arc::new(Mutex::new(BinaryHeap::new()));

        let entry_distance = query_vector.distance(&entry_node.vector, &self.metric);
        candidates.lock().unwrap().push(NeighborNode {
            id: entry_node.id,
            distance: OrderedFloat(entry_distance),
        });
        top_k.lock().unwrap().push(NeighborNode {
            id: entry_node.id,
            distance: OrderedFloat(entry_distance),
        });
        visited.lock().unwrap().insert(entry_node.id, true);

        while !candidates.lock().unwrap().is_empty() {
            let candidate_batch: Vec<_> = (0..self.ef_search)
                .filter_map(|_| candidates.lock().unwrap().pop())
                .collect();

            candidate_batch.into_par_iter().for_each(|candidate| {
                for layer in (0..=self.max_layers).rev() {
                    for &neighbor_id in &self.nodes[&candidate.id].connections[layer] {
                        let mut visited_guard = visited.lock().unwrap();
                        if visited_guard.contains_key(&neighbor_id) {
                            continue;
                        }
                        visited_guard.insert(neighbor_id, true);
                        drop(visited_guard);

                        let neighbor_node = &self.nodes[&neighbor_id];
                        let distance = query_vector.distance(&neighbor_node.vector, &self.metric);

                        let mut top_k_guard = top_k.lock().unwrap();
                        if top_k_guard.len() < k
                            || distance < -top_k_guard.peek().unwrap().distance.into_inner()
                        {
                            candidates.lock().unwrap().push(NeighborNode {
                                id: neighbor_id,
                                distance: OrderedFloat(distance),
                            });
                            top_k_guard.push(NeighborNode {
                                id: neighbor_id,
                                distance: OrderedFloat(distance),
                            });
                            if top_k_guard.len() > k {
                                top_k_guard.pop();
                            }
                        }
                    }
                }
            });
        }

        Arc::try_unwrap(top_k)
            .unwrap()
            .into_inner()
            .unwrap()
            .into_sorted_vec()
            .into_iter()
            .map(|neighbor| QueryResult {
                index: neighbor.id,
                score: neighbor.distance,
            })
            .collect()
    }

    fn find_closest_node<'a>(
        &'a self,
        start_node: &'a Node,
        query_vector: &SparseVector,
        layer: usize,
    ) -> &'a Node {
        let mut current_node = start_node;
        let mut closest_distance = query_vector.distance(&current_node.vector, &self.metric);

        loop {
            let mut updated = false;

            for &neighbor_id in &current_node.connections[layer] {
                let neighbor_node = &self.nodes[&neighbor_id];
                let distance = query_vector.distance(&neighbor_node.vector, &self.metric);
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

        let entry_distance = query_vector.distance(&entry_node.vector, &self.metric);
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
                    let distance = query_vector.distance(&neighbor_node.vector, &self.metric);
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

impl SparseIndex for HNSWIndex {
    fn add_vector(&mut self, item: &SparseVector) {
        self.vectors.push(item.clone());
    }

    fn remove_vector(&mut self, index: usize) -> Option<SparseVector> {
        if index >= self.vectors.len() {
            return None;
        }

        let removed_vector = self.vectors.swap_remove(index);

        if let Some(removed_node) = self.nodes.remove(&index) {
            let last_index = self.vectors.len();
            let mut connections_to_update: Vec<(usize, usize, HashSet<usize>)> = Vec::new();

            for layer in 0..=removed_node.layer {
                let mut layer_connections = HashSet::new();
                layer_connections.extend(removed_node.connections[layer].iter().cloned());

                if index != last_index {
                    if let Some(last_node) = self.nodes.get(&last_index) {
                        for neighbor_id in &last_node.connections[layer] {
                            layer_connections.insert(*neighbor_id);
                        }
                    }
                }

                connections_to_update.push((layer, index, layer_connections));
            }

            for (layer, removed_index, connections) in connections_to_update {
                for &neighbor_id in &connections {
                    if let Some(neighbor_node) = self.nodes.get_mut(&neighbor_id) {
                        neighbor_node.connections[layer]
                            .retain(|&id| id != removed_index && id != last_index);
                        if index != last_index && neighbor_id != last_index {
                            neighbor_node.connections[layer].push(removed_index);
                        }
                    }
                }
            }

            if index != last_index {
                if let Some(swapped_node) = self.nodes.get_mut(&last_index) {
                    swapped_node.id = index;
                }

                if let Some(swapped_node) = self.nodes.remove(&last_index) {
                    self.nodes.insert(index, swapped_node);
                }
            }

            Some(removed_vector)
        } else {
            None
        }
    }

    fn build(&mut self) {
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

    fn build_parallel(&mut self) {
        let nodes = Arc::new(Mutex::new(HashMap::new()));
        let vectors = Arc::new(self.vectors.clone());
        let max_layers = self.max_layers;
        let level_distribution_factor = self.level_distribution_factor;
        let random_seed = self.random_seed;

        (0..self.vectors.len()).into_par_iter().for_each(|i| {
            let mut rng = StdRng::seed_from_u64(random_seed);
            let mut layer = 0;
            while rng.gen::<f32>() < level_distribution_factor.powi(layer as i32)
                && layer < max_layers
            {
                layer += 1;
            }

            let new_node = Node {
                id: i,
                connections: vec![Vec::new(); max_layers + 1],
                vector: vectors[i].clone(),
                layer,
            };

            let mut nodes_guard = nodes.lock().unwrap();
            nodes_guard.insert(i, new_node);
        });

        self.nodes = Arc::try_unwrap(nodes).unwrap().into_inner().unwrap();
        let node_values: Vec<Node> = self.nodes.values().cloned().collect();
        // TODO: Parallelize this.
        for node in node_values {
            for layer in (0..=node.layer).rev() {
                self.connect_new_node(&node, layer);
            }
        }
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let entry_point = StdRng::seed_from_u64(self.random_seed).gen_range(0..self.nodes.len());
        let mut current_node = &self.nodes[&entry_point];

        for layer in (0..self.max_layers).rev() {
            current_node = self.find_closest_node(current_node, query_vector, layer);
        }

        self.knn_search(query_vector, current_node, k)
    }

    fn search_parallel(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let entry_point = StdRng::seed_from_u64(self.random_seed).gen_range(0..self.nodes.len());
        let mut current_node = &self.nodes[&entry_point];

        for layer in (0..self.max_layers).rev() {
            current_node = self.find_closest_node(current_node, query_vector, layer);
        }

        self.knn_search_parallel(query_vector, current_node, k)
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;
    use rand::thread_rng;

    use crate::test_utils::{get_complex_vectors, get_simple_vectors, is_in_actual_result};

    use super::*;

    #[test]
    fn test_search_parallel() {
        let random_seed = 42;
        let mut index = HNSWIndex::new(
            1.0 / 3.0,
            16,
            200,
            200,
            DistanceMetric::Euclidean,
            random_seed,
        );

        let (data, query_vectors) = get_simple_vectors();

        for vector in &data {
            index.add_vector(vector);
        }

        index.build();

        let results = index.search_parallel(&query_vectors[0], 2);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
    }

    #[test]
    fn test_build_parallel() {
        let random_seed = 42;
        let mut index = HNSWIndex::new(
            1.0 / 3.0,
            16,
            200,
            200,
            DistanceMetric::Euclidean,
            random_seed,
        );

        let (data, query_vectors) = get_simple_vectors();

        for vector in &data {
            index.add_vector(vector);
        }

        index.build_parallel();

        let results = index.search(&query_vectors[0], 2);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
    }

    #[test]
    fn test_remove_vector() {
        let mut index = HNSWIndex::new(0.5, 3, 10, 10, DistanceMetric::Cosine, 42);

        let mut vectors = vec![];
        for i in 0..5 {
            let vector = SparseVector {
                indices: vec![i],
                values: vec![OrderedFloat(1.0)],
            };
            vectors.push(vector);
        }

        for vector in &vectors {
            index.add_vector(vector);
        }

        index.build();

        let removed = index.remove_vector(2);

        assert_eq!(removed, Some(vectors[2].clone()));
        assert_eq!(index.vectors.len(), 4);
        assert_eq!(index.nodes.len(), 4);

        assert_eq!(index.vectors[0], vectors[0].clone());
        assert_eq!(index.vectors[1], vectors[1].clone());
        assert_eq!(index.vectors[2], vectors[4].clone());
        assert_eq!(index.vectors[3], vectors[3].clone());

        // Check if connections are updated correctly.
        for (_, node) in &index.nodes {
            for layer in 0..=node.layer {
                for &neighbor_id in &node.connections[layer] {
                    assert!(neighbor_id < 4);
                    assert!(index.nodes.contains_key(&neighbor_id));
                }
            }
        }
    }

    #[test]
    fn test_hnsw_index_simple() {
        let random_seed = 42;
        let mut index = HNSWIndex::new(0.5, 16, 50, 50, DistanceMetric::Euclidean, random_seed);

        let (data, query_vectors) = get_simple_vectors();
        for vector in &data {
            index.add_vector(vector);
        }

        index.build();

        let results = index.search(&query_vectors[0], 2);
        assert!(is_in_actual_result(&data, &query_vectors[0], &results));
    }

    #[test]
    fn test_hnsw_index_complex() {
        let random_seed = thread_rng().gen::<u64>();
        let mut index = HNSWIndex::new(
            1.0 / 3.0,
            16,
            200,
            200,
            DistanceMetric::Euclidean,
            random_seed,
        );

        let (data, query_vector) = get_complex_vectors();

        for vector in &data {
            index.add_vector(vector);
        }

        index.build();

        let results = index.search(&query_vector, 2);
        assert!(is_in_actual_result(&data, &query_vector, &results));
    }
}
