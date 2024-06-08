use std::{collections::HashSet, f32::consts::PI};

use crate::{
    data::{QueryResult, SparseVector},
    kmeans::kmeans,
};

use super::{neighbor::NeighborNode, SparseIndex};

mod construction;
mod prune;
mod search;

pub struct SSGIndex {
    pub(super) vectors: Vec<SparseVector>,
    pub(super) threshold: f32,
    pub(super) index_size: usize,
    pub(super) graph: Vec<Vec<usize>>,
    pub(super) num_clusters: usize,
    pub(super) root_nodes: Vec<usize>,
    pub(super) neighbor_neighbor_size: usize,
    pub(super) init_k: usize,
    pub(super) random_seed: u64,
}

impl SSGIndex {
    fn new(random_seed: u64, num_clusters: usize) -> Self {
        SSGIndex {
            vectors: Vec::new(),
            threshold: (30.0 / 180.0 * PI).cos(),
            index_size: 100,
            init_k: 10,
            graph: Vec::new(),
            neighbor_neighbor_size: 100,
            root_nodes: Vec::new(),
            num_clusters,
            random_seed,
        }
    }

    fn add_vector(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
    }

    fn build(&mut self) {
        self.construct_knn_graph(self.init_k);

        let len = self.vectors.len() * self.index_size;
        let mut pruned_graph = (0..len)
            .map(|i| NeighborNode::new(i, 0.0))
            .collect::<Vec<NeighborNode>>();
        self.link_each_nodes(&mut pruned_graph);

        (0..self.vectors.len()).enumerate().for_each(|(i, _)| {
            let offset = i * self.index_size;
            let pool_size = (0..self.index_size)
                .take_while(|j| pruned_graph[offset + j].distance.into_inner() == f32::MAX)
                .count()
                .max(1);
            self.graph[i] = (0..pool_size)
                .map(|j| pruned_graph[offset + j].id)
                .collect();
        });

        self.root_nodes = self.kmeans_index(256);
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<SparseVector> {
        self.search_bfs(query_vector, k)
    }
}

impl SSGIndex {
    fn kmeans_index(&self, epoch: usize) -> Vec<usize> {
        let centroids = kmeans(
            self.vectors.clone(),
            self.num_clusters,
            epoch,
            0.01,
            self.random_seed,
        );

        // Find the closest node to each cluster center.
        let closest_node_indices = centroids.iter().map(|center| {
            let mut closest_index = 0;
            let mut closest_distance = f32::MAX;

            self.vectors.iter().enumerate().for_each(|(i, node)| {
                let distance = node.euclidean_distance(center);
                if distance < closest_distance {
                    closest_index = i;
                    closest_distance = distance;
                }
            });

            closest_index
        });

        closest_node_indices.collect()
    }

    fn populate_expanded_neighbors(
        &self,
        query_point: usize,
        expand_neighbors: &mut Vec<NeighborNode>,
    ) {
        let mut visited = HashSet::with_capacity(self.neighbor_neighbor_size);
        visited.insert(query_point);

        for neighbor_id in self.graph[query_point].iter() {
            if *neighbor_id == query_point {
                continue;
            }

            self.graph[*neighbor_id]
                .iter()
                .filter(|node| **node != query_point && *neighbor_id != **node)
                .for_each(|second_neighbor_id| {
                    if visited.insert(*second_neighbor_id) {
                        let distance = self.vectors[query_point]
                            .euclidean_distance(&self.vectors[*second_neighbor_id]);
                        expand_neighbors.push(NeighborNode::new(*second_neighbor_id, distance));

                        /*if expand_neighbors.len() >= self.neighbor_neighbor_size {
                        return;
                        }*/
                    }
                });
        }
    }

    fn link_each_nodes(&mut self, pruned_graph: &mut [NeighborNode]) {
        let mut expanded_neighbors = Vec::new();

        for i in 0..self.vectors.len() {
            expanded_neighbors.clear();
            self.populate_expanded_neighbors(i, &mut expanded_neighbors);
            self.prune_graph(i, &mut expanded_neighbors, pruned_graph);
        }

        for i in 0..self.vectors.len() {
            self.interconnect_pruned_neighbors(i, self.index_size, pruned_graph);
        }
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use super::*;

    #[test]
    fn test_ssg_index() {
        let random_seed = 42;
        let num_clusters = 5;
        let mut index = SSGIndex::new(random_seed, num_clusters);

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
        assert!(false);
    }
}
