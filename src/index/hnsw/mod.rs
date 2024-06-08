// use std::{collections::BinaryHeap, sync::RwLock};

// use rand::Rng;

// use crate::{
//     data::HighDimVector,
//     index::{DistanceMetric, Index},
// };

// use super::neighbor::NeighborNode;

// mod construction;
// mod search;

// /// A Hierarchical Navigable Small World (HNSW) graph index for efficient similarity search
// /// in high-dimensional spaces. It maintains multiple layers of graphs, with each layer
// /// having progressively fewer nodes.
// pub struct HNSWIndex {
//     pub(super) vectors: Vec<HighDimVector>,
//     pub(super) metric: DistanceMetric,
//     pub(super) n_neighbor: usize,
//     pub(super) n_neighbor0: usize,
//     pub(super) max_layer: usize,
//     pub(super) ef_construction: usize,
//     pub(super) ef_search: usize,
//     pub(super) layer_to_neighbors: Vec<Vec<RwLock<Vec<usize>>>>,
//     pub(super) base_layer_neighbors: Vec<RwLock<Vec<usize>>>,
//     pub(super) root_node_id: usize,
//     pub(super) n_items: usize,
//     pub(super) current_level: usize,
//     pub(super) id_to_level: Vec<usize>,
//     pub(super) n_indexed_vectors: usize,
// }

// impl Index for HNSWIndex {
//     fn new(metric: DistanceMetric) -> Self {
//         HNSWIndex {
//             vectors: Vec::new(),
//             metric,
//             n_neighbor: 32,
//             n_neighbor0: 64,
//             max_layer: 20,
//             ef_construction: 500,
//             ef_search: 16,
//             layer_to_neighbors: Vec::new(),
//             base_layer_neighbors: Vec::new(),
//             id_to_level: Vec::new(),
//             root_node_id: 0,
//             n_items: 0,
//             current_level: 0,
//             n_indexed_vectors: 0,
//         }
//     }

//     fn add_vector(&mut self, vector: HighDimVector) {
//         let vid = vector.id;
//         let mut current_level = self.get_random_level();
//         if vid == 0 {
//             current_level = self.max_layer;
//             self.current_level = current_level;
//             self.root_node_id = vid;
//         }

//         let base_layer_neighbors = RwLock::new(Vec::with_capacity(self.n_neighbor0));
//         let mut neighbors = Vec::with_capacity(current_level);

//         for _ in 0..current_level {
//             neighbors.push(RwLock::new(Vec::with_capacity(self.n_neighbor)));
//         }

//         self.vectors.push(vector);
//         self.base_layer_neighbors.push(base_layer_neighbors);
//         self.layer_to_neighbors.push(neighbors);
//         self.id_to_level.push(current_level);
//         self.n_items += 1;
//     }

//     fn build(&mut self) {
//         (self.n_indexed_vectors..self.n_items).for_each(|insert_id| {
//             self.index_vector(insert_id);
//         });
//         self.n_indexed_vectors = self.n_items;
//     }

//     fn search(&self, query_vector: &HighDimVector, k: usize) -> Vec<HighDimVector> {
//         let mut knn_results = self.search_knn(query_vector, k);
//         let mut results = Vec::with_capacity(k);

//         (0..k).for_each(|_| {
//             if let Some(top) = knn_results.pop() {
//                 results.push(self.vectors[top.id].clone());
//             }
//         });

//         results.reverse();
//         results
//     }
// }

// impl HNSWIndex {
//     /// Retrieves the RwLock-protected neighbor list for a given node at a specific layer.
//     ///
//     /// # Arguments
//     ///
//     /// * `id` - The node identifier.
//     /// * `layer` - The graph layer from which neighbors are retrieved.
//     ///
//     /// # Returns
//     /// A reference to the `RwLock` guarding the neighbor list.
//     pub(super) fn get_neighbor(&self, id: usize, layer: usize) -> &RwLock<Vec<usize>> {
//         if layer == 0 {
//             return &self.base_layer_neighbors[id];
//         }

//         &self.layer_to_neighbors[id][layer - 1]
//     }

//     fn connect_neighbor(
//         &self,
//         current_id: usize,
//         sorted_candidates: &[NeighborNode],
//         level: usize,
//         is_update: bool,
//     ) -> usize {
//         let n_neighbor = if level == 0 {
//             self.n_neighbor0
//         } else {
//             self.n_neighbor
//         };
//         let selected_neighbors = self.get_neighbors_by_heuristic2(sorted_candidates, n_neighbor);
//         if selected_neighbors.len() > n_neighbor {
//             eprintln!(
//                 "selected neighbors is too large: {}",
//                 selected_neighbors.len()
//             );
//             return 0;
//         }

//         if selected_neighbors.is_empty() {
//             eprintln!("selected neighbors is empty");
//             return 0;
//         }

//         let next_closest_entry_point = selected_neighbors[0].id;

//         {
//             let mut current_neigh = self.get_neighbor(current_id, level).write().unwrap();
//             current_neigh.clear();
//             selected_neighbors.iter().for_each(|neighbor| {
//                 current_neigh.push(neighbor.id);
//             });
//         }

//         for selected_neighbor in selected_neighbors.iter() {
//             let mut neighbor_of_selected_neighbors = self
//                 .get_neighbor(selected_neighbor.id, level)
//                 .write()
//                 .unwrap();
//             if neighbor_of_selected_neighbors.len() > n_neighbor {
//                 eprintln!(
//                     "neighbor of selected neighbors is too large: {}",
//                     neighbor_of_selected_neighbors.len()
//                 );
//                 return 0;
//             }

//             if selected_neighbor.id == current_id {
//                 eprintln!("selected neighbor is current id");
//                 return 0;
//             }

//             let mut is_current_id_present = false;

//             if is_update {
//                 for iter in neighbor_of_selected_neighbors.iter() {
//                     if *iter == current_id {
//                         is_current_id_present = true;
//                         break;
//                     }
//                 }
//             }

//             if !is_current_id_present {
//                 if neighbor_of_selected_neighbors.len() < n_neighbor {
//                     neighbor_of_selected_neighbors.push(current_id);
//                 } else {
//                     let d_max = self.vectors[current_id]
//                         .distance(&self.vectors[selected_neighbor.id], self.metric);
//                     let mut candidates = BinaryHeap::new();
//                     candidates.push(NeighborNode::new(current_id, d_max));
//                     for iter in neighbor_of_selected_neighbors.iter() {
//                         let neighbor_id = *iter;
//                         let d_neigh = self.vectors[neighbor_id]
//                             .distance(&self.vectors[selected_neighbor.id], self.metric);
//                         candidates.push(NeighborNode::new(neighbor_id, d_neigh));
//                     }
//                     let return_list =
//                         self.get_neighbors_by_heuristic2(&candidates.into_sorted_vec(), n_neighbor);

//                     neighbor_of_selected_neighbors.clear();
//                     for neighbor_in_list in return_list {
//                         neighbor_of_selected_neighbors.push(neighbor_in_list.id);
//                     }
//                 }
//             }
//         }

//         next_closest_entry_point
//     }

//     /// Selects neighbors based on a heuristic from a sorted list of candidates, ensuring that
//     /// the chosen neighbors are within a specified maximum count.
//     ///
//     /// # Arguments
//     ///
//     /// * `sorted_list` - A sorted list of neighbors.
//     /// * `ret_size` - The maximum number of neighbors to return.
//     ///
//     /// # Returns
//     /// A vector of neighbors selected based on the heuristic.
//     fn get_neighbors_by_heuristic2(
//         &self,
//         sorted_list: &[NeighborNode],
//         maximum_size: usize,
//     ) -> Vec<NeighborNode> {
//         let sorted_list_len = sorted_list.len();
//         let mut return_list = Vec::with_capacity(sorted_list_len);

//         for iter in sorted_list.iter() {
//             if return_list.len() >= maximum_size {
//                 break;
//             }

//             let id = iter.id;
//             let distance = iter.distance;
//             if sorted_list_len < maximum_size {
//                 return_list.push(NeighborNode::new(id, distance.into_inner()));
//                 continue;
//             }

//             let mut good = true;

//             for ret_neighbor in return_list.iter() {
//                 let cur2ret_distance =
//                     self.vectors[id].distance(&self.vectors[ret_neighbor.id], self.metric);
//                 if cur2ret_distance < distance.into_inner() {
//                     good = false;
//                     break;
//                 }
//             }

//             if good {
//                 return_list.push(NeighborNode::new(id, distance.into_inner()));
//             }
//         }

//         return_list
//     }

//     pub(super) fn get_level(&self, id: usize) -> usize {
//         self.id_to_level[id]
//     }

//     pub(super) fn get_random_level(&self) -> usize {
//         let mut rng = rand::thread_rng();
//         let mut result = 0;

//         while result < self.max_layer {
//             if rng.gen_range(0.0..1.0) > 0.5 {
//                 result += 1;
//             } else {
//                 break;
//             }
//         }

//         result
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_adding_vector() {
//         let mut index = HNSWIndex::new(DistanceMetric::Euclidean);
//         let v0 = HighDimVector::new(0, vec![1.0, 2.0, 3.0]);
//         index.add_vector(v0.clone());

//         assert_eq!(index.vectors.len(), 1);
//         assert_eq!(index.vectors[0], v0);
//     }

//     #[test]
//     fn test_search() {
//         let mut index = HNSWIndex::new(DistanceMetric::Euclidean);

//         for i in 0..10 {
//             let v = HighDimVector::new(i, vec![i as f32, (i + 1) as f32, (i + 2) as f32]);
//             index.add_vector(v);
//         }

//         index.build();

//         let query_vector = HighDimVector::new(99999, vec![10.0, 11.0, 12.0]);
//         let result = index.search(&query_vector, 3);

//         assert_eq!(result.len(), 3);
//         assert_eq!(result[0].id, 9);
//         assert_eq!(result[1].id, 8);
//         assert_eq!(result[2].id, 7);
//     }
// }
