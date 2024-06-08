// use std::collections::HashSet;

// use crate::index::neighbor::NeighborNode;

// use super::HNSWIndex;

// impl HNSWIndex {
//     /// Constructs the index for a vector that has already been inserted into the index.
//     /// This function is called when the index is constructed for a single vector.
//     ///
//     /// # Arguments
//     ///
//     /// * `id` - The ID of the vector to construct the index for.
//     pub(super) fn index_vector(&self, id: usize) {
//         let insert_level = self.get_level(id);
//         let mut current_id = self.root_node_id;

//         if id == 0 {
//             return;
//         }

//         if insert_level < self.current_level {
//             let mut current_distance =
//                 self.vectors[current_id].distance(&self.vectors[id], self.metric);
//             let mut current_level = self.current_level;
//             while current_level > insert_level {
//                 let mut changed = true;
//                 while changed {
//                     changed = false;
//                     let current_neighbors =
//                         self.get_neighbor(current_id, current_level).read().unwrap();
//                     for current_neighbor in current_neighbors.iter() {
//                         if *current_neighbor > self.n_items {
//                             eprintln!("Invalid neighbor id: {}", current_neighbor);
//                             return;
//                         }
//                         let neighbor_distance = self.vectors[*current_neighbor]
//                             .distance(&self.vectors[id], self.metric);
//                         if neighbor_distance < current_distance {
//                             current_id = *current_neighbor;
//                             current_distance = neighbor_distance;
//                             changed = true;
//                         }
//                     }
//                 }
//                 current_level -= 1;
//             }
//         }

//         let mut level = if insert_level < self.current_level {
//             insert_level
//         } else {
//             self.current_level
//         };
//         let mut visited = HashSet::new();
//         let mut sorted_candidates = Vec::new();
//         let insert_data = &self.vectors[id];
//         visited.insert(id);
//         sorted_candidates.push(NeighborNode::new(
//             current_id,
//             insert_data.distance(&self.vectors[current_id], self.metric),
//         ));

//         loop {
//             let top_candidates = self.search_layer_with_candidate(
//                 insert_data,
//                 &sorted_candidates,
//                 &mut visited,
//                 level,
//             );

//             sorted_candidates = top_candidates.into_sorted_vec();
//             if sorted_candidates.is_empty() {
//                 eprintln!("sorted candidates is empty");
//                 return;
//             }

//             self.connect_neighbor(id, &sorted_candidates, level, false);
//             if level == 0 {
//                 break;
//             }

//             level -= 1;
//         }
//     }
// }

// #[cfg(test)]
// mod tests {
//     use crate::{
//         data::HighDimVector,
//         index::{DistanceMetric, Index},
//     };

//     use super::*;

//     #[test]
//     fn test_index_vector() {
//         let mut index = HNSWIndex::new(DistanceMetric::Euclidean);
//         let v0 = HighDimVector::new(0, vec![1.0, 2.0, 3.0]);
//         let v1 = HighDimVector::new(1, vec![4.0, 5.0, 6.0]);
//         let v2 = HighDimVector::new(2, vec![7.0, 8.0, 9.0]);

//         index.add_vector(v0);
//         index.add_vector(v1);
//         index.add_vector(v2);

//         index.index_vector(0);
//         index.index_vector(1);
//         index.index_vector(2);

//         assert_eq!(index.layer_to_neighbors[1].len(), index.id_to_level[1]);
//         assert_eq!(index.layer_to_neighbors[2].len(), index.id_to_level[2]);
//     }
// }
