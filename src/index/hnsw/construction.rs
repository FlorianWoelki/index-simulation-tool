use std::collections::HashSet;

use crate::index::neighbor::NeighborNode;

use super::HNSWIndex;

impl HNSWIndex {
    pub(super) fn index_vector(&self, id: usize) {
        let insert_level = self.get_level(id);
        let mut current_id = self.root_node_id;

        if id == 0 {
            return;
        }

        if insert_level < self.current_level {
            let mut current_distance =
                self.vectors[current_id].euclidean_distance(&self.vectors[id]);
            let mut current_level = self.current_level;
            while current_level > insert_level {
                let mut changed = true;
                while changed {
                    changed = false;
                    let current_neighbors =
                        self.get_neighbor(current_id, current_level).read().unwrap();
                    for current_neighbor in current_neighbors.iter() {
                        if *current_neighbor > self.n_items {
                            eprintln!("Invalid neighbor id: {}", current_neighbor);
                            return;
                        }
                        let neighbor_distance =
                            self.vectors[*current_neighbor].euclidean_distance(&self.vectors[id]);
                        if neighbor_distance < current_distance {
                            current_id = *current_neighbor;
                            current_distance = neighbor_distance;
                            changed = true;
                        }
                    }
                }
                current_level -= 1;
            }
        }

        let mut level = if insert_level < self.current_level {
            insert_level
        } else {
            self.current_level
        };
        let mut visited = HashSet::new();
        let mut sorted_candidates = Vec::new();
        let insert_data = &self.vectors[id];
        visited.insert(id);
        sorted_candidates.push(NeighborNode::new(
            current_id,
            insert_data.euclidean_distance(&self.vectors[current_id]),
        ));

        loop {
            let top_candidates = self.search_layer_with_candidate(
                insert_data,
                &sorted_candidates,
                &mut visited,
                level,
            );

            sorted_candidates = top_candidates.into_sorted_vec();
            if sorted_candidates.is_empty() {
                eprintln!("sorted candidates is empty");
                return;
            }

            self.connect_neighbor(id, &sorted_candidates, level, false);
            if level == 0 {
                break;
            }

            level -= 1;
        }
    }
}
