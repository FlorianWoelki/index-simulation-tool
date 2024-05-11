use std::collections::BinaryHeap;

use crate::index::neighbor::NeighborNode;

use super::SSGIndex;

impl SSGIndex {
    /// Constructs the k-nearest neighbor graph for the vectors where the graph is represented as
    /// an adjacency list.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of nearest neighbors to consider.
    pub(super) fn construct_knn_graph(&mut self, k: usize) {
        let mut neighbor_graph = vec![Vec::new(); self.vectors.len()];
        // TODO: Consider parallel processing.
        for (i, current_vector) in self.vectors.iter().enumerate() {
            let mut neighbor_heap = BinaryHeap::with_capacity(k + 1); // Extra capacity for efficiency.

            for (j, node) in self.vectors.iter().enumerate() {
                if i == j {
                    continue;
                }

                let distance = current_vector.distance(node, self.metric);
                neighbor_heap.push(NeighborNode::new(j, distance));

                // Ensures the heap does not grow beyond k elements.
                if neighbor_heap.len() > k {
                    neighbor_heap.pop();
                }
            }

            // Collects the k-nearest neighbors from the heap.
            let mut neighbors = Vec::with_capacity(neighbor_heap.len());
            while let Some(neighbor_node) = neighbor_heap.pop() {
                neighbors.push(neighbor_node.id);
            }

            neighbor_graph[i] = neighbors;
        }

        self.graph = neighbor_graph;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::HighDimVector,
        index::{DistanceMetric, Index},
    };

    #[test]
    fn test_build_knn_graph() {
        let v0 = HighDimVector::new(0, vec![1.0, 2.0, 3.0]);
        let v1 = HighDimVector::new(1, vec![4.0, 5.0, 6.0]);
        let v2 = HighDimVector::new(2, vec![7.0, 8.0, 9.0]);
        let mut ssg_index = SSGIndex::new(DistanceMetric::Euclidean);
        ssg_index.add_vector(v0.clone());
        ssg_index.add_vector(v1.clone());
        ssg_index.add_vector(v2.clone());
        ssg_index.construct_knn_graph(100);

        assert_eq!(ssg_index.graph.len(), 3, "Graph should have 3 vectors");
        for i in 0..3 {
            assert_eq!(
                ssg_index.graph[i].len(),
                2,
                "Each vector should have 2 neighbors"
            );
        }
    }

    #[test]
    fn test_build_knn_graph_correct_neighbors() {
        let v0 = HighDimVector::new(0, vec![1.0, 2.0, 3.0]);
        let v1 = HighDimVector::new(1, vec![4.0, 5.0, 6.0]);
        let v2 = HighDimVector::new(2, vec![7.0, 8.0, 9.0]);
        let mut ssg_index = SSGIndex::new(DistanceMetric::Euclidean);
        ssg_index.add_vector(v0.clone());
        ssg_index.add_vector(v1.clone());
        ssg_index.add_vector(v2.clone());
        ssg_index.construct_knn_graph(100);

        println!("{:?}", ssg_index.graph);
        for (i, neighbors) in ssg_index.graph.iter().enumerate() {
            for &neighbor in neighbors {
                println!("{} -> {}", i, neighbor);
                assert_ne!(i, neighbor, "No vector should be its own neighbor");
            }
        }
        assert_eq!(ssg_index.graph[0], vec![2, 1]);
        assert_eq!(ssg_index.graph[1], vec![0, 2]);
        assert_eq!(ssg_index.graph[2], vec![0, 1]);
    }

    #[test]
    fn test_build_knn_graph_k_parameter() {
        let v0 = HighDimVector::new(0, vec![1.0, 2.0, 3.0]);
        let v1 = HighDimVector::new(1, vec![4.0, 5.0, 6.0]);
        let v2 = HighDimVector::new(2, vec![7.0, 8.0, 9.0]);
        let mut ssg_index = SSGIndex::new(DistanceMetric::Euclidean);
        ssg_index.add_vector(v0.clone());
        ssg_index.add_vector(v1.clone());
        ssg_index.add_vector(v2.clone());
        ssg_index.construct_knn_graph(1);

        println!("{:?}", ssg_index.graph);
        assert_eq!(ssg_index.graph.len(), 3, "Graph should have 3 vectors");
        for i in 0..3 {
            assert_eq!(
                ssg_index.graph[i].len(),
                1,
                "Each vector should have 1 neighbor"
            );
        }
        assert_eq!(ssg_index.graph[0][0], 1);
        assert_eq!(ssg_index.graph[1][0], 2);
        assert_eq!(ssg_index.graph[2][0], 1);
    }
}
