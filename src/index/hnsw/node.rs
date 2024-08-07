use serde::{Deserialize, Serialize};

use crate::data::SparseVector;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(super) struct Node {
    pub(super) id: usize,
    pub(super) connections: Vec<Vec<usize>>,
    pub(super) vector: SparseVector,
    pub(super) layer: usize,
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use super::*;

    #[test]
    fn test_node_equality() {
        let sparse_vector = SparseVector {
            indices: vec![1, 2],
            values: vec![OrderedFloat(0.5), OrderedFloat(1.5)],
        };
        let node1 = Node {
            id: 1,
            connections: vec![vec![2, 3], vec![4, 5]],
            vector: sparse_vector.clone(),
            layer: 0,
        };
        let node2 = Node {
            id: 1,
            connections: vec![vec![2, 3], vec![4, 5]],
            vector: sparse_vector.clone(),
            layer: 0,
        };

        assert_eq!(node1, node2);
    }

    #[test]
    fn test_node_inequality() {
        let sparse_vector1 = SparseVector {
            indices: vec![1, 2],
            values: vec![OrderedFloat(0.5), OrderedFloat(1.5)],
        };
        let sparse_vector2 = SparseVector {
            indices: vec![1, 3],
            values: vec![OrderedFloat(0.5), OrderedFloat(2.5)],
        };
        let node1 = Node {
            id: 1,
            connections: vec![vec![2, 3], vec![4, 5]],
            vector: sparse_vector1,
            layer: 0,
        };
        let node2 = Node {
            id: 2,
            connections: vec![vec![2, 3], vec![4, 5]],
            vector: sparse_vector2,
            layer: 1,
        };

        assert_ne!(node1, node2);
    }
}
