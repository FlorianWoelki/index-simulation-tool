use crate::data::SparseVector;

#[derive(Clone, Debug)]
pub(super) struct Node {
    pub(super) id: usize,
    pub(super) connections: Vec<Vec<usize>>,
    pub(super) vector: SparseVector,
    pub(super) layer: usize,
}
