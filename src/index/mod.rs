use std::fmt::Debug;

use ordered_float::OrderedFloat;

use crate::data::HighDimVector;

pub mod hnsw;
pub mod linscan;
pub mod lsh;
pub mod naive;
pub mod neighbor;
pub mod ssg;

#[derive(PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    DotProduct,
    Jaccard,
}

impl Debug for DistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceMetric::Euclidean => write!(f, "Euclidean"),
            DistanceMetric::Manhattan => write!(f, "Manhattan"),
            DistanceMetric::Cosine => write!(f, "Cosine"),
            DistanceMetric::DotProduct => write!(f, "DotProduct"),
            DistanceMetric::Jaccard => write!(f, "Jaccard"),
        }
    }
}

impl Copy for DistanceMetric {}

impl Clone for DistanceMetric {
    fn clone(&self) -> DistanceMetric {
        *self
    }
}

pub trait Index {
    fn new(metric: DistanceMetric) -> Self
    where
        Self: Sized;
    fn add_vector(&mut self, vector: HighDimVector);
    fn build(&mut self);
    fn search(&self, query_vector: &HighDimVector, k: usize) -> Vec<HighDimVector>;
}

// TODO: Move this
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone)]
pub struct SparseVector {
    pub indices: Vec<usize>,
    pub values: Vec<OrderedFloat<f32>>,
}

// TODO: Move this
#[derive(Debug, PartialEq)]
pub struct QueryResult {
    pub vector: SparseVector,
    pub score: OrderedFloat<f32>,
}

impl Eq for QueryResult {}

impl PartialOrd for QueryResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for QueryResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

pub trait SparseIndex {
    fn new(metric: DistanceMetric) -> Self
    where
        Self: Sized;
    fn add_vector(&mut self, vector: &SparseVector);
    fn build(&mut self);
    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult>;
}
