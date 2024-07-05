use std::fmt::Debug;

use crate::data::{QueryResult, SparseVector};

pub mod annoy;
pub mod hnsw;
pub mod ivfpq;
pub mod linscan;
pub mod lsh;
pub mod neighbor;
pub mod nsw;
pub mod pq;

#[derive(PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    Jaccard,
    Angular,
}

impl Debug for DistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceMetric::Euclidean => write!(f, "Euclidean"),
            DistanceMetric::Cosine => write!(f, "Cosine"),
            DistanceMetric::Jaccard => write!(f, "Jaccard"),
            DistanceMetric::Angular => write!(f, "Angular"),
        }
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
