use std::fmt::Debug;

use crate::data::{QueryResult, SparseVector};

pub mod annoy;
pub mod hnsw;
pub mod linscan;
pub mod minhash;
pub mod neighbor;
pub mod pq;
pub mod simhash;
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

pub trait SparseIndex {
    fn new(metric: DistanceMetric) -> Self
    where
        Self: Sized;
    fn add_vector(&mut self, vector: &SparseVector);
    fn build(&mut self);
    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult>;
}
