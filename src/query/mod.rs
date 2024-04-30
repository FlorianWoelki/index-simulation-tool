use crate::{data::HighDimVector, index::DistanceMetric};

pub mod hnsw;
pub mod naive;

pub struct QueryResult {
    pub index: usize,
    pub distance: f64,
}

pub trait Query {
    fn new(query_vector: HighDimVector, k: usize) -> Self
    where
        Self: Sized;
    fn execute(&self, data: &Vec<HighDimVector>, metric: DistanceMetric) -> Vec<QueryResult>;
}
