use crate::{data::HighDimVector, index::DistanceMetric};

pub mod naive;

pub struct QueryResult {
    pub index: usize,
    pub distance: f64,
}

pub trait Query {
    fn execute(&self, data: &Vec<HighDimVector>, metric: DistanceMetric) -> Vec<QueryResult>;
}
