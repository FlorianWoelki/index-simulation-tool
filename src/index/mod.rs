use std::fmt::Debug;

use crate::data::HighDimVector;

pub mod naive;

#[derive(PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
}

impl Debug for DistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceMetric::Euclidean => write!(f, "Euclidean"),
            DistanceMetric::Manhattan => write!(f, "Manhattan"),
            DistanceMetric::Cosine => write!(f, "Cosine"),
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
    fn indexed_data(&self) -> &Vec<HighDimVector>;
    fn metric(&self) -> DistanceMetric;
}
