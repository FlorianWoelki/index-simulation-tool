use crate::{
    data::HighDimVector,
    index::{calculate_distance, Index},
};

pub fn range_query<'a>(
    index: &'a impl Index,
    query: &'a HighDimVector,
    range: f64,
) -> Vec<&'a HighDimVector> {
    index
        .iter()
        .filter(|&vector| {
            calculate_distance(query, vector, crate::index::DistanceMetric::Euclidean) <= range
        })
        .collect()
}
