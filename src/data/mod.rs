use ordered_float::OrderedFloat;

pub mod generator_sparse;
pub mod ms_marco;
pub mod pca;
pub mod plot;
pub mod tsne;
pub mod vector;

#[derive(Debug, PartialEq)]
pub struct QueryResult {
    //pub vector: SparseVector,
    pub index: usize,
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
