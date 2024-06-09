use ordered_float::OrderedFloat;

pub mod generator_dense;
pub mod generator_sparse;
pub mod ms_marco;
pub mod sift;

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone)]
pub struct SparseVector {
    pub indices: Vec<usize>,
    pub values: Vec<OrderedFloat<f32>>,
}

impl SparseVector {
    pub fn euclidean_distance(&self, other: &SparseVector) -> f32 {
        let mut p = 0;
        let mut q = 0;
        let mut distance = 0.0;

        while p < self.indices.len() && q < other.indices.len() {
            if self.indices[p] == other.indices[q] {
                let diff = self.values[p].into_inner() - other.values[q].into_inner();
                distance += diff * diff;
                p += 1;
                q += 1;
            } else if self.indices[p] < other.indices[q] {
                distance += self.values[p].into_inner() * self.values[p].into_inner();
                p += 1;
            } else {
                distance += other.values[q].into_inner() * other.values[q].into_inner();
                q += 1;
            }
        }

        while p < self.indices.len() {
            distance += self.values[p].into_inner() * self.values[p].into_inner();
            p += 1;
        }

        while q < other.indices.len() {
            distance += other.values[q].into_inner() * other.values[q].into_inner();
            q += 1;
        }

        distance.sqrt()
    }
}

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
