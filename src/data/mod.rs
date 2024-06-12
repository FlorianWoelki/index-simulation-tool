use ordered_float::OrderedFloat;

use crate::index::DistanceMetric;

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
    pub fn distance(&self, other: &SparseVector, metric: &DistanceMetric) -> f32 {
        match metric {
            DistanceMetric::Euclidean => self.euclidean_distance(other),
            DistanceMetric::Cosine => 1.0 - self.cosine_similarity(other),
            DistanceMetric::Jaccard => 1.0 - self.jaccard_similarity(other),
        }
    }

    pub fn jaccard_similarity(&self, other: &SparseVector) -> f32 {
        let mut p = 0;
        let mut q = 0;
        let mut intersection = 0;
        let mut union = 0;

        while p < self.indices.len() && q < other.indices.len() {
            if self.indices[p] == other.indices[q] {
                intersection += 1;
                union += 1;
                p += 1;
                q += 1;
            } else if self.indices[p] < other.indices[q] {
                union += 1;
                p += 1;
            } else {
                union += 1;
                q += 1;
            }
        }

        union += self.indices.len() - p;
        union += other.indices.len() - q;

        if union == 0 {
            return 1.0;
        }

        intersection as f32 / union as f32
    }

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

    pub fn cosine_similarity(&self, other: &SparseVector) -> f32 {
        let mut dot_product = 0.0f32;
        let mut magnitude_a = 0.0f32;
        let mut magnitude_b = 0.0f32;

        let mut ai = 0;
        let mut bi = 0;

        while ai < self.indices.len() && bi < other.indices.len() {
            if self.indices[ai] == other.indices[bi] {
                dot_product += (self.values[ai] * other.values[bi]).into_inner();
                magnitude_a += (self.values[ai] * self.values[ai]).into_inner();
                magnitude_b += (other.values[bi] * other.values[bi]).into_inner();
                ai += 1;
                bi += 1;
            } else if self.indices[ai] < other.indices[bi] {
                magnitude_a += (self.values[ai] * self.values[ai]).into_inner();
                ai += 1;
            } else {
                magnitude_b += (other.values[bi] * other.values[bi]).into_inner();
                bi += 1;
            }
        }

        while ai < self.indices.len() {
            magnitude_a += (self.values[ai] * self.values[ai]).into_inner();
            ai += 1;
        }

        while bi < other.indices.len() {
            magnitude_b += (other.values[bi] * other.values[bi]).into_inner();
            bi += 1;
        }

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return 0.0;
        }

        dot_product / (magnitude_a.sqrt() * magnitude_b.sqrt())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let v1 = SparseVector {
            indices: vec![0, 2, 3],
            values: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
        };
        let v2 = SparseVector {
            indices: vec![0, 1, 2],
            values: vec![OrderedFloat(4.0), OrderedFloat(5.0), OrderedFloat(6.0)],
        };
        assert_eq!(v1.euclidean_distance(&v2), 7.6811457);
    }

    #[test]
    fn test_cosine_similarity() {
        let v1 = SparseVector {
            indices: vec![0, 2, 3],
            values: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
        };
        let v2 = SparseVector {
            indices: vec![0, 1, 2],
            values: vec![OrderedFloat(4.0), OrderedFloat(5.0), OrderedFloat(6.0)],
        };
        assert_eq!(v1.cosine_similarity(&v2), 0.4873159);
    }
}
