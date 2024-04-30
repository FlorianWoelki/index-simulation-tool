pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
}

pub trait Index {
    fn new(metric: DistanceMetric) -> Self
    where
        Self: Sized;
    fn add_vector(&mut self, vector: HighDimVector);
    fn bulk_add_vectors(&mut self, vectors: Vec<HighDimVector>);
    fn find_nearest(&self, query: &HighDimVector) -> Option<&HighDimVector>;
    fn clear(&mut self);
    fn size(&self) -> usize;
}

/// Represents a high-dimensional vector.
///
/// This struct is used to store and manipulate high-dimensional vectors
/// typically used in vector databases for operations such as calculating
/// distances between vectors.
pub struct HighDimVector {
    /// A vector of `f64` representing the coordinates of the high-dimensional vector.
    dimensions: Vec<f64>,
}

impl HighDimVector {
    /// Constructs a new `HighDimVector` given a vector of `f64`.
    ///
    /// # Parameters
    /// * `dimensions` - A vector of `f64` representing the coordinates of the vector.
    ///
    /// # Examples
    /// ```
    /// let vector = HighDimVector::new(vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn new(dimensions: Vec<f64>) -> Self {
        HighDimVector { dimensions }
    }
}

/// A naive implementation of an index using a vector to store high-dimensional vectors.
///
/// This struct can be used to add vectors to the index and to find the nearest vector
/// to a given query vector.
pub struct NaiveIndex {
    /// A vector of `HighDimVector` representing the high-dimensional vectors.
    vectors: Vec<HighDimVector>,
    metric: DistanceMetric,
}

impl Index for NaiveIndex {
    /// Constructs a new `NaiveIndex`.
    ///
    /// Initialises an empty vector to store `HighDimVector` instances.
    ///
    /// # Examples
    /// ```
    /// let index = NaiveIndex::new();
    /// ```
    fn new(metric: DistanceMetric) -> Self {
        NaiveIndex {
            vectors: Vec::new(),
            metric,
        }
    }

    /// Adds a `HighDimVector` to the index.
    ///
    /// # Parameters
    /// * `vector` - A `HighDimVector` that will be added to the index.
    ///
    /// # Examples
    /// ```
    /// let mut index = NaiveIndex::new();
    /// index.add_vector(HighDimVector::new(vec![1.0, 2.0, 3.0]));
    /// ```
    fn add_vector(&mut self, vector: HighDimVector) {
        self.vectors.push(vector);
    }

    fn bulk_add_vectors(&mut self, vectors: Vec<HighDimVector>) {
        self.vectors.extend(vectors);
    }

    /// Finds the nearest vector in the index to a given query vector.
    ///
    /// # Parameters
    /// * `query` - A reference to a `HighDimVector` representing the query vector.
    ///
    /// # Returns
    /// Returns an `Option` containing a reference to the nearest `HighDimVector` in
    /// the index. If the index is empty, `None` is returned.
    ///
    /// # Examples
    /// ```
    /// let mut index = NaiveIndex::new();
    /// index.add_vector(HighDimVector::new(vec![1.0, 2.0, 3.0]));
    /// let query = HighDimVector::new(vec![1.0, 2.1, 2.9]);
    /// let nearest = index.find_nearest(&query).unwrap();
    /// ```
    fn find_nearest(&self, query: &HighDimVector) -> Option<&HighDimVector> {
        self.vectors.iter().min_by(|a, b| {
            self.distance(a, query)
                .partial_cmp(&self.distance(b, query))
                .unwrap()
        })
    }

    fn clear(&mut self) {
        self.vectors.clear();
    }

    fn size(&self) -> usize {
        self.vectors.len()
    }
}

impl NaiveIndex {
    fn distance(&self, a: &HighDimVector, b: &HighDimVector) -> f64 {
        match self.metric {
            DistanceMetric::Euclidean => a
                .dimensions
                .iter()
                .zip(b.dimensions.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt(),

            DistanceMetric::Manhattan => a
                .dimensions
                .iter()
                .zip(b.dimensions.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>(),

            DistanceMetric::Cosine => {
                let dot_product: f64 = a
                    .dimensions
                    .iter()
                    .zip(b.dimensions.iter())
                    .map(|(x, y)| x * y)
                    .sum();
                let norm_a: f64 = a.dimensions.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                let norm_b: f64 = b.dimensions.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                1.0 - dot_product / (norm_a * norm_b)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearest_neighbor() {
        let mut index = NaiveIndex::new(DistanceMetric::Euclidean);
        index.add_vector(HighDimVector::new(vec![1.0, 2.0, 3.0]));
        index.add_vector(HighDimVector::new(vec![4.0, 5.0, 6.0]));
        index.add_vector(HighDimVector::new(vec![7.0, 8.0, 9.0]));

        let query = HighDimVector::new(vec![1.0, 2.1, 2.9]);
        let nearest = index.find_nearest(&query).unwrap();

        assert_eq!(nearest.dimensions, vec![1.0, 2.0, 3.0]);
    }
}
