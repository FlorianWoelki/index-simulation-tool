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

    /// Calculates the Euclidean distance between this vector and another.
    ///
    /// # Parameters
    /// * `other` - A reference to another `HighDimVector` to calculate the distance to.
    ///
    /// # Returns
    /// Returns the Euclidean distance as an `f64`.
    ///
    /// # Examples
    /// ```
    /// let vector1 = HighDimVector::new(vec![1.0, 2.0, 3.0]);
    /// let vector2 = HighDimVector::new(vec![4.0, 5.0, 6.0]);
    /// let distance = vector1.distance(&vector2);
    /// ```
    pub fn distance(&self, other: &Self) -> f64 {
        self.dimensions
            .iter()
            .zip(other.dimensions.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// A naive implementation of an index using a vector to store high-dimensional vectors.
///
/// This struct can be used to add vectors to the index and to find the nearest vector
/// to a given query vector.
pub struct NaiveIndex {
    /// A vector of `HighDimVector` representing the high-dimensional vectors.
    vectors: Vec<HighDimVector>,
}

impl NaiveIndex {
    /// Constructs a new `NaiveIndex`.
    ///
    /// Initialises an empty vector to store `HighDimVector` instances.
    ///
    /// # Examples
    /// ```
    /// let index = NaiveIndex::new();
    /// ```
    pub fn new() -> Self {
        NaiveIndex {
            vectors: Vec::new(),
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
    pub fn add_vector(&mut self, vector: HighDimVector) {
        self.vectors.push(vector);
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
    pub fn find_nearest(&self, query: &HighDimVector) -> Option<&HighDimVector> {
        self.vectors
            .iter()
            .min_by(|a, b| a.distance(query).partial_cmp(&b.distance(query)).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearest_neighbor() {
        let mut index = NaiveIndex::new();
        index.add_vector(HighDimVector::new(vec![1.0, 2.0, 3.0]));
        index.add_vector(HighDimVector::new(vec![4.0, 5.0, 6.0]));
        index.add_vector(HighDimVector::new(vec![7.0, 8.0, 9.0]));

        let query = HighDimVector::new(vec![1.0, 2.1, 2.9]);
        let nearest = index.find_nearest(&query).unwrap();

        assert_eq!(nearest.dimensions, vec![1.0, 2.0, 3.0]);
    }
}
