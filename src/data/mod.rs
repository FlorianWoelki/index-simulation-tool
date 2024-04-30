pub mod generator;

#[derive(Debug)]
pub struct HighDimVector {
    pub dimensions: Vec<f64>,
}

impl HighDimVector {
    pub fn new(dimensions: Vec<f64>) -> Self {
        HighDimVector { dimensions }
    }
}
