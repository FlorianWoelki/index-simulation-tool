use std::time::{Duration, Instant};

use crate::{
    index::Index,
    query::{Query, QueryResult},
};

pub struct BenchmarkResult {
    pub total_execution_time: Duration,
    pub index_execution_time: Duration,
    pub query_execution_time: Duration,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
}

pub struct Benchmark {
    index_type: Box<dyn Index>,
    query_type: Box<dyn Query>,
}

impl Benchmark {
    pub fn new(index_type: Box<dyn Index>, query_type: Box<dyn Query>) -> Self {
        Benchmark {
            index_type,
            query_type,
        }
    }

    pub fn run(&mut self) -> BenchmarkResult {
        let start_time = Instant::now();

        // Builds the index.
        self.index_type.build();
        let index_execution_time = start_time.elapsed();

        // Perform the query.
        let query_results = self
            .query_type
            .execute(&self.index_type.indexed_data(), self.index_type.metric());
        let query_execution_time = start_time.elapsed() - index_execution_time;

        let total_execution_time = start_time.elapsed();

        let (precision, recall, f1_score) = self.calculate_metrics(&query_results);

        BenchmarkResult {
            total_execution_time,
            index_execution_time,
            query_execution_time,
            precision,
            recall,
            f1_score,
        }
    }

    fn calculate_metrics(&self, query_results: &[QueryResult]) -> (f64, f64, f64) {
        (0.0, 0.0, 0.0)
    }
}
