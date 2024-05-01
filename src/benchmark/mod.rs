use std::time::{Duration, Instant};

use crate::{index::Index, query::Query};

mod metrics;

pub struct BenchmarkResult {
    pub total_execution_time: Duration,
    pub index_execution_time: Duration,
    pub query_execution_time: Duration,
    pub queries_per_second: f64,
    pub dataset_size: usize,
    pub dataset_dimensionality: usize,
    pub scalability_factor: f64,
}

pub struct Benchmark {
    index_type: Box<dyn Index>,
    query_type: Box<dyn Query>,
    previous_benchmark_result: Option<BenchmarkResult>,
}

impl Benchmark {
    pub fn new(
        index_type: Box<dyn Index>,
        query_type: Box<dyn Query>,
        previous_benchmark_result: Option<BenchmarkResult>,
    ) -> Self {
        Benchmark {
            index_type,
            query_type,
            previous_benchmark_result,
        }
    }

    pub fn run(&mut self, dataset_size: usize, dimensions: usize) -> BenchmarkResult {
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

        let queries_per_second = metrics::calculate_queries_per_second(query_execution_time);

        let scalability_factor = if let Some(previous_result) = &self.previous_benchmark_result {
            metrics::calculate_scalability_factor(
                (queries_per_second, dataset_size, dimensions),
                &previous_result,
            )
        } else {
            1.0
        };

        BenchmarkResult {
            total_execution_time,
            index_execution_time,
            query_execution_time,
            queries_per_second,
            scalability_factor,
            dataset_size,
            dataset_dimensionality: dimensions,
        }
    }
}
