use std::fs::File;

use csv::Writer;
use serde::Serialize;

use super::{metrics::DEFAULT_SCALABILITY_FACTOR, BenchmarkResult};

/// A struct that represents a single benchmark record that can be serialized to CSV.
///
/// This struct is similar to the `BenchmarkResult` struct, but it is designed to be serialized
/// to CSV. It contains the same fields as `BenchmarkResult`, but the fields are of different types.
#[derive(Serialize)]
struct BenchmarkRecord {
    pub total_execution_time: f32,
    pub index_execution_time: f32,
    pub query_execution_time: f32,
    pub queries_per_second: f32,
    pub dataset_size: usize,
    pub dataset_dimensionality: usize,
    pub scalability_factor: f32,
}

pub struct BenchmarkLogger {
    records: Vec<BenchmarkResult>,
}

impl BenchmarkLogger {
    pub fn new() -> Self {
        BenchmarkLogger {
            records: Vec::new(),
        }
    }

    pub fn add_record(&mut self, result: &BenchmarkResult) {
        self.records.push(*result);
    }

    pub fn write_to_csv(&self, file_path: &str) -> Result<(), csv::Error> {
        let file = File::create(file_path)?;
        let mut writer = Writer::from_writer(file);

        for record in &self.records {
            writer.serialize(&BenchmarkRecord {
                total_execution_time: record.total_execution_time.as_secs_f32(),
                index_execution_time: record.index_execution_time.as_secs_f32(),
                query_execution_time: record.query_execution_time.as_secs_f32(),
                queries_per_second: record.queries_per_second,
                dataset_size: record.dataset_size,
                dataset_dimensionality: record.dataset_dimensionality,
                scalability_factor: record
                    .scalability_factor
                    .unwrap_or(DEFAULT_SCALABILITY_FACTOR),
            })?;
        }

        writer.flush()?;
        Ok(())
    }
}
