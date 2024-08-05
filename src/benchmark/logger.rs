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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, io::Read, time::Duration};

    #[test]
    fn test_add_record() {
        let mut logger = BenchmarkLogger::new();
        let result = BenchmarkResult {
            total_execution_time: Duration::new(10, 0),
            index_execution_time: Duration::new(2, 0),
            query_execution_time: Duration::new(1, 0),
            queries_per_second: 100.0,
            dataset_size: 1000,
            dataset_dimensionality: 128,
            scalability_factor: Some(1.5),
        };

        logger.add_record(&result);
        assert_eq!(logger.records.len(), 1);
        assert_eq!(logger.records[0].total_execution_time, Duration::new(10, 0));
    }

    #[test]
    fn test_write_to_csv() {
        let mut logger = BenchmarkLogger::new();
        let result = BenchmarkResult {
            total_execution_time: Duration::new(10, 0),
            index_execution_time: Duration::new(2, 0),
            query_execution_time: Duration::new(1, 0),
            queries_per_second: 100.0,
            dataset_size: 1000,
            dataset_dimensionality: 128,
            scalability_factor: Some(1.5),
        };

        logger.add_record(&result);

        let file_path = "test_benchmark.csv";
        logger.write_to_csv(file_path).unwrap();

        let mut file = File::open(file_path).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        assert!(contents.contains("total_execution_time,index_execution_time,query_execution_time,queries_per_second,dataset_size,dataset_dimensionality,scalability_factor"));
        assert!(contents.contains("10.0,2.0,1.0,100.0,1000,128,1.5"));

        fs::remove_file(file_path).unwrap();
    }
}
