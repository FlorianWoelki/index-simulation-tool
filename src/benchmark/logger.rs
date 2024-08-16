use std::fs::File;

use csv::Writer;

use super::SerializableBenchmark;

pub struct BenchmarkLogger<T: SerializableBenchmark> {
    records: Vec<T>,
}

impl<T: SerializableBenchmark> BenchmarkLogger<T> {
    pub fn new() -> Self {
        BenchmarkLogger {
            records: Vec::new(),
        }
    }

    pub fn add_record(&mut self, result: T) {
        self.records.push(result);
    }

    pub fn write_to_csv(&self, file_path: String) -> Result<(), csv::Error> {
        let file = File::create(file_path)?;
        let mut writer = Writer::from_writer(file);

        for record in &self.records {
            writer.serialize(record)?;
        }

        writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::benchmark::GenericBenchmarkResult;

    use super::*;
    use std::{fs, io::Read, time::Duration};

    #[test]
    fn test_add_record() {
        let mut logger: BenchmarkLogger<GenericBenchmarkResult> = BenchmarkLogger::new();
        let result = GenericBenchmarkResult {
            execution_time: Duration::new(10, 0).as_secs_f32(),
            dataset_size: 1000,
            dataset_dimensionality: 128,
            consumed_cpu: 1.0,
            consumed_memory: 1.0,
        };

        logger.add_record(result);
        assert_eq!(logger.records.len(), 1);
        assert_eq!(
            logger.records[0].execution_time,
            Duration::new(10, 0).as_secs_f32()
        );
    }

    #[test]
    fn test_write_to_csv() {
        let mut logger: BenchmarkLogger<GenericBenchmarkResult> = BenchmarkLogger::new();
        let result = GenericBenchmarkResult {
            execution_time: Duration::new(10, 0).as_secs_f32(),
            dataset_size: 1000,
            dataset_dimensionality: 128,
            consumed_cpu: 1.0,
            consumed_memory: 1.0,
        };

        logger.add_record(result);

        let file_path = "test_benchmark.csv";
        logger.write_to_csv(file_path.to_string()).unwrap();

        let mut file = File::open(file_path).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        assert!(contents.contains("total_execution_time,index_execution_time,query_execution_time,queries_per_second,dataset_size,dataset_dimensionality,scalability_factor"));
        assert!(contents.contains("10.0,2.0,1.0,100.0,1000,128,1.5"));

        fs::remove_file(file_path).unwrap();
    }
}
