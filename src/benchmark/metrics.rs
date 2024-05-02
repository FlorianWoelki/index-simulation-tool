use std::time::Duration;

use super::BenchmarkResult;

pub const DEFAULT_SCALABILITY_FACTOR: f64 = 1.0;

/// Calculates the number of queries per second from the total time taken to execute
/// the queries.
///
/// # Arguments
/// - `total_queries_execution_time`: The total time taken to execute the queries.
///
/// # Examples
/// ```
/// use std::time::Duration;
/// use benchmark::metrics::calculate_queries_per_second;
///
/// let total_queries_execution_time = Duration::from_nanos(1_000_000_000);
/// let queries_per_second = calculate_queries_per_second(total_queries_execution_time);
/// assert_eq!(queries_per_second, 1.0);
/// ```
pub fn calculate_queries_per_second(total_queries_execution_time: Duration) -> f64 {
    if !total_queries_execution_time.is_zero() {
        1_000_000_000.0 / total_queries_execution_time.as_nanos() as f64
    } else {
        0.0
    }
}

/// Calculates the scalability factor based on the performance change between two benchmarks.
///
/// This function computes how the queries per second (QPS) performance metric scales with
/// changes in dataset size and dimensionality. A scalability factor greater than 1 indicates
/// performance improvements relative to dataset growth, while a value less than 1 indicates
/// performance degradation.
///
/// # Arguments
/// - (`queries_per_second`, `dataset_size`, `dataset_dimensionality`): The performance metrics
/// of the current benchmark.
/// - `previous_result`: The performance metrics of the previous benchmark.
///
/// # Examples
/// ```
/// use benchmark::metrics::calculate_scalability_factor;
/// use benchmark::BenchmarkResult;
///
/// let previous_result = BenchmarkResult {
///    total_execution_time: std::time::Duration::from_secs(1),
///    index_execution_time: std::time::Duration::from_secs(1),
///    query_execution_time: std::time::Duration::from_secs(1),
///    queries_per_second: 1.0,
///    dataset_size: 1,
///    dataset_dimensionality: 1,
///    scalability_factor: None,
/// };
///
/// let current_result = BenchmarkResult {
///   total_execution_time: std::time::Duration::from_secs(1),
///   index_execution_time: std::time::Duration::from_secs(1),
///   query_execution_time: std::time::Duration::from_secs(1),
///   queries_per_second: 2.0,
///   dataset_size: 2,
///   dataset_dimensionality: 2,
///   scalability_factor: None,
/// };
///
/// let scalability_factor = calculate_scalability_factor(
///   (current_result.queries_per_second, current_result.dataset_size, current_result.dataset_dimensionality),
///   &previous_result,
/// );
/// assert_eq!(scalability_factor, 2.0);
/// ```
pub fn calculate_scalability_factor(
    (queries_per_second, dataset_size, dataset_dimensionality): (f64, usize, usize),
    previous_result: &BenchmarkResult,
) -> f64 {
    let dataset_size_ratio = (dataset_size as f64) / (previous_result.dataset_size as f64);
    let dimensionality_ratio =
        (dataset_dimensionality as f64) / (previous_result.dataset_dimensionality as f64);

    let qps_scalability_factor = queries_per_second / previous_result.queries_per_second;

    qps_scalability_factor / (dataset_size_ratio * dimensionality_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_queries_per_second_normal_case() {
        let duration = Duration::from_secs(1);
        let qps = calculate_queries_per_second(duration);
        assert_eq!(qps, 1.0);
    }

    #[test]
    fn test_queries_per_second_fractional_result() {
        let duration = Duration::from_millis(500);
        let qps = calculate_queries_per_second(duration);
        assert_eq!(qps, 2.0);
    }

    #[test]
    fn test_queries_per_second_zero_duration() {
        let duration = Duration::from_secs(0);
        let qps = calculate_queries_per_second(duration);
        assert_eq!(qps, 0.0);
    }

    #[test]
    fn test_scalability_factor_improvement() {
        let previous_result = BenchmarkResult {
            total_execution_time: Duration::from_secs(1),
            index_execution_time: Duration::from_secs(1),
            query_execution_time: Duration::from_secs(1),
            queries_per_second: 100.0,
            dataset_size: 1000,
            dataset_dimensionality: 10,
            scalability_factor: None,
        };
        let current_qps = 210.0;
        let current_dataset_size = 2000;
        let current_dataset_dimensionality = 10;

        let scalability_factor = calculate_scalability_factor(
            (
                current_qps,
                current_dataset_size,
                current_dataset_dimensionality,
            ),
            &previous_result,
        );

        assert!(scalability_factor > 1.0);
    }

    #[test]
    fn test_scalability_factor_degradation() {
        let previous_result = BenchmarkResult {
            total_execution_time: Duration::from_secs(1),
            index_execution_time: Duration::from_secs(1),
            query_execution_time: Duration::from_secs(1),
            queries_per_second: 200.0,
            dataset_size: 1000,
            dataset_dimensionality: 10,
            scalability_factor: None,
        };
        let current_qps = 150.0;
        let current_dataset_size = 2000;
        let current_dataset_dimensionality = 10;

        let scalability_factor = calculate_scalability_factor(
            (
                current_qps,
                current_dataset_size,
                current_dataset_dimensionality,
            ),
            &previous_result,
        );

        assert!(scalability_factor < 1.0);
    }

    #[test]
    fn test_scalability_factor_constant() {
        let previous_result = BenchmarkResult {
            total_execution_time: Duration::from_secs(1),
            index_execution_time: Duration::from_secs(1),
            query_execution_time: Duration::from_secs(1),
            queries_per_second: 100.0,
            dataset_size: 1000,
            dataset_dimensionality: 10,
            scalability_factor: None,
        };
        let current_qps = 100.0;
        let current_dataset_size = 1000;
        let current_dataset_dimensionality = 10;

        let scalability_factor = calculate_scalability_factor(
            (
                current_qps,
                current_dataset_size,
                current_dataset_dimensionality,
            ),
            &previous_result,
        );

        assert_eq!(scalability_factor, 1.0);
    }
}
