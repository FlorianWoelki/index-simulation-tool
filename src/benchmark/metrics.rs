use std::time::Duration;

use super::BenchmarkResult;

/// Calculates the number of queries per second from the total time taken to execute
/// the queries.
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
}
