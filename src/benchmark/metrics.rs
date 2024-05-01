use std::time::Duration;

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
