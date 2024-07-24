macro_rules! measure_time {
    ($func:expr) => {{
        let start = Instant::now();
        let result = $func;
        let end = Instant::now();
        let duration = end.duration_since(start);
        (result, duration)
    }};
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    #[test]
    fn test_measure_time() {
        let (result, duration) = measure_time!({
            std::thread::sleep(Duration::from_millis(100));
            42
        });
        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(100));
    }

    #[test]
    fn test_measure_time_zero_duration() {
        let (result, duration) = measure_time!(42);
        assert_eq!(result, 42);
        assert!(duration < Duration::from_millis(1));
    }

    #[test]
    fn test_measure_time_multiple_statements() {
        let (result, duration) = measure_time!({
            let mut sum = 0;
            for i in 1..100 {
                sum += i;
            }
            sum
        });
        assert_eq!(result, 4950);
        assert!(duration < Duration::from_millis(1));
    }

    #[test]
    fn test_measure_time_empty_block() {
        let (result, duration) = measure_time!({});
        assert_eq!(result, ());
        assert!(duration < Duration::from_millis(1));
    }
}
