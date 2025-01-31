use std::time::Duration;

use crate::data::vector::SparseVector;

use super::IndexBenchmarkResult;

#[allow(dead_code)]
pub const DEFAULT_SCALABILITY_FACTOR: f32 = 1.0;

pub fn calculate_queries_per_second(num_queries: usize, query_duration: Duration) -> f32 {
    num_queries as f32 / query_duration.as_secs_f32()
}

pub fn calculate_recall(
    search_results: &[SparseVector],
    groundtruth: &[SparseVector],
    k: usize,
) -> f32 {
    let mut correct_results = 0;
    for result in search_results.iter().take(k) {
        if groundtruth.iter().any(|gt| gt.indices == result.indices) {
            correct_results += 1;
        }
    }

    correct_results as f32 / groundtruth.len() as f32
}

pub fn calculate_precision(
    search_results: &[SparseVector],
    groundtruth: &[SparseVector],
    k: usize,
) -> f32 {
    let mut correct_results = 0;
    for result in search_results.iter().take(k) {
        if groundtruth.iter().any(|gt| gt.indices == result.indices) {
            correct_results += 1;
        }
    }

    correct_results as f32 / k as f32
}

pub fn calculate_f1_score(precision: f32, recall: f32) -> f32 {
    if precision + recall > 0.0 {
        (2.0 * precision * recall) / (precision + recall)
    } else {
        0.0
    }
}

/// A scalability factor greater than one indicates that the algorithm is scaling
/// better than linear expectations, maintaining or improving its relative performance
/// despite increases in data size and dimensionality.
/// Conversely, a factor equal to one suggests perfect linear scaling, while a factor
/// less than one indicates that the indexing algorithm's performance decreases more
/// than what linear scaling would predict as the dataset grows or becomes more complex.
pub fn calculate_scalability_factor(
    (queries_per_second, dataset_size, dataset_dimensionality): (f32, usize, usize),
    previous_result: &IndexBenchmarkResult,
) -> f32 {
    let dataset_size_ratio = (dataset_size as f32) / (previous_result.dataset_size as f32);
    let dimensionality_ratio =
        (dataset_dimensionality as f32) / (previous_result.dataset_dimensionality as f32);

    let expected_qps =
        previous_result.queries_per_second / (dataset_size_ratio * dimensionality_ratio);

    queries_per_second / expected_qps
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use super::*;
    use std::time::Duration;

    #[test]
    fn test_perfect_precision() {
        let search_results = vec![
            SparseVector {
                indices: vec![1, 2, 3],
                values: vec![OrderedFloat(0.1), OrderedFloat(0.2), OrderedFloat(0.3)],
            },
            SparseVector {
                indices: vec![2, 3, 4],
                values: vec![OrderedFloat(0.2), OrderedFloat(0.3), OrderedFloat(0.4)],
            },
        ];
        let groundtruth = vec![
            SparseVector {
                indices: vec![1, 2, 3],
                values: vec![OrderedFloat(0.1), OrderedFloat(0.2), OrderedFloat(0.3)],
            },
            SparseVector {
                indices: vec![2, 3, 4],
                values: vec![OrderedFloat(0.2), OrderedFloat(0.3), OrderedFloat(0.4)],
            },
        ];

        assert_eq!(calculate_precision(&search_results, &groundtruth, 2), 1.0);
    }

    #[test]
    fn test_partial_precision() {
        let search_results = vec![
            SparseVector {
                indices: vec![1, 2, 3],
                values: vec![OrderedFloat(0.1), OrderedFloat(0.2), OrderedFloat(0.3)],
            },
            SparseVector {
                indices: vec![2, 3, 4],
                values: vec![OrderedFloat(0.2), OrderedFloat(0.3), OrderedFloat(0.4)],
            },
        ];
        let groundtruth = vec![
            SparseVector {
                indices: vec![1, 2, 3],
                values: vec![OrderedFloat(0.1), OrderedFloat(0.2), OrderedFloat(0.3)],
            },
            SparseVector {
                indices: vec![3, 4, 5],
                values: vec![OrderedFloat(0.3), OrderedFloat(0.4), OrderedFloat(0.5)],
            },
        ];

        assert_eq!(calculate_precision(&search_results, &groundtruth, 2), 0.5);
    }

    #[test]
    fn test_f1_score() {
        assert_eq!(calculate_f1_score(1.0, 1.0), 1.0);
        assert_eq!(calculate_f1_score(0.5, 0.5), 0.5);
        let f1 = calculate_f1_score(0.5, 1.0);
        assert!(f1 > 0.66 && f1 < 0.67);
        assert_eq!(calculate_f1_score(0.0, 0.0), 0.0);
        assert_eq!(calculate_f1_score(0.0, 0.5), 0.0);
        assert_eq!(calculate_f1_score(0.5, 0.0), 0.0);
    }

    #[test]
    fn test_perfect_recall() {
        let search_results = vec![
            SparseVector {
                indices: vec![1, 2, 3],
                values: vec![OrderedFloat(0.1), OrderedFloat(0.2), OrderedFloat(0.3)],
            },
            SparseVector {
                indices: vec![2, 3, 4],
                values: vec![OrderedFloat(0.2), OrderedFloat(0.3), OrderedFloat(0.4)],
            },
        ];
        let groundtruth = vec![
            SparseVector {
                indices: vec![1, 2, 3],
                values: vec![OrderedFloat(0.1), OrderedFloat(0.2), OrderedFloat(0.3)],
            },
            SparseVector {
                indices: vec![2, 3, 4],
                values: vec![OrderedFloat(0.2), OrderedFloat(0.3), OrderedFloat(0.4)],
            },
        ];

        assert_eq!(calculate_recall(&search_results, &groundtruth, 2), 1.0);
    }

    #[test]
    fn test_partial_recall() {
        let search_results = vec![
            SparseVector {
                indices: vec![1, 2, 3],
                values: vec![OrderedFloat(0.1), OrderedFloat(0.2), OrderedFloat(0.3)],
            },
            SparseVector {
                indices: vec![2, 3, 4],
                values: vec![OrderedFloat(0.2), OrderedFloat(0.3), OrderedFloat(0.4)],
            },
            SparseVector {
                indices: vec![4, 5, 6],
                values: vec![OrderedFloat(0.4), OrderedFloat(0.5), OrderedFloat(0.6)],
            },
        ];
        let groundtruth = vec![
            SparseVector {
                indices: vec![1, 2, 3],
                values: vec![OrderedFloat(0.1), OrderedFloat(0.2), OrderedFloat(0.3)],
            },
            SparseVector {
                indices: vec![4, 5, 6],
                values: vec![OrderedFloat(0.4), OrderedFloat(0.5), OrderedFloat(0.6)],
            },
            SparseVector {
                indices: vec![7, 8, 9],
                values: vec![OrderedFloat(0.7), OrderedFloat(0.8), OrderedFloat(0.9)],
            },
        ];

        assert_eq!(
            calculate_recall(&search_results, &groundtruth, 3),
            2.0 / 3.0
        );
    }

    #[test]
    fn test_queries_per_second_normal_case() {
        let num_queries = 100;
        let duration = Duration::from_secs(1);
        let qps = calculate_queries_per_second(num_queries, duration);
        assert_eq!(qps, 100.0);
    }

    #[test]
    fn test_queries_per_second_fractional_result() {
        let num_queries = 100;
        let duration = Duration::from_millis(500);
        let qps = calculate_queries_per_second(num_queries, duration);
        assert_eq!(qps, 200.0);
    }

    #[test]
    fn test_queries_per_second_zero_duration() {
        let num_queries = 100;
        let duration = Duration::from_secs(0);
        let qps = calculate_queries_per_second(num_queries, duration);
        assert!(qps.is_infinite());
    }

    #[test]
    fn test_scalability_factor_improvement() {
        let previous_result = IndexBenchmarkResult {
            execution_time: Duration::from_secs(1).as_secs_f32(),
            dataset_size: 1000,
            dataset_dimensionality: 10,
            build_time: Duration::from_secs(1).as_secs_f32(),
            search_time: Duration::from_secs(1).as_secs_f32(),
            queries_per_second: 100.0,
            scalability_factor: None,
            add_vector_performance: Duration::from_secs(1).as_secs_f32(),
            remove_vector_performance: Duration::from_secs(1).as_secs_f32(),
            index_disk_space: 0.0,
            index_loading_time: Duration::from_secs(1).as_secs_f32(),
            index_saving_time: Duration::from_secs(1).as_secs_f32(),
            recall: 0.0,
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
        let previous_result = IndexBenchmarkResult {
            execution_time: Duration::from_secs(1).as_secs_f32(),
            dataset_size: 1000,
            dataset_dimensionality: 10,
            build_time: Duration::from_secs(1).as_secs_f32(),
            search_time: Duration::from_secs(1).as_secs_f32(),
            queries_per_second: 200.0,
            scalability_factor: None,
            add_vector_performance: Duration::from_secs(1).as_secs_f32(),
            remove_vector_performance: Duration::from_secs(1).as_secs_f32(),
            index_disk_space: 0.0,
            index_loading_time: Duration::from_secs(1).as_secs_f32(),
            index_saving_time: Duration::from_secs(1).as_secs_f32(),
            recall: 0.0,
        };
        let current_qps = 50.0;
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

        println!("{:?}", scalability_factor);
        assert!(scalability_factor < 1.0);
    }

    #[test]
    fn test_scalability_factor_constant() {
        let previous_result = IndexBenchmarkResult {
            execution_time: Duration::from_secs(1).as_secs_f32(),
            dataset_size: 1000,
            dataset_dimensionality: 10,
            build_time: Duration::from_secs(1).as_secs_f32(),
            search_time: Duration::from_secs(1).as_secs_f32(),
            queries_per_second: 100.0,
            scalability_factor: None,
            add_vector_performance: Duration::from_secs(1).as_secs_f32(),
            remove_vector_performance: Duration::from_secs(1).as_secs_f32(),
            index_disk_space: 0.0,
            index_loading_time: Duration::from_secs(1).as_secs_f32(),
            index_saving_time: Duration::from_secs(1).as_secs_f32(),
            recall: 0.0,
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
