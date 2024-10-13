use std::{ops::Index, time::Duration};

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

    for result in search_results {
        if groundtruth
            .iter()
            .any(|gt_vector| gt_vector.indices == result.indices)
        {
            correct_results += 1;
        }
    }

    correct_results as f32 / k as f32
}

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
