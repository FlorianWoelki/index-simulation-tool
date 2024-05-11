#[cfg(test)]
mod tests {
    use index_simulation_tool::{
        data::HighDimVector,
        index::{DistanceMetric, Index},
    };

    fn test_n_times<I: Index + 'static>(n: usize, dataset_size: usize) {
        let expected = vec![
            HighDimVector::new(80, vec![208.0, 208.0, 208.0]),
            HighDimVector::new(81, vec![209.0, 209.0, 209.0]),
            HighDimVector::new(79, vec![207.0, 207.0, 207.0]),
            HighDimVector::new(82, vec![210.0, 210.0, 210.0]),
            HighDimVector::new(78, vec![206.0, 206.0, 206.0]),
        ];
        for _ in 0..n {
            let index = create_index::<I>(dataset_size);
            let result = search(index);
            assert_eq!(result.len(), 5);

            assert_eq!(expected[0], result[0]);
            for i in 1..result.len() {
                assert!(expected.contains(&result[i]));
            }
        }
    }

    fn create_index<I: Index + 'static>(n: usize) -> I {
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            samples.push(HighDimVector::new(i, vec![128.0 + i as f64; 3]));
        }
        let mut index = I::new(DistanceMetric::Euclidean);
        for sample in samples {
            index.add_vector(sample);
        }
        index.build();
        index
    }

    fn search<I: Index + 'static>(index: I) -> Vec<HighDimVector> {
        let query_vector = HighDimVector::new(999999999, vec![208.0; 3]);
        let k = 5;
        index.search(&query_vector, k)
    }

    #[test]
    fn test_hnsw_index() {
        test_n_times::<index_simulation_tool::index::hnsw::HNSWIndex>(3, 1000);
    }

    #[test]
    fn test_ssg_index() {
        test_n_times::<index_simulation_tool::index::ssg::SSGIndex>(3, 1000);
    }
}
