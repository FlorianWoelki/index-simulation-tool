#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use index_simulation_tool::{
        data::SparseVector,
        index::lsh::{LSHHashType, LSHIndex},
    };
    use ordered_float::OrderedFloat;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_sim_hash_index() {
        let seed = 42;
        let num_buckets = 40;
        let num_hash_functions = 10;
        let num_vectors = 1000;
        let num_dimensions = 100;
        let sparsity = 0.1;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut index = LSHIndex::new(num_buckets, num_hash_functions, LSHHashType::SimHash);

        let mut vectors = Vec::new();

        for _ in 0..num_vectors {
            let mut indices = HashSet::new();
            while indices.len() < (num_dimensions as f32 * sparsity) as usize {
                indices.insert(rng.gen_range(0..num_dimensions));
            }

            let values: Vec<OrderedFloat<f32>> =
                indices.iter().map(|_| OrderedFloat(rng.gen())).collect();

            let vector = SparseVector {
                indices: indices.into_iter().collect(),
                values,
            };

            vectors.push(vector.clone());
            index.add_vector(&vector);
        }

        let query_vector = vectors[500].clone();

        let results = index.search(&query_vector, 10);

        for result in &results {
            println!("{:?}", result);
        }

        assert_eq!(results.len(), 10);

        assert_eq!(results[0].index, 500);
        assert_eq!(results[0].score.into_inner(), 1.0);

        for result in results.iter().skip(1) {
            assert!(result.score.into_inner() > 0.0);
            assert!(result.index != 500);
        }

        assert!(true);
    }

    #[test]
    fn test_min_hash_index() {
        let seed = 42;
        let num_buckets = 40;
        let num_hash_functions = 10;
        let num_vectors = 1000;
        let num_dimensions = 100;
        let sparsity = 0.1;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut index = LSHIndex::new(num_buckets, num_hash_functions, LSHHashType::MinHash);

        let mut vectors = Vec::new();

        for _ in 0..num_vectors {
            let mut indices = HashSet::new();
            while indices.len() < (num_dimensions as f32 * sparsity) as usize {
                indices.insert(rng.gen_range(0..num_dimensions));
            }

            let values: Vec<OrderedFloat<f32>> =
                indices.iter().map(|_| OrderedFloat(rng.gen())).collect();

            let vector = SparseVector {
                indices: indices.into_iter().collect(),
                values,
            };

            vectors.push(vector.clone());
            index.add_vector(&vector);
        }

        let query_vector = vectors[500].clone();

        let results = index.search(&query_vector, 10);

        for result in &results {
            println!("{:?}", result);
        }

        assert_eq!(results.len(), 10);

        assert_eq!(results[0].index, 500);
        assert_eq!(results[0].score.into_inner(), 1.0);

        for result in results.iter().skip(1) {
            assert!(result.score.into_inner() > 0.0);
            assert!(result.index != 500);
        }

        assert!(true);
    }

    /*use index_simulation_tool::{
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
            samples.push(HighDimVector::new(i, vec![128.0 + i as f32; 3]));
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
        }*/
}
