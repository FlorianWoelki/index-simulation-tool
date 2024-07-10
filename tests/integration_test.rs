#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use index_simulation_tool::{
        data::SparseVector,
        index::{
            annoy::AnnoyIndex,
            lsh::{LSHHashType, LSHIndex},
            DistanceMetric, SparseIndex,
        },
    };
    use ordered_float::OrderedFloat;
    use rand::{Rng, SeedableRng};

    fn test_index(seed: u64, index: &mut dyn SparseIndex) {
        let num_vectors = 1000;
        let num_dimensions = 100;
        let sparsity = 0.1;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

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

        index.build();

        let query_vector = vectors[500].clone();

        let results = index.search(&query_vector, 10);

        assert_eq!(results.len(), 10);
        assert_eq!(results[0].index, 500);
        assert_eq!(results[0].score.into_inner(), 0.0);

        for result in results.iter().skip(1) {
            assert!(result.score.into_inner() > 0.0);
            assert!(result.index != 500);
        }
    }

    #[test]
    fn test_lsh_index() {
        let seed = 42;
        let num_buckets = 10;
        let num_hash_functions = 10;
        let mut index = LSHIndex::new(
            num_buckets,
            num_hash_functions,
            LSHHashType::SimHash,
            DistanceMetric::Cosine,
        );
        test_index(seed, &mut index);

        let mut index = LSHIndex::new(
            num_buckets,
            num_hash_functions,
            LSHHashType::MinHash,
            DistanceMetric::Cosine,
        );
        test_index(seed, &mut index);
    }

    #[test]
    fn test_annoy_index() {
        let seed = 42;
        let n_trees = 10;
        let max_points = 10;
        let search_k = 20;
        let mut index = AnnoyIndex::new(n_trees, max_points, search_k, DistanceMetric::Cosine);
        test_index(seed, &mut index);
    }
}
