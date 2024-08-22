#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use index_simulation_tool::{
        data::SparseVector,
        index::{
            annoy::AnnoyIndex,
            hnsw::HNSWIndex,
            ivfpq::IVFPQIndex,
            linscan::LinScanIndex,
            lsh::{LSHHashType, LSHIndex},
            nsw::NSWIndex,
            pq::PQIndex,
            DistanceMetric, IndexType, SparseIndex,
        },
    };
    use ordered_float::OrderedFloat;
    use rand::{Rng, SeedableRng};

    const SEED: u64 = 42;

    fn test_index(index: &mut IndexType) {
        let num_vectors = 1000;
        let num_dimensions = 100;
        let sparsity = 0.1;
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

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
            index.add_vector_before_build(&vector);
        }

        index.build();

        let query_vector = vectors[500].clone();

        let results = index.search(&query_vector, 10);

        assert_eq!(results.len(), 10);
        assert_eq!(results[0].index, 500);
        // assert_eq!(results[0].score.into_inner(), 0.0);

        for result in results.iter().skip(1) {
            // assert!(result.score.into_inner() > 0.0);
            assert!(result.index != 500);
        }
    }

    #[test]
    fn test_lsh_index() {
        let num_buckets = 10;
        let num_hash_functions = 10;
        let mut index = IndexType::LSH(LSHIndex::new(
            num_buckets,
            num_hash_functions,
            LSHHashType::SimHash,
            DistanceMetric::Cosine,
        ));
        test_index(&mut index);

        let mut index = IndexType::LSH(LSHIndex::new(
            num_buckets,
            num_hash_functions,
            LSHHashType::MinHash,
            DistanceMetric::Cosine,
        ));
        test_index(&mut index);
    }

    #[test]
    fn test_annoy_index() {
        let n_trees = 10;
        let max_points = 10;
        let search_k = 200;
        let mut index = IndexType::Annoy(AnnoyIndex::new(
            n_trees,
            max_points,
            search_k,
            DistanceMetric::Cosine,
        ));
        test_index(&mut index);
    }

    #[test]
    fn test_pq_index() {
        let num_subvectors = 3;
        let num_clusters = 50;
        let iterations = 256;
        let tolerance = 0.01;
        let mut index = IndexType::PQ(PQIndex::new(
            num_subvectors,
            num_clusters,
            iterations,
            tolerance,
            DistanceMetric::Cosine,
            SEED,
        ));
        test_index(&mut index);
    }

    #[test]
    fn test_ivfpq_index() {
        let num_subvectors = 3;
        let num_clusters = 50;
        let num_coarse_clusters = 100;
        let iterations = 256;
        let tolerance = 0.01;
        let mut index = IndexType::IVFPQ(IVFPQIndex::new(
            num_subvectors,
            num_clusters,
            num_coarse_clusters,
            iterations,
            tolerance,
            DistanceMetric::Cosine,
            SEED,
        ));
        test_index(&mut index);
    }

    #[test]
    fn test_hnsw_index() {
        let level_distribution_factor = 0.5;
        let max_layers = 8;
        let ef_construction = 50;
        let ef_search = 50;

        let mut index = IndexType::HNSW(HNSWIndex::new(
            level_distribution_factor,
            max_layers,
            ef_construction,
            ef_search,
            DistanceMetric::Cosine,
            SEED,
        ));
        test_index(&mut index);
    }

    #[test]
    fn test_nsw_index() {
        let ef_construction = 50;
        let ef_search = 50;

        let mut index = IndexType::NSW(NSWIndex::new(
            ef_construction,
            ef_search,
            DistanceMetric::Cosine,
            SEED,
        ));
        test_index(&mut index);
    }

    #[test]
    fn test_linscan_index() {
        let mut index = IndexType::LinScan(LinScanIndex::new(DistanceMetric::Cosine));
        test_index(&mut index);
    }
}
