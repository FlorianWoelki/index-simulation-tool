use ordered_float::OrderedFloat;

use crate::{
    data::{QueryResult, SparseVector},
    data_structures::min_heap::MinHeap,
    kmeans::kmeans,
};

use super::DistanceMetric;

pub struct IVFPQIndex {
    num_subvectors: usize,
    num_clusters: usize,
    num_coarse_clusters: usize,
    vectors: Vec<SparseVector>,
    coarse_centroids: Vec<SparseVector>,
    sub_quantizers: Vec<Vec<Vec<SparseVector>>>,
    coarse_codes: Vec<usize>,
    pq_codes: Vec<Vec<usize>>,
    iterations: usize,
    tolerance: f32,
    metric: DistanceMetric,
    random_seed: u64,
}

impl IVFPQIndex {
    pub fn new(
        num_subvectors: usize,
        num_clusters: usize,
        num_coarse_clusters: usize,
        iterations: usize,
        tolerance: f32,
        metric: DistanceMetric,
        random_seed: u64,
    ) -> Self {
        IVFPQIndex {
            num_subvectors,
            num_clusters,
            num_coarse_clusters,
            random_seed,
            iterations,
            tolerance,
            metric,
            vectors: Vec::new(),
            coarse_centroids: Vec::new(),
            sub_quantizers: Vec::new(),
            coarse_codes: Vec::new(),
            pq_codes: Vec::new(),
        }
    }

    pub fn add_vector(&mut self, vector: &SparseVector) {
        self.vectors.push(vector.clone());
    }

    pub fn build(&mut self) {
        // Perform coarse quantization using k-means clustering while assigning
        // each vector to the nearest centroid.
        self.coarse_centroids = kmeans(
            &self.vectors,
            self.num_coarse_clusters,
            self.iterations,
            self.tolerance,
            self.random_seed,
        );
        self.coarse_codes = self.encode_coarse(&self.vectors);

        // Perform product quantization within each Voronoi cell where each sub-vector
        // is quantized separately.
        self.sub_quantizers = vec![Vec::new(); self.num_coarse_clusters];
        for c in 0..self.num_coarse_clusters {
            let mut cluster_vectors: Vec<SparseVector> = Vec::new();
            for (i, &code) in self.coarse_codes.iter().enumerate() {
                if code == c {
                    cluster_vectors.push(self.vectors[i].clone());
                }
            }

            if cluster_vectors.is_empty() {
                continue;
            }

            let mut cluster_codewords = Vec::new();
            for m in 0..self.num_subvectors {
                let mut sub_vectors_m: Vec<SparseVector> = Vec::new();
                for vec in &cluster_vectors {
                    let sub_vec_dims = vec.indices.len() / self.num_subvectors;
                    let start_idx = m * sub_vec_dims;
                    let end_idx = ((m + 1) * sub_vec_dims).min(vec.indices.len());
                    let indices = vec.indices[start_idx..end_idx].to_vec();
                    let values = vec.values[start_idx..end_idx].to_vec();
                    sub_vectors_m.push(SparseVector { indices, values });
                }

                let codewords_m = kmeans(
                    &sub_vectors_m,
                    self.num_clusters,
                    self.iterations,
                    self.tolerance,
                    self.random_seed,
                );
                cluster_codewords.push(codewords_m);
            }
            self.sub_quantizers[c] = cluster_codewords;
        }

        self.pq_codes = self.encode(&self.vectors);
    }

    pub fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        let mut coarse_distances: Vec<(usize, f32)> = Vec::new();
        for (i, coarse_codeword) in self.coarse_centroids.iter().enumerate() {
            let distance = query_vector.distance(&coarse_codeword, &self.metric);
            coarse_distances.push((i, distance));
        }
        coarse_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut heap: MinHeap<QueryResult> = MinHeap::new();
        let sub_vec_dims = query_vector.indices.len() / self.num_subvectors;

        for &(coarse_index, _) in coarse_distances.iter().take(k) {
            for (n, &coarse_code) in self.coarse_codes.iter().enumerate() {
                if coarse_code != coarse_index {
                    continue;
                }

                let mut distance = 0.0;
                for m in 0..self.num_subvectors {
                    let start_idx = m * sub_vec_dims;
                    let end_idx = ((m + 1) * sub_vec_dims).min(query_vector.indices.len());
                    let query_sub_indices = query_vector.indices[start_idx..end_idx].to_vec();
                    let query_sub_values = query_vector.values[start_idx..end_idx].to_vec();

                    let query_sub = SparseVector {
                        indices: query_sub_indices,
                        values: query_sub_values,
                    };
                    let sub_distance = &query_sub.distance(
                        &self.sub_quantizers[coarse_index][m][self.pq_codes[n][m]],
                        &self.metric,
                    );
                    distance += sub_distance;
                }

                if heap.len() < k || distance < heap.peek().unwrap().score.into_inner() {
                    heap.push(
                        QueryResult {
                            index: n,
                            score: OrderedFloat(distance),
                        },
                        OrderedFloat(-distance),
                    );
                    if heap.len() > k {
                        heap.pop();
                    }
                }
            }
        }

        heap.into_sorted_vec()
            .iter()
            .map(|query_result| QueryResult {
                index: query_result.index,
                score: OrderedFloat(query_result.score.into_inner()),
            })
            .collect()
    }

    fn encode_coarse(&self, vectors: &Vec<SparseVector>) -> Vec<usize> {
        let mut coarse_codes: Vec<usize> = Vec::new();
        for vec in vectors {
            let mut min_distance = f32::MAX;
            let mut min_distance_code_index = 0;

            for (i, coarse_codeword) in self.coarse_centroids.iter().enumerate() {
                let distance = vec.distance(&coarse_codeword, &self.metric);
                if distance < min_distance {
                    min_distance = distance;
                    min_distance_code_index = i;
                }
            }

            coarse_codes.push(min_distance_code_index);
        }

        coarse_codes
    }

    fn encode(&self, vectors: &Vec<SparseVector>) -> Vec<Vec<usize>> {
        let mut vector_codes: Vec<Vec<usize>> = Vec::new();
        for (i, vec) in vectors.iter().enumerate() {
            let coarse_code = self.coarse_codes[i];
            let sub_vec_dims: usize = vec.indices.len() / self.num_subvectors;
            let mut subvectors: Vec<SparseVector> = Vec::new();
            for m in 0..self.num_subvectors {
                let start_idx = m * sub_vec_dims;
                let end_idx = ((m + 1) * sub_vec_dims).min(vec.indices.len());
                let indices = vec.indices[start_idx..end_idx].to_vec();
                let values = vec.values[start_idx..end_idx].to_vec();
                subvectors.push(SparseVector { indices, values });
            }
            vector_codes.push(self.vector_quantize(&subvectors, coarse_code));
        }

        vector_codes
    }

    fn vector_quantize(&self, vectors: &[SparseVector], coarse_code: usize) -> Vec<usize> {
        let mut codes: Vec<usize> = Vec::new();

        for (m, subvector) in vectors.iter().enumerate() {
            let mut min_distance = f32::MAX;
            let mut min_distance_code_index = 0;

            for (k, code) in self.sub_quantizers[coarse_code][m].iter().enumerate() {
                let distance = subvector.distance(&code, &self.metric);
                if distance < min_distance {
                    min_distance = distance;
                    min_distance_code_index = k;
                }
            }

            codes.push(min_distance_code_index);
        }

        codes
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;
    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn test_ivfpq_index_simple() {
        let random_seed = 42;
        let num_subvectors = 2;
        let num_clusters = 3;
        let num_coarse_clusters = 2;
        let iterations = 10;
        let mut index = IVFPQIndex::new(
            num_subvectors,
            num_clusters,
            num_coarse_clusters,
            iterations,
            0.01,
            DistanceMetric::Euclidean,
            random_seed,
        );

        let vectors = vec![
            SparseVector {
                indices: vec![0, 1, 2],
                values: vec![OrderedFloat(0.1), OrderedFloat(0.2), OrderedFloat(0.3)],
            },
            SparseVector {
                indices: vec![1, 2, 3],
                values: vec![OrderedFloat(0.4), OrderedFloat(0.5), OrderedFloat(0.6)],
            },
            SparseVector {
                indices: vec![2, 3, 4],
                values: vec![OrderedFloat(0.7), OrderedFloat(0.8), OrderedFloat(0.9)],
            },
            SparseVector {
                indices: vec![3, 4, 5],
                values: vec![OrderedFloat(1.0), OrderedFloat(1.1), OrderedFloat(1.2)],
            },
            SparseVector {
                indices: vec![4, 5, 6],
                values: vec![OrderedFloat(1.3), OrderedFloat(1.4), OrderedFloat(1.5)],
            },
        ];

        for vector in &vectors {
            index.add_vector(vector);
        }

        index.build();

        let results = index.search(&vectors[1], 3);

        println!("{:?}", results);
        assert!(true);
    }

    #[test]
    fn test_ivfpq_index_complex() {
        let random_seed = thread_rng().gen::<u64>();
        let num_subvectors = 2;
        let num_clusters = 10;
        let num_coarse_clusters = 40;
        let iterations = 256;
        let mut index = IVFPQIndex::new(
            num_subvectors,
            num_clusters,
            num_coarse_clusters,
            iterations,
            0.01,
            DistanceMetric::Euclidean,
            random_seed,
        );

        let mut vectors = vec![];
        for i in 0..100 {
            vectors.push(SparseVector {
                indices: vec![i % 10, (i / 10) % 10],
                values: vec![OrderedFloat((i % 10) as f32), OrderedFloat((i / 10) as f32)],
            });
        }

        for vector in &vectors {
            index.add_vector(vector);
        }

        index.build();

        let query_vector = SparseVector {
            indices: vec![5, 9],
            values: vec![OrderedFloat(5.0), OrderedFloat(9.0)],
        };
        let results = index.search(&query_vector, 10);
        println!("Results for search on query vector: {:?}", results);
        println!("Top Search: {:?}", vectors[results[0].index]);
        println!("Groundtruth: {:?}", query_vector);

        assert!(false);
    }
}
