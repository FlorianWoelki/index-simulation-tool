# TO DO

- [x] Create sparse data generator
- [x] Adjust artificially data generator for sparse data
- [x] Implement MinHash
- [x] Implement SimHash
- [x] Implement LinScan
- [x] Implement HNSW Sparse
- [x] Implement Annoy
- [x] Implement IVFPQ
- [x] Implement PQ Index
- [x] Combine MinHash and SimHash into one LSH module
- [x] Implement dimensionality reduction mechanisms
  - [x] PCA
  - [x] t-SNE
    - [x] Remove nalgebra and use pure ndarray for PCA
- [x] Implement saving/loading functionality
- [ ] Cleanup repository
  - [ ] Use similar functionalities for multi threaded and single threaded functionality
   -> maybe use something like:

   ```rust
   // Initialize global pool with number of threads.
   rayon::ThreadPoolBuilder::new()
       .num_threads(1)
       .build_global()
       .unwrap();
   ```

- [ ] Investigate why IVFPQ is so slow

## Experiments

- [ ] Define experiments that are going to be created
  - [ ] Single-Threaded experiment
  - [ ] Multi-Threaded experiment
- [ ] Experiment using different distance metrics
- [ ] Experiment using different distance and similarity measures

## Optional

- [ ] Maybe convert back to f64 because msmarco is in f64
- [ ] Convert dense vectors to sparse vectors -> use something like SPLADE to create a sparse vector
  - Maybe some python to Rust embeddings?
  - Maybe use the python scripts to generate the custom datasets?
