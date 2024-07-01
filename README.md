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
- [ ] Cleanup repository
  - [ ] Use similar functionalities for multi threaded and single threaded functionality

- [ ] Develop async functionalities for each algorithm
  - [ ] HNSW
  - [ ] NSW
  - [x] ANNOY
  - [x] LinScan
  - [x] LSH
  - [x] IVFPQ
  - [x] PQ

## Experiments

- [ ] Define experiments that are going to be created
- [ ] Experiment using different distance metrics
- [ ] Experiment using different distance and similarity measures

## Optional

- [ ] Maybe convert back to f64 because msmarco is in f64
- [ ] Convert dense vectors to sparse vectors -> use something like SPLADE to create a sparse vector
  - Maybe some python to Rust embeddings?
  - Maybe use the python scripts to generate the custom datasets?

## Plan for implement async functionalities

- Use rayon to execute things in parallel

- Add search_parallel function to search in parallel
- Add build_parallel, if applicable and possible
