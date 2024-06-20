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
- [ ] Combine MinHash and SimHash into one LSH module
- [ ] Implement dimensionality reduction mechanisms
  - [x] PCA
  - [ ] t-SNE
- [ ] Cleanup repository

- [ ] Develop async functionalities for each algorithm
  - [ ] HNSW
  - [ ] ANNOY
  - [ ] LinScan
  - [ ] IVFPQ
  - [ ] PQ

## Experiments

- [ ] Define experiments that are going to be created
- [ ] Experiment using different distance metrics
- [ ] Experiment using different distance and similarity measures

## Optional

- [ ] Maybe convert back to f64 because msmarco is in f64
- [ ] Convert dense vectors to sparse vectors -> use something like SPLADE to create a sparse vector
  - Maybe some python to Rust embeddings?
  - Maybe use the python scripts to generate the custom datasets?
