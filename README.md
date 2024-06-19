# TO DO

- [x] Create sparse data generator
- [x] Adjust artificially data generator for sparse data
- [x] Implement MinHash
- [x] Implement SimHash
- [x] Implement LinScan
- [x] Implement HNSW Sparse
- [x] Implement Annoy
- [x] Implement IVFPQ
  - [ ] Investigate correctness of IVFPQ Index
- [x] Implement PQ Index
  - [x] Investigate correctness of PQ index
- [ ] Implement dimensionality reduction mechanisms
  - [x] PCA
  - [ ] t-SNE
- [ ] Cleanup repository

## Experiments

- [ ] Define experiments that are going to be created
- [ ] Experiment using different distance metrics
- [ ] Experiment using different distance and similarity measures

## Optional

- [ ] Maybe convert back to f64 because msmarco is in f64
- [ ] Convert dense vectors to sparse vectors -> use something like SPLADE to create a sparse vector
  - Maybe some python to Rust embeddings?
  - Maybe use the python scripts to generate the custom datasets?
