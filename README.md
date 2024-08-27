## Experiments

- [ ] Having a timeout (5min) that terminates the current run
-> use `thread::spawn` and `mpsc::channel`

- [ ] Single-Threaded experiments
  - [ ] Using PCA
  - [ ] Using t-sne
  - [ ] Using Euclidean distance metric
  - [ ] Using Cosine distance metric
  - [ ] Using Jaccard distance metric
  - [ ] Using Angular distance metric
- [ ] Multi-Threaded experiments
  - [ ] Using PCA
  - [ ] Using t-sne
  - [ ] Using Euclidean distance metric
  - [ ] Using Cosine distance metric
  - [ ] Using Jaccard distance metric
  - [ ] Using Angular distance metric

- [ ] Multi-Threaded experiments with MS MARCO

## Optional

- [ ] Maybe convert back to f64 because msmarco is in f64
- [ ] Convert dense vectors to sparse vectors -> use something like SPLADE to create a sparse vector
  - Maybe some python to Rust embeddings?
  - Maybe use the python scripts to generate the custom datasets?
