use std::{fmt::Debug, fs::File};

use annoy::AnnoyIndex;
use hnsw::HNSWIndex;
use ivfpq::IVFPQIndex;
use linscan::LinScanIndex;
use lsh::LSHIndex;
use nsw::NSWIndex;
use pq::PQIndex;
use serde::{Deserialize, Serialize};

use crate::data::{QueryResult, SparseVector};

pub mod annoy;
pub mod hnsw;
pub mod ivfpq;
pub mod linscan;
pub mod lsh;
pub mod neighbor;
pub mod nsw;
pub mod pq;

#[derive(PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    Jaccard,
    Angular,
}

impl Debug for DistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceMetric::Euclidean => write!(f, "Euclidean"),
            DistanceMetric::Cosine => write!(f, "Cosine"),
            DistanceMetric::Jaccard => write!(f, "Jaccard"),
            DistanceMetric::Angular => write!(f, "Angular"),
        }
    }
}

pub trait SparseIndex {
    fn add_vector_before_build(&mut self, vector: &SparseVector);
    fn add_vector(&mut self, vector: &SparseVector);
    fn remove_vector(&mut self, id: usize) -> Option<SparseVector>;
    fn build(&mut self);
    fn build_parallel(&mut self);
    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult>;
    fn search_parallel(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult>;

    fn save(&self, file: &mut File);
    fn load(&self, file: &File) -> Self;
}

pub enum IndexType {
    LSH(LSHIndex),
    Annoy(AnnoyIndex),
    PQ(PQIndex),
    IVFPQ(IVFPQIndex),
    HNSW(HNSWIndex),
    NSW(NSWIndex),
    LinScan(LinScanIndex),
}

impl SparseIndex for IndexType {
    fn add_vector_before_build(&mut self, vector: &SparseVector) {
        match self {
            IndexType::LSH(index) => index.add_vector_before_build(vector),
            IndexType::Annoy(index) => index.add_vector_before_build(vector),
            IndexType::PQ(index) => index.add_vector_before_build(vector),
            IndexType::IVFPQ(index) => index.add_vector_before_build(vector),
            IndexType::HNSW(index) => index.add_vector_before_build(vector),
            IndexType::NSW(index) => index.add_vector_before_build(vector),
            IndexType::LinScan(index) => index.add_vector_before_build(vector),
        }
    }

    fn add_vector(&mut self, vector: &SparseVector) {
        match self {
            IndexType::LSH(index) => index.add_vector(vector),
            IndexType::Annoy(index) => index.add_vector(vector),
            IndexType::PQ(index) => index.add_vector(vector),
            IndexType::IVFPQ(index) => index.add_vector(vector),
            IndexType::HNSW(index) => index.add_vector(vector),
            IndexType::NSW(index) => index.add_vector(vector),
            IndexType::LinScan(index) => index.add_vector(vector),
        }
    }

    fn remove_vector(&mut self, id: usize) -> Option<SparseVector> {
        match self {
            IndexType::LSH(index) => index.remove_vector(id),
            IndexType::Annoy(index) => index.remove_vector(id),
            IndexType::PQ(index) => index.remove_vector(id),
            IndexType::IVFPQ(index) => index.remove_vector(id),
            IndexType::HNSW(index) => index.remove_vector(id),
            IndexType::NSW(index) => index.remove_vector(id),
            IndexType::LinScan(index) => index.remove_vector(id),
        }
    }

    fn build(&mut self) {
        match self {
            IndexType::LSH(index) => index.build(),
            IndexType::Annoy(index) => index.build(),
            IndexType::PQ(index) => index.build(),
            IndexType::IVFPQ(index) => index.build(),
            IndexType::HNSW(index) => index.build(),
            IndexType::NSW(index) => index.build(),
            IndexType::LinScan(index) => index.build(),
        }
    }

    fn build_parallel(&mut self) {
        match self {
            IndexType::LSH(index) => index.build_parallel(),
            IndexType::Annoy(index) => index.build_parallel(),
            IndexType::PQ(index) => index.build_parallel(),
            IndexType::IVFPQ(index) => index.build_parallel(),
            IndexType::HNSW(index) => index.build_parallel(),
            IndexType::NSW(index) => index.build_parallel(),
            IndexType::LinScan(index) => index.build(),
        }
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        match self {
            IndexType::LSH(index) => index.search(query_vector, k),
            IndexType::Annoy(index) => index.search(query_vector, k),
            IndexType::PQ(index) => index.search(query_vector, k),
            IndexType::IVFPQ(index) => index.search(query_vector, k),
            IndexType::HNSW(index) => index.search(query_vector, k),
            IndexType::NSW(index) => index.search(query_vector, k),
            IndexType::LinScan(index) => index.search(query_vector, k),
        }
    }

    fn search_parallel(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        match self {
            IndexType::LSH(index) => index.search_parallel(query_vector, k),
            IndexType::Annoy(index) => index.search_parallel(query_vector, k),
            IndexType::PQ(index) => index.search_parallel(query_vector, k),
            IndexType::IVFPQ(index) => index.search_parallel(query_vector, k),
            IndexType::HNSW(index) => index.search_parallel(query_vector, k),
            IndexType::NSW(index) => index.search_parallel(query_vector, k),
            IndexType::LinScan(index) => index.search(query_vector, k),
        }
    }

    fn save(&self, file: &mut File) {
        match self {
            IndexType::LSH(index) => index.save(file),
            IndexType::Annoy(index) => index.save(file),
            IndexType::PQ(index) => index.save(file),
            IndexType::IVFPQ(index) => index.save(file),
            IndexType::HNSW(index) => index.save(file),
            IndexType::NSW(index) => index.save(file),
            IndexType::LinScan(index) => index.save(file),
        }
    }

    fn load(&self, file: &File) -> Self {
        match self {
            IndexType::LSH(index) => IndexType::LSH(index.load(file)),
            IndexType::Annoy(index) => IndexType::Annoy(index.load(file)),
            IndexType::PQ(index) => IndexType::PQ(index.load(file)),
            IndexType::IVFPQ(index) => IndexType::IVFPQ(index.load(file)),
            IndexType::HNSW(index) => IndexType::HNSW(index.load(file)),
            IndexType::NSW(index) => IndexType::NSW(index.load(file)),
            IndexType::LinScan(index) => IndexType::LinScan(index.load(file)),
        }
    }
}
