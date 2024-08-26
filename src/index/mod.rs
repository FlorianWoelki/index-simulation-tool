use std::{
    fmt::Debug,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
};

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
pub mod nsw;
pub mod pq;

#[derive(PartialEq, Serialize, Deserialize, Copy, Clone)]
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
    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult>;

    fn save(&self, file: &mut File);
    fn load_index(file: &File) -> Self;
}

#[allow(dead_code)]
pub enum IndexType {
    LSH(LSHIndex),
    Annoy(AnnoyIndex),
    PQ(PQIndex),
    IVFPQ(IVFPQIndex),
    HNSW(HNSWIndex),
    NSW(NSWIndex),
    LinScan(LinScanIndex),
}

#[derive(Debug, PartialEq)]
enum IndexIdentifier {
    LSH,
    Annoy,
    PQ,
    IVFPQ,
    HNSW,
    NSW,
    LinScan,
}

impl IndexIdentifier {
    fn to_u32(&self) -> u32 {
        match self {
            IndexIdentifier::LSH => 0,
            IndexIdentifier::Annoy => 1,
            IndexIdentifier::PQ => 2,
            IndexIdentifier::IVFPQ => 3,
            IndexIdentifier::HNSW => 4,
            IndexIdentifier::NSW => 5,
            IndexIdentifier::LinScan => 6,
        }
    }

    fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(IndexIdentifier::LSH),
            1 => Some(IndexIdentifier::Annoy),
            2 => Some(IndexIdentifier::PQ),
            3 => Some(IndexIdentifier::IVFPQ),
            4 => Some(IndexIdentifier::HNSW),
            5 => Some(IndexIdentifier::NSW),
            6 => Some(IndexIdentifier::LinScan),
            _ => None,
        }
    }
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

    fn load_index(file: &File) -> Self {
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(0)).unwrap();

        let mut buffer = [0u8; 4];
        reader
            .read_exact(&mut buffer)
            .expect("Failed to read metadata");
        let index_type =
            IndexIdentifier::from_u32(u32::from_be_bytes(buffer)).expect("Wrong index type");

        match index_type {
            IndexIdentifier::LSH => IndexType::LSH(LSHIndex::load_index(file)),
            IndexIdentifier::Annoy => IndexType::Annoy(AnnoyIndex::load_index(file)),
            IndexIdentifier::PQ => IndexType::PQ(PQIndex::load_index(file)),
            IndexIdentifier::IVFPQ => IndexType::IVFPQ(IVFPQIndex::load_index(file)),
            IndexIdentifier::HNSW => IndexType::HNSW(HNSWIndex::load_index(file)),
            IndexIdentifier::NSW => IndexType::NSW(NSWIndex::load_index(file)),
            IndexIdentifier::LinScan => IndexType::LinScan(LinScanIndex::load_index(file)),
        }
    }
}
