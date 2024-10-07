use std::{
    fmt::Debug,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    str::FromStr,
};

use annoy::AnnoyIndex;
use hnsw::HNSWIndex;
use ivfpq::IVFPQIndex;
use linscan::LinScanIndex;
use lsh::LSHIndex;
use nsw::NSWIndex;
use pq::PQIndex;
use serde::{Deserialize, Serialize};

use crate::data::{vector::SparseVector, QueryResult};

pub mod annoy;
pub mod hnsw;
pub mod ivfpq;
pub mod linscan;
pub mod lsh;
pub mod nsw;
pub mod pq;

#[derive(PartialEq, Serialize, Deserialize, Copy, Clone)]
pub enum DistanceMetric {
    Dot,
    Euclidean,
    Cosine,
    Jaccard,
    Angular,
}

impl FromStr for DistanceMetric {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "dot" => Ok(DistanceMetric::Dot),
            "euclidean" => Ok(DistanceMetric::Euclidean),
            "cosine" => Ok(DistanceMetric::Cosine),
            "jaccard" => Ok(DistanceMetric::Jaccard),
            "angular" => Ok(DistanceMetric::Angular),
            _ => Err(format!("Unknown distance metric: {}", s)),
        }
    }
}

impl Debug for DistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceMetric::Dot => write!(f, "Dot"),
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
#[derive(Clone, Debug)]
pub enum IndexType {
    Lsh(LSHIndex),
    Annoy(AnnoyIndex),
    Pq(PQIndex),
    Ivfpq(IVFPQIndex),
    Hnsw(HNSWIndex),
    Nsw(NSWIndex),
    LinScan(LinScanIndex),
}

#[derive(Debug, PartialEq)]
enum IndexIdentifier {
    Lsh,
    Annoy,
    Pq,
    Ivfpq,
    Hnsw,
    Nsw,
    LinScan,
}

impl IndexIdentifier {
    fn to_u32(&self) -> u32 {
        match self {
            IndexIdentifier::Lsh => 0,
            IndexIdentifier::Annoy => 1,
            IndexIdentifier::Pq => 2,
            IndexIdentifier::Ivfpq => 3,
            IndexIdentifier::Hnsw => 4,
            IndexIdentifier::Nsw => 5,
            IndexIdentifier::LinScan => 6,
        }
    }

    fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(IndexIdentifier::Lsh),
            1 => Some(IndexIdentifier::Annoy),
            2 => Some(IndexIdentifier::Pq),
            3 => Some(IndexIdentifier::Ivfpq),
            4 => Some(IndexIdentifier::Hnsw),
            5 => Some(IndexIdentifier::Nsw),
            6 => Some(IndexIdentifier::LinScan),
            _ => None,
        }
    }
}

impl SparseIndex for IndexType {
    fn add_vector_before_build(&mut self, vector: &SparseVector) {
        match self {
            IndexType::Lsh(index) => index.add_vector_before_build(vector),
            IndexType::Annoy(index) => index.add_vector_before_build(vector),
            IndexType::Pq(index) => index.add_vector_before_build(vector),
            IndexType::Ivfpq(index) => index.add_vector_before_build(vector),
            IndexType::Hnsw(index) => index.add_vector_before_build(vector),
            IndexType::Nsw(index) => index.add_vector_before_build(vector),
            IndexType::LinScan(index) => index.add_vector_before_build(vector),
        }
    }

    fn add_vector(&mut self, vector: &SparseVector) {
        match self {
            IndexType::Lsh(index) => index.add_vector(vector),
            IndexType::Annoy(index) => index.add_vector(vector),
            IndexType::Pq(index) => index.add_vector(vector),
            IndexType::Ivfpq(index) => index.add_vector(vector),
            IndexType::Hnsw(index) => index.add_vector(vector),
            IndexType::Nsw(index) => index.add_vector(vector),
            IndexType::LinScan(index) => index.add_vector(vector),
        }
    }

    fn remove_vector(&mut self, id: usize) -> Option<SparseVector> {
        match self {
            IndexType::Lsh(index) => index.remove_vector(id),
            IndexType::Annoy(index) => index.remove_vector(id),
            IndexType::Pq(index) => index.remove_vector(id),
            IndexType::Ivfpq(index) => index.remove_vector(id),
            IndexType::Hnsw(index) => index.remove_vector(id),
            IndexType::Nsw(index) => index.remove_vector(id),
            IndexType::LinScan(index) => index.remove_vector(id),
        }
    }

    fn build(&mut self) {
        match self {
            IndexType::Lsh(index) => index.build(),
            IndexType::Annoy(index) => index.build(),
            IndexType::Pq(index) => index.build(),
            IndexType::Ivfpq(index) => index.build(),
            IndexType::Hnsw(index) => index.build(),
            IndexType::Nsw(index) => index.build(),
            IndexType::LinScan(index) => index.build(),
        }
    }

    fn search(&self, query_vector: &SparseVector, k: usize) -> Vec<QueryResult> {
        match self {
            IndexType::Lsh(index) => index.search(query_vector, k),
            IndexType::Annoy(index) => index.search(query_vector, k),
            IndexType::Pq(index) => index.search(query_vector, k),
            IndexType::Ivfpq(index) => index.search(query_vector, k),
            IndexType::Hnsw(index) => index.search(query_vector, k),
            IndexType::Nsw(index) => index.search(query_vector, k),
            IndexType::LinScan(index) => index.search(query_vector, k),
        }
    }

    fn save(&self, file: &mut File) {
        match self {
            IndexType::Lsh(index) => index.save(file),
            IndexType::Annoy(index) => index.save(file),
            IndexType::Pq(index) => index.save(file),
            IndexType::Ivfpq(index) => index.save(file),
            IndexType::Hnsw(index) => index.save(file),
            IndexType::Nsw(index) => index.save(file),
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
            IndexIdentifier::Lsh => IndexType::Lsh(LSHIndex::load_index(file)),
            IndexIdentifier::Annoy => IndexType::Annoy(AnnoyIndex::load_index(file)),
            IndexIdentifier::Pq => IndexType::Pq(PQIndex::load_index(file)),
            IndexIdentifier::Ivfpq => IndexType::Ivfpq(IVFPQIndex::load_index(file)),
            IndexIdentifier::Hnsw => IndexType::Hnsw(HNSWIndex::load_index(file)),
            IndexIdentifier::Nsw => IndexType::Nsw(NSWIndex::load_index(file)),
            IndexIdentifier::LinScan => IndexType::LinScan(LinScanIndex::load_index(file)),
        }
    }
}
