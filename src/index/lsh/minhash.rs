use std::hash::{DefaultHasher, Hash, Hasher};

use crate::data::SparseVector;

pub(super) fn minhash(vector: &SparseVector, hash_function_index: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    let mut min_hash: u64 = u64::MAX;

    for (&index, &value) in vector.indices.iter().zip(vector.values.iter()) {
        let mut combined_hash = hash_function_index as u64;
        index.hash(&mut hasher);
        combined_hash = combined_hash.wrapping_mul(hasher.finish());
        value.hash(&mut hasher);
        combined_hash = combined_hash.wrapping_mul(hasher.finish());

        min_hash = min_hash.min(combined_hash);
        hasher = DefaultHasher::new();
        // let mut element_hash = HashSet::new();
        // element_hash.insert(index);
        // element_hash.insert(value.to_bits() as usize);

        // for item in element_hash {
        //     item.hash(&mut hasher);
        // }

        // let hash = hasher.finish() ^ (hash_function_index as u64);
        // min_hash = min_hash.min(hash);
        // hasher = DefaultHasher::new();
    }

    min_hash
}
