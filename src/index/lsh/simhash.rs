use std::{
    collections::HashSet,
    hash::{DefaultHasher, Hash, Hasher},
};

use crate::data::vector::SparseVector;

pub(super) fn simhash(vector: &SparseVector, hash_function_index: usize) -> u64 {
    let mut v = [0i32; 64];
    let mut simhash: u64 = 0;

    let mut hasher = DefaultHasher::new();
    for (&index, &value) in vector.indices.iter().zip(vector.values.iter()) {
        let mut element_hash = HashSet::new();
        element_hash.insert(index);
        element_hash.insert(value.to_bits() as usize);

        for item in element_hash {
            item.hash(&mut hasher);
        }
    }
    let hash = hasher.finish() ^ (hash_function_index as u64);

    for i in 0..64 {
        let bit = (hash >> i) & 1;
        if bit == 1 {
            v[i] = v[i].saturating_add(1);
        } else {
            v[i] = v[i].saturating_sub(1);
        }
    }

    for q in 0..64 {
        if v[q] > 0 {
            simhash |= 1 << q;
        }
    }

    simhash
}
