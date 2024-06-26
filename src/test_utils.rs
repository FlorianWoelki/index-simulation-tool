use ordered_float::OrderedFloat;

use crate::data::SparseVector;

/// Returns a tuple where the first entry is easy to index vector data
/// and the second entry are possible query vectors to use.
pub fn get_simple_vectors() -> (Vec<SparseVector>, Vec<SparseVector>) {
    return (
        vec![
            SparseVector {
                indices: vec![0, 2],
                values: vec![OrderedFloat(1.0), OrderedFloat(2.0)],
            },
            SparseVector {
                indices: vec![1, 3],
                values: vec![OrderedFloat(3.0), OrderedFloat(4.0)],
            },
            SparseVector {
                indices: vec![0, 2],
                values: vec![OrderedFloat(5.0), OrderedFloat(6.0)],
            },
            SparseVector {
                indices: vec![1, 3],
                values: vec![OrderedFloat(7.0), OrderedFloat(8.0)],
            },
            SparseVector {
                indices: vec![0, 2],
                values: vec![OrderedFloat(9.0), OrderedFloat(10.0)],
            },
        ],
        vec![
            SparseVector {
                indices: vec![0, 2],
                values: vec![OrderedFloat(6.0), OrderedFloat(7.0)],
            },
            SparseVector {
                indices: vec![0, 2],
                values: vec![OrderedFloat(2.0), OrderedFloat(3.0)],
            },
        ],
    );
}

pub fn get_complex_vectors() -> (Vec<SparseVector>, SparseVector) {
    let mut vectors = vec![];
    for i in 0..100 {
        vectors.push(SparseVector {
            indices: vec![i % 10, (i / 10) % 10],
            values: vec![OrderedFloat((i % 10) as f32), OrderedFloat((i / 10) as f32)],
        });
    }

    (
        vectors,
        SparseVector {
            indices: vec![5, 9],
            values: vec![OrderedFloat(5.0), OrderedFloat(9.0)],
        },
    )
}
