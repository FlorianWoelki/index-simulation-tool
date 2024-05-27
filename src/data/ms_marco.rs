use std::{
    fs::File,
    io::{BufReader, Error, Seek, SeekFrom},
    path::PathBuf,
};

use byteorder::{LittleEndian, ReadBytesExt};
use sprs::CsMat;

// Dataset details:
// https://github.com/harsha-simhadri/big-ann-benchmarks/blob/1fafd825989fbfca9c7896384ba86a5a359bf388/dataset_preparation/sparse_dataset.md?plain=1#L34

fn read_sparse_matrix_fields(fname: &PathBuf) -> (Vec<f32>, Vec<usize>, Vec<usize>, usize) {
    let mut reader = BufReader::new(File::open(fname).expect("Unable to open file"));

    let mut sizes = [0i64; 3];
    reader
        .read_i64_into::<LittleEndian>(&mut sizes)
        .expect("Unable to read size");
    let (nrow, ncol, nnz) = (sizes[0] as usize, sizes[1] as usize, sizes[2] as usize);

    let mut indptr = vec![0usize; (nrow + 1) as usize];
    for i in &mut indptr {
        *i = reader
            .read_i64::<LittleEndian>()
            .expect("Unable to read indptr") as usize;
    }
    assert_eq!(nnz, *indptr.last().unwrap());

    let mut indices = vec![0usize; nnz];
    for i in &mut indices {
        *i = reader
            .read_i32::<LittleEndian>()
            .expect("Unable to read indices") as usize;
    }

    let mut data = vec![0f32; nnz as usize];
    reader
        .read_f32_into::<LittleEndian>(&mut data)
        .expect("Unable to read data");

    (data, indices, indptr, ncol as usize)
}

fn read_sparse_matrix(fname: &PathBuf) -> CsMat<f32> {
    let (data, indices, indptr, ncol) = read_sparse_matrix_fields(fname);
    CsMat::new((indptr.len() - 1, ncol), indptr, indices, data)
}

fn knn_result_read(fname: &PathBuf) -> (Vec<Vec<i32>>, Vec<Vec<f32>>) {
    let mut file = File::open(fname).expect("Unable to open file");

    let n = file.read_u32::<LittleEndian>().expect("Unable to read n") as usize;
    let d = file.read_u32::<LittleEndian>().expect("Unable to read d") as usize;

    let expected_size = 8 + n * d * (4 + 4);
    assert_eq!(expected_size as u64, file.metadata().unwrap().len());

    file.seek(SeekFrom::Start(8)).unwrap();

    let mut ids = vec![vec![0i32; d]; n];
    for row in &mut ids {
        for id in row {
            *id = file.read_i32::<LittleEndian>().expect("Unable to read id");
        }
    }

    let mut scores = vec![vec![0f32; d]; n];
    for row in &mut scores {
        for score in row {
            *score = file
                .read_f32::<LittleEndian>()
                .expect("Unable to read score");
        }
    }

    (ids, scores)
}

type SparseVectorData = Vec<(Vec<usize>, Vec<f32>)>; // indices, values

// groundtruth, vectors, query_vectors
pub fn load_msmarco_dataset() -> Result<
    (
        (Vec<Vec<i32>>, Vec<Vec<f32>>),
        SparseVectorData,
        SparseVectorData,
    ),
    Error,
> {
    let current_dir = std::env::current_dir()?;
    let msmarco_dir = current_dir.join("src/data/examples/msmarco");

    let gt_file = msmarco_dir.join("base_small.dev.gt");
    let vectors_file = msmarco_dir.join("base_small.csr");
    let query_file = msmarco_dir.join("queries.dev.csr");

    let groundtruth = knn_result_read(&gt_file);
    let vector_data = read_sparse_matrix(&vectors_file);
    let query_data = read_sparse_matrix(&query_file);

    let mut result_vectors = vec![];
    for i in 0..vector_data.rows() {
        let indices = vector_data.outer_view(i).unwrap().indices().to_vec();
        let values = vector_data.outer_view(i).unwrap().data().to_vec();
        result_vectors.push((indices, values));
    }

    let mut query_vectors = vec![];
    for i in 0..query_data.rows() {
        let indices = query_data.outer_view(i).unwrap().indices().to_vec();
        let values = query_data.outer_view(i).unwrap().data().to_vec();
        query_vectors.push((indices, values));
    }

    Ok((groundtruth, result_vectors, query_vectors))
}
