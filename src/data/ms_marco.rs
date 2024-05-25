use std::{
    io::{Error, SeekFrom},
    path::PathBuf,
};

use byteorder::{ByteOrder, LittleEndian};
use sprs::CsMat;
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt},
    task::{self, JoinSet},
};

async fn read_sparse_matrix_fields(
    file_path: &PathBuf,
) -> Result<(Vec<f32>, Vec<usize>, Vec<usize>, usize), Error> {
    let mut file = File::open(file_path).await?;

    let mut sizes = [0u8; 24];
    file.read_exact(&mut sizes).await?;
    let nrow = LittleEndian::read_i64(&sizes[0..8]);
    let ncol = LittleEndian::read_i64(&sizes[8..16]);
    let nnz = LittleEndian::read_i64(&sizes[16..24]);

    let mut indptr = vec![0usize; (nrow + 1) as usize];
    let mut buf = vec![0u8; indptr.len() * 8];
    file.read_exact(&mut buf).await?;
    for i in 0..indptr.len() {
        indptr[i] = LittleEndian::read_i64(&buf[i * 8..(i + 1) * 8]) as usize;
    }
    assert_eq!(nnz, *indptr.last().unwrap() as i64);

    let mut indices = vec![0usize; nnz as usize];
    let mut buf = vec![0u8; indices.len() * 4];
    file.read_exact(&mut buf).await?;
    for i in 0..indices.len() {
        indices[i] = LittleEndian::read_i32(&buf[i * 4..(i + 1) * 4]) as usize;
    }

    let mut data = vec![0f32; nnz as usize];
    let mut buf = vec![0u8; data.len() * 4];
    file.read_exact(&mut buf).await?;
    LittleEndian::read_f32_into(&buf, &mut data);

    Ok((data, indices, indptr, ncol as usize))
}

async fn read_sparse_matrix(file_path: &PathBuf) -> Result<CsMat<f32>, Error> {
    let (data, indices, indptr, ncol) = read_sparse_matrix_fields(file_path).await?;
    Ok(CsMat::new((indptr.len() - 1, ncol), indptr, indices, data))
}

async fn knn_result_read(file_path: &PathBuf) -> Result<(Vec<Vec<i32>>, Vec<Vec<f32>>), Error> {
    // TODO: Maybe change that to match the other data types.
    let mut file = File::open(file_path).await?;

    let mut sizes = [0u8; 8];
    file.read_exact(&mut sizes).await?;
    let n = LittleEndian::read_u32(&sizes[0..4]) as usize;
    let d = LittleEndian::read_u32(&sizes[4..8]) as usize;

    let expected_size = 8 + n * d * (4 + 4);
    assert_eq!(expected_size as u64, file.metadata().await?.len());

    file.seek(SeekFrom::Start(8)).await?;

    let mut ids = vec![vec![0i32; d]; n];
    let mut buf = vec![0u8; n * d * 4];
    file.read_exact(&mut buf).await?;
    for (i, row) in ids.iter_mut().enumerate() {
        for (j, id) in row.iter_mut().enumerate() {
            *id = LittleEndian::read_i32(&buf[(i * d + j) * 4..]);
        }
    }

    let mut scores = vec![vec![0f32; d]; n];
    let mut buf = vec![0u8; n * d * 4];
    file.read_exact(&mut buf).await?;
    for (i, row) in scores.iter_mut().enumerate() {
        for (j, score) in row.iter_mut().enumerate() {
            *score = LittleEndian::read_f32(&buf[(i * d + j) * 4..]);
        }
    }

    Ok((ids, scores))
}

type SparseVectorData = Vec<(Vec<usize>, Vec<f32>)>; // indices, values

// groundtruth, vectors, query_vectors
pub async fn load_msmarco_dataset() -> Result<
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

    let gt_future = task::spawn(async move { knn_result_read(&gt_file).await });
    let vectors_future = task::spawn(async move { read_sparse_matrix(&vectors_file).await });
    let queries_future = task::spawn(async move { read_sparse_matrix(&query_file).await });

    let groundtruth = gt_future.await??;
    let data = vectors_future.await??;
    let query_data = queries_future.await??;

    let mut join_set = JoinSet::new();
    for i in 0..data.rows() {
        let indices = data.outer_view(i).unwrap().indices().to_vec();
        let values = data.outer_view(i).unwrap().data().to_vec();
        join_set.spawn(async move { (indices, values) });
    }

    let mut vectors: SparseVectorData = Vec::with_capacity(data.rows());
    while let Some(res) = join_set.join_next().await {
        vectors.push(res?);
    }

    let mut join_set = JoinSet::new();
    for i in 0..query_data.rows() {
        let indices = query_data.outer_view(i).unwrap().indices().to_vec();
        let values = query_data.outer_view(i).unwrap().data().to_vec();
        join_set.spawn(async move { (indices, values) });
    }

    let mut query_vectors: SparseVectorData = Vec::with_capacity(query_data.rows());
    while let Some(res) = join_set.join_next().await {
        query_vectors.push(res?);
    }

    Ok((groundtruth, vectors, query_vectors))
}
