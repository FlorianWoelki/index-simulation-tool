import os
from scipy.sparse import csr_matrix
import numpy as np

def read_sparse_matrix_fields(fname):
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
        indptr = np.fromfile(f, dtype='int64', count=nrow + 1)
        assert nnz == indptr[-1]
        indices = np.fromfile(f, dtype='int32', count=nnz)
        assert np.all(indices >= 0) and np.all(indices < ncol)
        data = np.fromfile(f, dtype='float32', count=nnz)
        return data, indices, indptr, ncol

def read_sparse_matrix(fname):
    data, indices, indptr, ncol = read_sparse_matrix_fields(fname)
    return csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, ncol))

def knn_result_read(fname):
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
    f = open(fname, "rb")
    f.seek(4 + 4)
    ids = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    scores = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return ids, scores

def csr_to_sparse_vector(point_csr):
    indices = point_csr.indices.tolist()
    values = point_csr.data.tolist()
    return indices, values

# Groundtruth: https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_small.dev.gt
gt_indices, gt_scores = knn_result_read("base_small.dev.gt")
assert len(gt_indices) == len(gt_scores)
gt_len = len(gt_indices)
top_len = len(gt_indices[0])
assert top_len == len(gt_scores[0])

print("Ground truth contains %d queries" % gt_len)

print("gt:", gt_indices[0])

# https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_small.csr.gz
data = read_sparse_matrix("base_small.csr")
vec_count = data.shape[0]

for i in range(0, vec_count):
    point = data[i]
    indices, values = csr_to_sparse_vector(point)
    break

# https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/queries.dev.csr.gzip
query_data = read_sparse_matrix("queries.dev.csr")
query_count = query_data.shape[0]

for i in range(0, query_count):
    point = query_data[i]
    query_vector = csr_to_sparse_vector(point)
    print(query_vector)

    # TODO: Search for nearest neighbor
    results = []

    # Checking the groundtruth
    expected_scores = gt_scores[i]
    expected_ids = gt_indices[i]
    for j in range(0, len(results)):
        result = results[j]
        result_score = result.score
        expected_score = float(expected_scores[j])

        # if not compare_floats_percentage(result_score, expected_score, 1):
            # print("Score mismatch: %f != %f" % (result_score, expected_score))

        result_id = result.id
        expected_id = expected_ids[j]
        if result_id != expected_id:
            print("ID mismatch: %d != %d" % (result_id, expected_id))

    break
