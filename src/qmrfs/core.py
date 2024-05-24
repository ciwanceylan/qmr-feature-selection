import numpy as np
import torch
import numba as nb


def switch_rows(matrix, row1, row2):
    matrix[[row1, row2], :] = matrix[[row2, row1], :]
    return matrix


def switch_cols(matrix, col1, col2):
    matrix[:, [col1, col2]] = matrix[:, [col2, col1]]
    return matrix


def calc_pivot_norm(Q, M, R, pivot_row, col):
    # Columns beyond `col` in Q and rows below `col` in M do not need to be used since those rows in M will always be
    # zero. This small optimization is excluded from the paper for clarity of presentation.
    vector_from_pivots = Q[:, :col + 1] @ (M[:col + 1, pivot_row:col + 1] @ R[pivot_row:col + 1, col])
    pivot_norm = torch.linalg.norm(vector_from_pivots)
    return pivot_norm


def is_independent(pivot_norm: float, feature_norm: float, theta: float, num_candidates: int, dtype: torch.dtype):
    threshold = theta * feature_norm
    a = pivot_norm > threshold
    rtol = 1e-3
    atol = num_candidates * 1e-15 if dtype == torch.float64 else num_candidates * 1e-6
    isclose = torch.abs(pivot_norm - threshold) <= (atol + rtol * torch.abs(threshold))
    b = not isclose
    return a and b


def qmr_fs_core(features: torch.Tensor, theta: float):
    device = features.device
    dtype = features.dtype
    recon_errors = []
    feature_norms = []
    column_mask = torch.zeros(features.shape[1], dtype=torch.bool, device=features.device)

    Q, Rref = torch.linalg.qr(features, mode='reduced')
    num_rows = Rref.shape[0]
    M = torch.eye(Rref.shape[0], dtype=Rref.dtype, device=device)
    pivot_row = 0
    for col in range(Rref.shape[1]):
        pivot_norm = calc_pivot_norm(Q, M, Rref, pivot_row, col)
        pivot_candidates_ = Rref[pivot_row:col + 1, col]
        feature_norm = torch.linalg.norm(features[:, col])
        if is_independent(pivot_norm, feature_norm, theta, len(pivot_candidates_), dtype):
            pivot_index = torch.argmax(torch.abs(pivot_candidates_))
            new_pivot_row = pivot_index + pivot_row
            # Switch place to that largest element is at the pivot row
            Rref = switch_rows(Rref, pivot_row, new_pivot_row)
            M = switch_cols(M, pivot_row, new_pivot_row)

            reduction_ratios = (Rref[pivot_row + 1:min(col + 1, num_rows), col] / Rref[pivot_row, col])
            Rref[pivot_row + 1:min(col + 1, num_rows), :] -= torch.outer(reduction_ratios, Rref[pivot_row, :])
            M[:, pivot_row] += torch.einsum('ij,j->i', M[:, pivot_row + 1:min(col + 1, num_rows)], reduction_ratios)

            pivot_row += 1
            column_mask[col] = 1
        else:
            recon_errors.append(pivot_norm.item())
            feature_norms.append(feature_norm.item())
        # Set small elements to zeros. Done also if column is kept to remove rounding errors.
        Rref[pivot_row:, col] = torch.zeros(Rref.shape[0] - pivot_row, dtype=dtype, device=device)

    return column_mask, Q, M, Rref, recon_errors, feature_norms


def get_keep_columns_qr(features: torch.Tensor, tolerance: float):
    if tolerance < 0:
        column_mask = torch.ones(features.shape[1], dtype=torch.bool, device=features.device)
        recon_errors = []
        feature_norms = []
    else:
        column_mask, Q, M, Rref, recon_errors, feature_norms = qmr_fs_core(features, theta=tolerance)

    return column_mask, recon_errors, feature_norms


@nb.jit(nb.types.Tuple((nb.int64[::1], nb.float32[::1]))(nb.float32[:, :]), nopython=True, nogil=True, parallel=True)
def count_unique_f32(features: np.ndarray):
    num_instances, num_dims = features.shape
    num_unique = np.empty((num_dims,), dtype=np.int64)
    entropy = np.empty((num_dims,), dtype=np.float32)
    for i in nb.prange(num_dims):
        unique_vals = np.unique(features[:, i])
        num_unique_ = len(unique_vals)
        if num_unique_ < 10:
            entropy_ = 0.
            for j in range(num_unique_):
                p = np.sum(features[:, i] == unique_vals[j]) / num_instances
                entropy_ -= p * np.log(p)
            entropy[i] = entropy_
        else:
            # Approximation of continuous values
            entropy[i] = np.log(num_unique_)
        num_unique[i] = num_unique_
    return num_unique, entropy


@nb.jit(nb.types.Tuple((nb.int64[::1], nb.float64[::1]))(nb.float32[:, :]), nopython=True, nogil=True, parallel=True)
def count_unique_f64(features: np.ndarray):
    num_instances, num_dims = features.shape
    num_unique = np.empty((num_dims,), dtype=np.int64)
    entropy = np.empty((num_dims,), dtype=np.float64)
    for i in nb.prange(num_dims):
        unique_vals = np.unique(features[:, i])
        num_unique_ = len(unique_vals)
        if num_unique_ < 10:
            entropy_ = 0.
            for j in range(num_unique_):
                p = np.sum(features[:, i] == unique_vals[j]) / num_instances
                entropy_ -= p * np.log(p)
            entropy[i] = entropy_
        else:
            # Approximation of continuous values
            entropy[i] = np.log(num_unique_)
        num_unique[i] = num_unique_
    return num_unique, entropy
