import torch


def switch_rows(matrix, row1, row2):
    matrix[[row1, row2], :] = matrix[[row2, row1], :]
    return matrix


def switch_cols(matrix, col1, col2):
    matrix[:, [col1, col2]] = matrix[:, [col2, col1]]
    return matrix


def calc_pivot_norm(Q, M, R, pivot_row, col):
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


def qr2ref_with_M(features: torch.Tensor, theta: float):
    device = features.device
    dtype = features.dtype
    Q, Rref = torch.linalg.qr(features, mode='reduced')
    num_rows = Rref.shape[0]
    # Q = np.ascontiguousarray(Q)
    M = torch.eye(Rref.shape[0], dtype=Rref.dtype, device=device)
    pivot_row = 0
    for col in range(Rref.shape[1]):
        pivot_norm = calc_pivot_norm(Q, M, Rref, pivot_row, col)
        pivot_candidates_ = Rref[pivot_row:col + 1, col]
        if is_independent(pivot_norm, torch.linalg.norm(features[:, col]), theta, len(pivot_candidates_), dtype):
            pivot_index = torch.argmax(torch.abs(pivot_candidates_))
            new_pivot_row = pivot_index + pivot_row
            # Switch place to that largest element is at the pivot row
            Rref = switch_rows(Rref, pivot_row, new_pivot_row)
            M = switch_cols(M, pivot_row, new_pivot_row)

            reduction_ratios = (Rref[pivot_row + 1:min(col + 1, num_rows), col] / Rref[pivot_row, col])
            Rref[pivot_row + 1:min(col + 1, num_rows), :] -= torch.outer(reduction_ratios, Rref[pivot_row, :])
            M[:, pivot_row] += torch.einsum('ij,j->i', M[:, pivot_row + 1:min(col + 1, num_rows)], reduction_ratios)

            pivot_row += 1
        # Set small elements to zeros
        Rref[pivot_row:, col] = torch.zeros(Rref.shape[0] - pivot_row, dtype=dtype, device=device)
    return Rref, M


def get_pivot_columns(matrix: torch.Tensor):
    is_pivot_column = torch.zeros(matrix.shape[1], dtype=torch.bool, device=matrix.device)
    eps = 1e-15 if matrix.dtype == torch.float64 else 1e-6
    for i in range(matrix.shape[0]):
        pivot_candidates = (torch.abs(matrix[i]) > eps).to(torch.int8)
        if torch.any(pivot_candidates):
            pivot_column = torch.argmax(pivot_candidates)
            is_pivot_column[pivot_column] = 1
    return is_pivot_column


def get_keep_columns_qr(features: torch.Tensor, tolerance: float):
    if tolerance < 0:
        columns_to_keep = torch.ones(features.shape[1], dtype=torch.bool, device=features.device)
    else:
        rref, M = qr2ref_with_M(features, theta=tolerance)
        columns_to_keep = get_pivot_columns(rref)

    return columns_to_keep
