import torch
import numpy as np
from qmrfs.core import calc_pivot_norm, is_independent, switch_rows, switch_cols


def qmr_fs_core(features: torch.Tensor, theta: float):
    device = features.device
    dtype = features.dtype
    recon_errors = []
    feature_norms = []
    column_mask = torch.zeros(features.shape[1], dtype=torch.bool, device=features.device)

    Q, Rref = torch.linalg.qr(features, mode='reduced')

    num_rows = Rref.shape[0]
    M = torch.eye(Rref.shape[0], dtype=Rref.dtype, device=device)
    print("Q: \n", np.around(Q.numpy(), 3))
    print("M: \n", np.around(M.numpy(), 3))
    print("R: \n", np.around(Rref.numpy(), 3))
    print("---------------------------------------------------")
    print()
    pivot_row = 0
    for col in range(Rref.shape[1]):
        pivot_norm = calc_pivot_norm(Q, M, Rref, pivot_row, col)
        pivot_candidates_ = Rref[pivot_row:col + 1, col]
        feature_norm = torch.linalg.norm(features[:, col])
        if is_independent(pivot_norm, feature_norm, theta, len(pivot_candidates_), dtype):
            pivot_index = torch.argmax(torch.abs(pivot_candidates_))
            new_pivot_row = pivot_index + pivot_row
            # Switch place to that largest element is at the pivot row
            # Rref = switch_rows(Rref, pivot_row, new_pivot_row)
            # M = switch_cols(M, pivot_row, new_pivot_row)

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
        print("M: \n", np.around(M.numpy(), 3))
        print("R: \n", np.around(Rref.numpy(), 3))
        print()
        print("Recon: \n", np.around(M @ Rref, 3))
        print("---------------------------------------------------")
        print()

    return column_mask, Q, M, Rref, recon_errors, feature_norms


def main():
    b = torch.tensor([
        [1, 1, 2, 1, 1],
        [1, 0, 0, 1, 2],
        [1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0]
    ], dtype=torch.float32)

    column_mask, Q, M, Rref, recon_errors, feature_norms = qmr_fs_core(b, theta=1e-5)


if __name__ == "__main__":
    main()
