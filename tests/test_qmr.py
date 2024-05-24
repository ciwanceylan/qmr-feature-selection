import pytest
import torch

import qmrfs.core as qmrfscore


def evaluate_qmr_decomp(features, theta):
    column_mask, Q, M, Rref, recon_errors, feature_norms = qmrfscore.qmr_fs_core(features, theta=theta)

    recon_features = Q @ M @ Rref
    torch.testing.assert_close(features[:, column_mask], recon_features[:, column_mask])
    lost_features = features[:, ~column_mask]

    norms = torch.linalg.vector_norm(features, dim=0)
    torch.testing.assert_close(norms[~column_mask], torch.tensor(feature_norms))

    delta = torch.linalg.vector_norm(lost_features - recon_features[:, ~column_mask], dim=0)
    torch.testing.assert_close(delta, torch.tensor(recon_errors))
    assert torch.all(delta <= theta * torch.linalg.vector_norm(lost_features)).item()
    # torch.testing.assert_close(features[:, ~column_mask], recon_features[:, ~column_mask])


@pytest.mark.parametrize('seed', [234213423, 73422321, 67293234, 2342353423, 1234156, 172336])
@pytest.mark.parametrize('theta', [0., 0.05, 0.1, 0.3, 0.6, 0.8])
def test_qmr_decomp(seed: int, theta: float):
    torch.manual_seed(seed)
    features = torch.randn(15, 10, dtype=torch.float32)
    evaluate_qmr_decomp(features, theta=theta)

    features = torch.rand(15, 10, dtype=torch.float32)
    evaluate_qmr_decomp(features, theta=theta)

    features = torch.randint(low=0, high=3, size=(15, 10), dtype=torch.float32)
    evaluate_qmr_decomp(features, theta=theta)
