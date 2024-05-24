from typing import Literal
import numpy as np
import torch
import qmrfs.core as core

SortingStrategy = Literal['entropy_high2low', 'entropy_low2high', 'default', 'random']


class QMRFeatureSelector:

    def __init__(self, tolerance: float):
        self.tolerance = tolerance
        self.recon_errors_ = None
        self.feature_norms_ = None

    def fit_transform(self, features: torch.Tensor):
        ones = torch.ones((features.shape[0], 1), dtype=features.dtype, device=features.device)
        features = torch.cat((ones, features), dim=1)

        columns_to_keep_mask, recon_errors, feature_norms = core.get_keep_columns_qr(
            features, tolerance=self.tolerance
        )
        if columns_to_keep_mask.sum() > 1:
            columns_to_keep_mask[0] = 0
        self.recon_errors_ = np.asarray(recon_errors)
        self.feature_norms_ = np.asarray(feature_norms)

        pruned_features = features[:, columns_to_keep_mask]
        return pruned_features, columns_to_keep_mask


def compute_orthogonality_scores(features: torch.Tensor):
    features = features / torch.linalg.vector_norm(features, ord=2, dim=0, keepdims=True)
    orthogonality_scores = torch.abs(features.T @ features)
    return orthogonality_scores


def create_initial_ordering(features: np.ndarray, sorting_strategy: SortingStrategy, device: torch.device,
                            as_f64: bool = False, seed: int = None):
    num_instances, num_features = features.shape
    features = features.astype(np.float64) if as_f64 else features.astype(np.float32)

    if sorting_strategy in {'entropy_low2high', 'entropy_high2low'}:
        num_unique_values, entropy = core.count_unique_f64(features) if as_f64 else core.count_unique_f32(features)
        if sorting_strategy == 'entropy_low2high':
            feature_order = np.argsort(entropy)
        else:
            feature_order = np.argsort(-entropy)
    elif sorting_strategy == 'default':
        feature_order = np.arange(num_features, dtype=np.int64)
    elif sorting_strategy == 'random':
        rng = np.random.default_rng(seed)
        feature_order = rng.permutation(num_features)
    else:
        raise NotImplementedError

    features = torch.from_numpy(features).to(device)

    return features[:, feature_order]


def qmr_fs(features: np.ndarray, tolerance: float, sorting_strategy: SortingStrategy,
           device: torch.device = None, seed: int = None):
    if device is None:
        device = torch.device('cpu')
    features = features[:, features.std(axis=0) > 0]
    features = create_initial_ordering(features, sorting_strategy, device=device, seed=seed)

    selector = QMRFeatureSelector(tolerance=tolerance)
    selected_features, columns_to_keep_mask = selector.fit_transform(features)
    return selected_features.cpu().numpy(), selector.recon_errors_, selector.feature_norms_
