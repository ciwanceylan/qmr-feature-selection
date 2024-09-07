from typing import Literal
import numpy as np
import scipy

import qmrfs.core as core

SortingStrategy = Literal['entropy_high2low', 'entropy_low2high', 'default', 'random']


class QRPFeatureSelector:

    def __init__(self, num_features: int):
        self.num_features = num_features
        self.recon_errors_ = None

    def fit_transform(self, features: np.ndarray):
        ones = np.ones((features.shape[0], 1), dtype=features.dtype)
        features = np.concatenate((ones, features), axis=1)
        features_normed = features / np.linalg.norm(features, axis=0, keepdims=True)

        q, r, p = scipy.linalg.qr(features_normed, mode='economic', pivoting=True)
        p_without_constant = p[p > 0]
        selected_features = p_without_constant[:self.num_features]
        pruned_features = features[:, selected_features]

        r_block_removed = r[-self.num_features:, :]
        recon_errors = np.linalg.norm(r_block_removed, axis=0)
        recon_errors = recon_errors[-self.num_features:]
        self.recon_errors_ = recon_errors

        return pruned_features


def create_initial_ordering(features: np.ndarray, sorting_strategy: SortingStrategy,
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

    return features[:, feature_order]


def qrp_fs(features: np.ndarray, num_features: int, sorting_strategy: SortingStrategy, seed: int = None):
    features = features[:, features.std(axis=0) > 0]
    features = create_initial_ordering(features, sorting_strategy, seed=seed)

    selector = QRPFeatureSelector(num_features=num_features)
    selected_features = selector.fit_transform(features)
    return selected_features, selector.recon_errors_
