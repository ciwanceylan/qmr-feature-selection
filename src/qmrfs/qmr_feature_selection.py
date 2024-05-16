from typing import Literal
import numpy as np
import scipy.stats as ss
import torch
import qmrfs.core as core

SortingStrategy = Literal[
    'entropy_low2high', 'entropy_high2low', 'default', 'random', 'mm_score_low2high', 'mm_score_high2low']


class QMRFeatureSelector:

    def __init__(self, tolerance: float):
        self.tolerance = tolerance

    def fit_transform(self, features: torch.Tensor):
        columns_to_keep_mask = core.get_keep_columns_qr(features, tolerance=self.tolerance)
        pruned_features = features[:, columns_to_keep_mask]
        return pruned_features, columns_to_keep_mask


def multimodality_score(features: np.ndarray):
    n = features.shape[0]
    assert n > 3
    g = ss.skew(features, axis=0)
    k = ss.kurtosis(features, axis=0, fisher=True)

    c = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    b = (g ** 2 + 1) / (k + c)
    return b


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
    elif sorting_strategy in {'mm_score_low2high', 'mm_score_high2low'}:
        mm_score = multimodality_score(features)
        if sorting_strategy == 'mm_score_low2high':
            feature_order = np.argsort(mm_score)
        else:
            feature_order = np.argsort(-mm_score)
    elif sorting_strategy == 'default':
        feature_order = np.arange(num_features, dtype=np.int64)
    elif sorting_strategy == 'random':
        rng = np.random.default_rng(seed)
        feature_order = rng.permutation(num_features)
    else:
        raise NotImplementedError

    # tie_breaker_score2 = num_unique_values + multimodality_score(features)
    # tie_breaker_score = entropy
    #
    # tie_breaker_score = torch.from_numpy(tie_breaker_score).to(device)

    features = torch.from_numpy(features).to(device)

    # orthogonality_scores = compute_orthogonality_scores(features)
    # first_feature = torch.argmax(tie_breaker_score).item()
    # feature_order = [first_feature]
    # tie_breaker_score[first_feature] = -torch.inf
    # for i in range(1, num_features):
    #     # Line below is a heuristic to primarily sort using the orthogonality score and use mm_score as a tie-breaker
    #     feature_select_score = (num_unique_values.max() + 1) * (
    #                 1 - torch.mean(orthogonality_scores[feature_order], dim=0)) + tie_breaker_score
    #     next_feature = torch.argmax(feature_select_score).item()
    #     feature_order.append(next_feature)
    #     tie_breaker_score[next_feature] = -torch.inf

    return features[:, feature_order]


def qmr_fs(features: np.ndarray, tolerance: float, sorting_strategy: SortingStrategy, device: torch.device = None,
           seed: int = None):
    if device is None:
        device = torch.device('cpu')
    features = features[:, features.std(axis=0) > 0]
    features = create_initial_ordering(features, sorting_strategy, device=device, seed=seed)

    selector = QMRFeatureSelector(tolerance=tolerance)
    selected_features, columns_to_keep_mask = selector.fit_transform(features)
    return selected_features.numpy()
