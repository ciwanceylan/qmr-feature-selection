import numpy as np


def compute_svd_entropy(features: np.ndarray):
    num_features = features.shape[1]
    s = np.linalg.svd(features, compute_uv=False, hermitian=False, full_matrices=False)

    square_s = np.power(s, 2)
    v = square_s / np.sum(square_s)
    entropy = - (1. / np.log(num_features)) * np.dot(v[v > 0], np.log(v[v > 0]))
    return entropy


def svd_entropy_sr(features: np.ndarray, k: int):
    num_features = features.shape[1]
    entropy_full = compute_svd_entropy(features)

    rank_values = []
    mask = np.ones((num_features,), dtype=bool)
    for i in range(num_features):
        subset_mask = mask.copy()
        subset_mask[i] = 0
        sub_entropy = compute_svd_entropy(features[:, subset_mask])
        rank_values.append(entropy_full - sub_entropy)

    rank_values = np.asarray(rank_values)
    feat_ranking = np.argsort(-rank_values)
    keep_mask = np.zeros((num_features,), dtype=bool)
    keep_mask[feat_ranking[:k]] = 1
    features = features[:, keep_mask]

    return features, keep_mask
