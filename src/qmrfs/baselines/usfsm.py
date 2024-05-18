import numpy as np
from scipy.spatial.distance import pdist, squareform


def compute_W(features: np.ndarray):
    num_instances, num_feat = features.shape
    max_vals = np.max(features, axis=0)
    min_vals = np.min(features, axis=0)
    norm_val = max_vals - min_vals
    norm_val[(norm_val == 0) & (max_vals == 0) & (min_vals == 0)] = 1
    norm_val[(norm_val == 0) & ~(max_vals == 0)] = max_vals[(norm_val == 0) & ~(max_vals == 0)]

    features_norm = features / norm_val
    W = np.zeros((num_instances, num_instances), dtype=features.dtype)
    for p in range(num_feat):
        W += 1. - np.abs(np.subtract.outer(features_norm[:, p], features_norm[:, p]))
    return W / num_feat


def compute_L(W: np.ndarray):
    num_instances = W.shape[0]
    deg = np.sum(W, axis=1)
    deg[deg == 0] = 1
    deg_inv_sqrt = np.sqrt(1. / deg)

    L = np.eye(num_instances) - (deg_inv_sqrt.reshape(-1, 1) * W * deg_inv_sqrt.reshape(1, -1))
    return L


def get_spectrum(features: np.ndarray):
    L = compute_L(compute_W(features))
    spectrum = np.sort(np.linalg.eigvals(L))
    return spectrum


def gamma_function(spectrum: np.ndarray, k: int):
    tau = np.sum(spectrum[2:k + 1])
    gamma = 0.0
    for i in range(2, k + 1):
        for j in range(i + 1, k + 2):
            gamma += np.abs((spectrum[i] - spectrum[j]) / tau)
    return gamma


def usfsm(features: np.ndarray, k: int):
    num_feat = features.shape[1]
    keep_mask = np.ones((num_feat,), dtype=bool)
    spectrum = get_spectrum(features)
    gamma = gamma_function(spectrum, k=k)
    feat_ranking_value = []

    for i in range(num_feat):
        sub_feat_mask = keep_mask.copy()
        sub_feat_mask[i] = 0
        sub_spectrum = get_spectrum(features[:, sub_feat_mask])
        sub_gamma = gamma_function(sub_spectrum, k=k)
        feat_ranking_value.append(gamma - sub_gamma)

    feat_ranking_value = np.asarray(feat_ranking_value)
    feat_ranking = np.argsort(-feat_ranking_value)
    keep_mask = np.zeros((num_feat,), dtype=bool)
    keep_mask[feat_ranking[:k]] = 1
    features = features[:, keep_mask]

    return features, keep_mask
