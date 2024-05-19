import numpy as np
import numba as nb


@nb.jit(nb.float32[:](nb.types.Array(nb.types.float32, 2, 'A', readonly=True)),
        nopython=True, nogil=True, parallel=True)
def nb_max(array):
    out = np.empty(array.shape[0], dtype=np.float32)
    for i in nb.prange(array.shape[0]):
        out[i] = np.max(array[i])

    return out


@nb.jit(nb.float32[:](nb.types.Array(nb.types.float32, 2, 'A', readonly=True)),
        nopython=True, nogil=True, parallel=True)
def nb_min(array):
    out = np.empty(array.shape[0], dtype=np.float32)
    for i in nb.prange(array.shape[0]):
        out[i] = np.min(array[i])

    return out


@nb.jit(nb.float32[:, :](nb.types.Array(nb.types.float32, 2, 'A', readonly=True)), nopython=True, nogil=True)
def compute_W(features: np.ndarray):
    num_instances, num_feat = features.shape
    max_vals = nb_max(features.T)
    min_vals = nb_min(features.T)
    # max_vals = np.max(features, axis=0)
    # min_vals = np.min(features, axis=0)
    norm_val = max_vals - min_vals
    norm_val[(norm_val == 0) & (max_vals == 0) & (min_vals == 0)] = 1
    norm_val[(norm_val == 0) & ~(max_vals == 0)] = max_vals[(norm_val == 0) & ~(max_vals == 0)]

    norm_val = norm_val[np.newaxis, ...]
    assert norm_val.shape[1] == num_feat and norm_val.shape[0] == 1
    features_norm = features / norm_val
    W = np.zeros((num_instances, num_instances), dtype=features.dtype)
    for p in range(num_feat):
        feature_col = features_norm[:, p]
        W += 1. - np.abs(feature_col[..., np.newaxis] - feature_col[np.newaxis, ...])
    return W / num_feat


@nb.jit(nb.float32[:, :](nb.types.Array(nb.types.float32, 2, 'A', readonly=True)), nopython=True, nogil=True)
def compute_L(W: np.ndarray):
    num_instances = W.shape[0]
    deg = np.sum(W, axis=1)
    deg[deg == 0] = 1
    deg_inv_sqrt = np.sqrt(1. / deg)

    L = np.eye(num_instances, dtype=np.float32) - (deg_inv_sqrt[..., np.newaxis] * W * deg_inv_sqrt[np.newaxis, ...])
    assert len(L.shape) == 2
    return L.astype(np.float32)


@nb.jit(nb.float32[:, :](nb.types.Array(nb.types.float32, 2, 'A', readonly=True)), nopython=True, nogil=True)
def compute_L_from_features(features: np.ndarray):
    W = compute_W(features)
    L = compute_L(W)
    return L


@nb.jit(nb.float32[::1](nb.float32[:, :]), nopython=True, nogil=True)
def get_spectrum(L: np.ndarray):
    spectrum = np.linalg.eigvalsh(L)
    return spectrum


@nb.jit(nb.float32(nb.float32[:], nb.int64), nopython=True, nogil=True)
def gamma_function(spectrum: np.ndarray, k: int):
    tau = np.sum(spectrum[2:k + 1])
    gamma = 0.0
    for i in range(2, k + 1):
        for j in range(i + 1, k + 2):
            gamma += np.abs((spectrum[i] - spectrum[j]) / tau).item()
    return gamma


@nb.jit(nb.types.Tuple((nb.float32[:, :], nb.bool_[::1]))(nb.types.Array(nb.types.float32, 2, 'A', readonly=True), nb.int64), nopython=True, nogil=True)
def usfsm_(features: np.ndarray, k: int):
    num_feat = features.shape[1]
    keep_mask = np.ones((num_feat,), dtype=np.bool_)
    L = compute_L_from_features(features)
    spectrum = get_spectrum(L)
    gamma = gamma_function(spectrum, k=k)
    feat_ranking_value = np.zeros((num_feat,), dtype=np.float32)

    for i in range(num_feat):
        print(f"{i} / {num_feat}")
        sub_feat_mask = keep_mask.copy()
        sub_feat_mask[i] = 0
        sub_L = compute_L_from_features(features[:, sub_feat_mask])
        sub_spectrum = get_spectrum(sub_L)
        sub_gamma = gamma_function(sub_spectrum, k=k)
        feat_ranking_value[i] = gamma - sub_gamma

    feat_ranking_value = np.asarray(feat_ranking_value)
    feat_ranking = np.argsort(-feat_ranking_value)
    keep_mask = np.zeros((num_feat,), dtype=np.bool_)
    keep_mask[feat_ranking[:k]] = 1
    features = features[:, keep_mask]

    return features, keep_mask


def usfsm(features: np.ndarray, k: int):
    return usfsm_(features.astype(np.float32), k=k)
