import numpy as np


def compute_similarity(features: np.ndarray):
    variances = np.var(features, axis=0, keepdims=True)
    var_x_p_var_y = variances + variances.T
    var_x_t_var_y = variances * variances.T
    # corr = np.cov(features.T) / np.sqrt(var_x_t_var_y)

    # sim = 0.5 * (var_x_p_var_y - np.sqrt(np.power(var_x_p_var_y, 2) - 4 * var_x_t_var_y * (1 - np.power(corr, 2))))
    sim = 0.5 * (var_x_p_var_y - np.sqrt(np.power(var_x_p_var_y, 2) - 4 * var_x_t_var_y + 4 * np.power(np.cov(features.T), 2)))
    return sim


def fsfs(features: np.ndarray, k=None):
    num_feat = features.shape[1]
    k = num_feat - 1
    keep_mask = np.ones((num_feat,), dtype=bool)

    while k > 1:
        similarities = compute_similarity(features[:, keep_mask])
        sim_idx_sorted = np.argsort(similarities, axis=1)
        sorted_similarities = np.take_along_axis(similarities, sim_idx_sorted, axis=1)
        rk = sorted_similarities[:, k]

        idx_min_r = np.argmin(rk)
        to_remove = sim_idx_sorted[idx_min_r, :k + 1]
        keep_mask[to_remove] = 0
        if idx_min_r in to_remove:
            keep_mask[idx_min_r] = 1

        rk_iprime = rk[idx_min_r]
        eps = rk_iprime
        num_kept = np.sum(keep_mask)
        if k > num_kept - 1:
            k = num_kept - 1
        if k > 1:
            while rk_iprime > eps:
                k -= 1
                rk = sorted_similarities[:, k]
                rk_iprime = np.min(rk[keep_mask])
                if k == 1:
                    break
    return features[:, keep_mask], keep_mask
