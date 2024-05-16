from typing import Optional
import json
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

import qmrfs.experiments.utils as utils
import qmrfs.qmr_feature_selection as qmrfs
import torch


def load_us_census():
    return utils.load_dataset(dataset_id=116)


def load_musae_git(as_dense: bool):
    targets = pd.read_csv("datasets/musae_git/musae_git_target.csv", index_col=0)
    y = targets['ml_target'].to_numpy()
    with open("datasets/musae_git/musae_git_features.json", 'r') as fp:
        data = json.load(fp)
    X = sp.dok_array((len(data), 4006), dtype=np.float32)
    for idx, features in data.items():
        for f in features:
            X[int(idx), f] = 1.0

    if as_dense:
        X = X.toarray()
    else:
        X = X.tocsr()
    return X, y


def time_fs(X_data, device: Optional[torch.device]):
    start = time.perf_counter()
    pruned_x, recon_errors, feature_norms = qmrfs.qmr_fs(
        X_data,
        tolerance=1e-1,
        sorting_strategy='entropy_high2low',
        seed=None,
        device=device
    )
    duration = time.perf_counter() - start
    num_orig_features = X_data.shape[1]
    num_kept_features = pruned_x.shape[1]
    return duration, num_orig_features, num_kept_features


def main():
    X_data, y = load_musae_git(as_dense=True)

    # X_data, X_orig, y = load_us_census()

    duration, num_orig_features, num_kept_features = time_fs(X_data, device=torch.device('cpu'))
    print("Duration: ", duration)
    print("Num start dim.: ", num_orig_features)
    print("Num end dim.: ", num_kept_features)


if __name__ == "__main__":
    main()
