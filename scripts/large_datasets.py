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


def load_snap_patents():
    X_data = np.load("datasets/snap_patents/snap_patents.npy")
    y = np.array([])
    return X_data, y


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


def load_kddcup1999():
    feature_dtypes = {
        "duration": float,
        "protocol_type": 'category',
        "service": 'category',
        "flag": 'category',
        "src_bytes": float,
        "dst_bytes": float,
        "land": 'category',
        "wrong_fragment": float,
        "urgent": float,
        "hot": float,
        "num_failed_logins": float,
        "logged_in": 'category',
        "num_compromised": float,
        "root_shell": float,
        "su_attempted": float,
        "num_root": float,
        "num_file_creations": float,
        "num_shells": float,
        "num_access_files": float,
        "num_outbound_cmds": float,
        "is_host_login": 'category',
        "is_guest_login": 'category',
        "count": float,
        "srv_count": float,
        "serror_rate": float,
        "srv_serror_rate": float,
        "rerror_rate": float,
        "srv_rerror_rate": float,
        "same_srv_rate": float,
        "diff_srv_rate": float,
        "srv_diff_host_rate": float,
        "dst_host_count": float,
        "dst_host_srv_count": float,
        "dst_host_same_srv_rate": float,
        "dst_host_diff_srv_rate": float,
        "dst_host_same_src_port_rate": float,
        "dst_host_srv_diff_host_rate": float,
        "dst_host_serror_rate": float,
        "dst_host_srv_serror_rate": float,
        "dst_host_rerror_rate": float,
        "dst_host_srv_rerror_rate": float,
        "label": 'category'
    }

    train_data = pd.read_csv(
        "datasets/kddcup99/kddcup.data",
        names=list(feature_dtypes.keys()),
        dtype=feature_dtypes
    )
    y = pd.factorize(train_data.pop('label'))
    del feature_dtypes['label']
    test_data = pd.read_csv(
        "datasets/kddcup99/kddcup.testdata.unlabeled",
        names=list(feature_dtypes.keys()),
        dtype=feature_dtypes
    )
    data = pd.concat((train_data, test_data), axis=0, ignore_index=True)
    X_data = pd.get_dummies(data).astype(float).to_numpy()
    return X_data, y


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
    num_kept_features = pruned_x.shape[1]
    return duration, num_kept_features


def main():
    # X_data, y = load_musae_git(as_dense=True)
    X_data, y = load_kddcup1999()
    # X_data, X_orig, y = load_us_census()

    n, m = X_data.shape
    print("Num. instances: ", n)
    print("Num. features: ", m)
    duration, num_kept_features = time_fs(X_data, device=torch.device('cuda'))
    print("Duration: ", duration)
    print("Num end dim.: ", num_kept_features)


if __name__ == "__main__":
    main()
