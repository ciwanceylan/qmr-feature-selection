import json
import numpy as np
import scipy.sparse as sp


def load_musae_git(as_dense: bool):
    with open("dataset/musae_git_features.json", 'r') as fp:
        data = json.load(fp)
    X = sp.dok_array((len(data), 4006), dtype=np.float32)
    for node, features in data.items():
        for f in features:
            X[node, f] = 1.0

    if as_dense:
        X = X.toarray()
    else:
        X = X.tocsr()
    return X
