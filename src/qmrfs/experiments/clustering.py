import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from qmrfs import qmr_feature_selection as qmrfs
from qmrfs.experiments.utils import DATASET_INFO, load_dataset


def evaluate_clustering(features, y, seed: int, num_reps: int = 20):
    num_class = len(np.unique(y))
    seeds = np.random.SeedSequence(seed).generate_state(num_reps)

    f_mean = features.mean(axis=0, keepdims=True)
    f_std = features.std(axis=0, keepdims=True)
    f_std[f_std == 0] = 1.
    features_std = (features - f_mean) / f_std

    scores = []

    for i, seed_ in enumerate(seeds):
        clf = KMeans(n_clusters=num_class, random_state=seed_)
        clf = clf.fit(features_std)
        pred_y = clf.labels_
        score = normalized_mutual_info_score(y, pred_y)
        scores.append(score)
    return np.mean(scores), np.std(scores)


def run_clustering_experiment(tolerance: float, seed: int, verbose: bool = False):
    if verbose:
        print("Clustering evaluation")
    rel_scores = dict()
    abs_scores = []

    for dataset, info in DATASET_INFO.items():
        if verbose:
            print(f"Running clustering for dataset {dataset}")
        X_data, X_orig, y = load_dataset(info.uci_id)
        start = time.perf_counter()
        pruned_x = qmrfs.qmr_fs(X_data, tolerance=tolerance)
        duration = time.perf_counter() - start

        full_score, full_score_std = evaluate_clustering(X_data, y, seed=seed)
        red_score, red_score_std = evaluate_clustering(pruned_x, y, seed=seed)

        abs_scores.append({
                "ref_val": DATASET_INFO[dataset].acc_ref,
                "full_mean": full_score,
                "full_std": full_score_std,
                "red_mean": red_score,
                "red_std": red_score_std,
                "rel_score": red_score / full_score,
                "full_dim": X_data.shape[1],
                "red_dim": pruned_x.shape[1],
                "dim_ratio": pruned_x.shape[1] / X_data.shape[1],
                "duration": duration,
                "tolerance": tolerance,
                "dataset": dataset
            }
        )

        rel_scores[dataset] = red_score / full_score
    return pd.DataFrame(abs_scores), pd.Series(rel_scores)
