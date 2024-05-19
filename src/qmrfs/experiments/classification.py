from typing import Optional, Union, Literal, List, Dict
import time
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import LinearSVC
from scipy.io import loadmat
import tqdm
import glob
import copy
from joblib import Parallel, delayed

from qmrfs import qmr_feature_selection as qmrfs
import qmrfs.experiments.utils as utils


def evaluate_clf(features, y, num_reps: int, seed: int):
    f_mean = features.mean(axis=0, keepdims=True)
    f_std = features.std(axis=0, keepdims=True)
    f_std[f_std == 0] = 1.
    features_std = (features - f_mean) / f_std

    seeds = np.random.SeedSequence(seed).generate_state(num_reps)

    # scores = []

    def train(X, y, seed_):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_)
            clf = LinearSVC(dual='auto')
            scores = []
            for j, (train_index, test_index) in enumerate(kf.split(X, y=y)):
                train_x, train_y = X[train_index], y[train_index]
                test_x, test_y = X[test_index], y[test_index]

                clf = clf.fit(train_x, train_y)
                pred_y = clf.predict(test_x)
                score = accuracy_score(test_y, pred_y)
                scores.append({"accuracy": score, "seed": seed_, "split": j})
        return scores

    # all_scores = []
    # for i, seed_ in enumerate(seeds):
    #     all_scores += train(seed_)

    all_scores = Parallel(n_jobs=num_reps)(delayed(train)(features_std, y, seed_) for seed_ in seeds)
    all_scores = sum(all_scores, [])
    return all_scores


def run_classification_experiment(tolerance: Union[float, Literal['auto']], sorting_strategy: qmrfs.SortingStrategy,
                                  feature_translation: qmrfs.TranslationMode, seed: int,
                                  use_factorize_categorical: bool, feature_order_seed: Optional[int] = None,
                                  verbose: bool = False):
    if verbose:
        print("Classification evaluation")
    rel_scores = dict()
    abs_scores = []

    for dataset, info in tqdm.tqdm(utils.DATASET_INFO.items(), total=len(utils.DATASET_INFO)):
        tol = utils.DATASET2THETA[dataset] if tolerance == 'auto' else tolerance
        if verbose:
            print(f"Running classification for dataset {dataset}")
        X_data, X_orig, y = utils.load_dataset(info.uci_id, use_factorize_categorical=use_factorize_categorical)
        start = time.perf_counter()
        pruned_x, recon_errors, feature_norms = qmrfs.qmr_fs(
            X_data,
            tolerance=tol,
            sorting_strategy=sorting_strategy,
            feature_translation=feature_translation,
            seed=feature_order_seed
        )
        duration = time.perf_counter() - start

        total_error = np.sqrt(np.power(recon_errors, 2).sum()).item()
        total_feature_norm = np.sqrt(np.power(feature_norms, 2).sum()).item()
        total_rel_error = total_error / total_feature_norm if total_feature_norm > 0.0 else 0.0
        max_rel_error = np.max(recon_errors / feature_norms).item() if len(recon_errors) > 0 else 0.0

        full_score, full_score_std = evaluate_clf(X_data, y, seed=seed)
        red_score, red_score_std = evaluate_clf(pruned_x, y, seed=seed)

        abs_scores.append({
            "ref_val": utils.DATASET_INFO[dataset].acc_ref,
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
            "dataset": dataset,
            "sorting_strategy": sorting_strategy,
            "feature_order_seed": feature_order_seed,
            "total_error": total_error,
            "total_feature_norm": total_feature_norm,
            "total_rel_error": total_rel_error,
            "max_rel_error": max_rel_error
        })

        rel_scores[dataset] = red_score / full_score

    return pd.DataFrame(abs_scores), pd.Series(rel_scores)


def enrich_scores(scores: List[Dict], kwargs: Dict):
    for score in scores:
        score.update(kwargs)
    return scores


def run_classification_evaluation_on_precomputed_features(
        *,
        num_reps: int = 5,
        seed: int,
        use_factorize_categorical: bool,
        verbose: bool = False
):
    if verbose:
        print("Classification evaluation")
    all_scores = []
    mode = "factorize" if use_factorize_categorical else "dummy"
    dataset_info = copy.deepcopy(utils.DATASET_INFO)

    class PlaceholderObject(object):
        pass

    fake_info = PlaceholderObject()
    fake_info.uci_id = 'isolet'
    dataset_info['isolet'] = fake_info

    for dataset, info in tqdm.tqdm(dataset_info.items(), total=len(dataset_info)):
        if verbose:
            print(f"Running clustering for dataset {dataset}")
        if dataset == 'isolet':
            X_data, y = utils.load_isolet()
        else:
            continue
            X_data, X_orig, y = utils.load_dataset(info.uci_id, use_factorize_categorical=use_factorize_categorical)
        full_dims = X_data.shape[1]

        scores = evaluate_clf(X_data, y, seed=seed, num_reps=num_reps)
        scores = enrich_scores(
            scores,
            kwargs={
                "dataset": dataset,
                "method": "baseline_full",
                "duration": 0.0,
                "dim_ratio": 1.0,
                "full_dim": full_dims,
                "red_dim": full_dims
            }
        )
        all_scores += scores
        if dataset == 'isolet':
            glob_paths = glob.glob(f"baseline_features/{info.uci_id}/*/*.mat")
        else:
            glob_paths = glob.glob(f"baseline_features/{info.uci_id}/{mode}/*/*.mat")
        for mat_file_path in tqdm.tqdm(glob_paths):
            data = loadmat(mat_file_path)
            method = os.path.split(os.path.dirname(mat_file_path))[-1]
            X_red = data['X_red']
            nan_cols = np.isnan(X_red).any(axis=0)
            X_red = X_red[:, ~nan_cols]

            scores = evaluate_clf(X_red, y, seed=seed, num_reps=num_reps)
            red_dims = X_red.shape[1]
            scores = enrich_scores(
                scores,
                kwargs={
                    "dataset": dataset,
                    "method": method,
                    "duration": data["duration"].item(),
                    "dim_ratio": float(red_dims) / float(full_dims),
                    "full_dim": full_dims,
                    "red_dim": red_dims
                }
            )

            all_scores += scores

    return pd.DataFrame(all_scores)
