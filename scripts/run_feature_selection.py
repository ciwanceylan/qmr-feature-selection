import os
import time
import tqdm
import numpy as np
from scipy.io import savemat
import qmrfs.experiments.utils as utils
from qmrfs.experiments.utils import DATASET_INFO, DATA_CACHE_LOCATION, load_dataset
# from qmrfs.baselines.fsfs import fsfs
from qmrfs.baselines.svd_entropy import svd_entropy_sr
from qmrfs.baselines.usfsm import usfsm
import qmrfs.qmr_feature_selection as qmrfs


def save_datafiles_as_mat():
    for dataset, info in tqdm.tqdm(DATASET_INFO.items(), total=len(DATASET_INFO)):
        savefolder = os.path.join(DATA_CACHE_LOCATION, str(info.uci_id))
        factorized_data, _, y = utils.load_dataset(info.uci_id, use_cache=True, use_factorize_categorical=True)
        savemat(os.path.join(savefolder, "factorized_pp_data.mat"), {"data": factorized_data})

        dummy_data, _, y = utils.load_dataset(info.uci_id, use_cache=True, use_factorize_categorical=False)
        savemat(os.path.join(savefolder, "dummy_pp_data.mat"), {"data": dummy_data})


def run_save_baseline_selected_features(use_factorize_categorical: bool):
    models = {"svd_entropy": svd_entropy_sr, "usfsm": usfsm}
    for dataset, info in tqdm.tqdm(DATASET_INFO.items(), total=len(DATASET_INFO)):
        for model_name, model in tqdm.tqdm(models.items()):
            if use_factorize_categorical:
                save_folder = os.path.join(f"baseline_features/{info.uci_id}/factorize/{model_name}")
            else:
                save_folder = os.path.join(f"baseline_features/{info.uci_id}/dummy/{model_name}")
            os.makedirs(save_folder, exist_ok=True)
            X_data, X_orig, y = load_dataset(info.uci_id, use_factorize_categorical=use_factorize_categorical)
            for p in np.arange(0.2, 1., 0.1):
                k = int(np.round(p * X_data.shape[1]))
                start = time.perf_counter()
                pruned_x, _ = model(X_data, k)
                duration = time.perf_counter() - start
                res = {
                    "X_red": pruned_x,
                    "dataset": dataset,
                    "method": model_name,
                    "duration": duration,
                    "num_feat": X_data.shape[1],
                    "num_red_feat": pruned_x.shape[1],
                    "feat_ratio": pruned_x.shape[1] / X_data.shape[1],
                }
                savemat(os.path.join(save_folder, f"features_{int(100 * p)}.mat"), res)


def run_save_qmrfs_selected_features(use_factorize_categorical: bool):
    for dataset, info in tqdm.tqdm(DATASET_INFO.items(), total=len(DATASET_INFO)):
        if use_factorize_categorical:
            save_folder = os.path.join(f"baseline_features/{info.uci_id}/factorize/qmrfs")
        else:
            save_folder = os.path.join(f"baseline_features/{info.uci_id}/dummy/qmrfs")
        os.makedirs(save_folder, exist_ok=True)
        X_data, X_orig, y = load_dataset(info.uci_id, use_factorize_categorical=use_factorize_categorical)
        tolerances = np.concatenate((np.linspace(0.95, 0.05, 13), np.asarray([1e-2, 5e-3, 1e-3])))
        for tol in tolerances:
            start = time.perf_counter()
            pruned_x, _, _ = qmrfs.qmr_fs(X_data, tolerance=tol, sorting_strategy='entropy_high2low',
                                          feature_translation='non-negative')
            duration = time.perf_counter() - start
            res = {
                "X_red": pruned_x,
                "dataset": dataset,
                "method": "qmrfs",
                "duration": duration,
                "num_feat": X_data.shape[1],
                "num_red_feat": pruned_x.shape[1],
                "feat_ratio": pruned_x.shape[1] / X_data.shape[1],
            }
            savemat(os.path.join(save_folder, f"features_{int(1000 * tol)}.mat"), res)


if __name__ == "__main__":
    # save_datafiles_as_mat()
    # run_save_baseline_selected_features(use_factorize_categorical=True)
    # run_save_baseline_selected_features(use_factorize_categorical=False)

    run_save_qmrfs_selected_features(use_factorize_categorical=True)
    run_save_qmrfs_selected_features(use_factorize_categorical=False)
