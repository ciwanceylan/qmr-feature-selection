import json
import os
import time
import tqdm
import numpy as np
import pandas as pd
from scipy.io import savemat
import qmrfs.experiments.utils as utils
from qmrfs.experiments.utils import DATASET_INFO, DATA_CACHE_LOCATION, load_dataset
from qmrfs.baselines.fsfs import fsfs
from qmrfs.baselines.svd_entropy import svd_entropy_sr
from qmrfs.baselines.usfsm import usfsm


def save_datafiles_as_mat():
    for dataset, info in tqdm.tqdm(DATASET_INFO.items(), total=len(DATASET_INFO)):
        savefolder = os.path.join(DATA_CACHE_LOCATION, str(info.uci_id))
        factorized_data, _, y = utils.load_dataset(info.uci_id, use_cache=True, use_factorize_categorical=True)
        savemat(os.path.join(savefolder, "factorized_pp_data.mat"), {"data": factorized_data})

        dummy_data, _, y = utils.load_dataset(info.uci_id, use_cache=True, use_factorize_categorical=False)
        savemat(os.path.join(savefolder, "dummy_pp_data.mat"), {"data": dummy_data})


def run_save_baseline_selected_features(target_percentage: float, use_factorize_categorical: bool):
    models = {"fsfs": fsfs, "svd_entropy": svd_entropy_sr, "usfsm": usfsm}
    for dataset, info in tqdm.tqdm(DATASET_INFO.items(), total=len(DATASET_INFO)):
        for model_name, model in tqdm.tqdm(models.items()):
            if use_factorize_categorical:
                save_folder = os.path.join(f"baseline_features/{info.uci_id}/factorize/{model_name}")
            else:
                save_folder = os.path.join(f"baseline_features/{info.uci_id}/dummy/{model_name}")
            os.makedirs(save_folder, exist_ok=True)
            X_data, X_orig, y = load_dataset(info.uci_id, use_factorize_categorical=use_factorize_categorical)
            k = int(target_percentage * X_data.shape[1]) + 1
            start = time.perf_counter()
            pruned_x, _ = model(X_data, k)
            duration = time.perf_counter() - start
            res = {
                "dataset": dataset,
                "method": model_name,
                "duration": duration,
                "full_dim": X_data.shape[1],
                "red_dim": pruned_x.shape[1],
                "dim_ratio": pruned_x.shape[1] / X_data.shape[1],
            }

            np.save(os.path.join(save_folder, "features.npy"), pruned_x)
            with open(os.path.join(save_folder, "info.json"), 'w') as fp:
                json.dump(res, fp, indent=2)


if __name__ == "__main__":
    # save_datafiles_as_mat()
    run_save_baseline_selected_features(0.7, use_factorize_categorical=True)
    run_save_baseline_selected_features(0.7, use_factorize_categorical=False)