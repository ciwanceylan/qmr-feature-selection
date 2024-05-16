import os
import dataclasses as dc
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo

DATA_CACHE_LOCATION = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'datasets'))


@dc.dataclass(frozen=True)
class DatasetInfo:
    uci_id: int
    pretty_name: str
    acc_ref: Optional[float]
    nmi_ref: Optional[float]


DATASET_INFO = {
    "automobile": DatasetInfo(uci_id=10, pretty_name="Automobile", acc_ref=0.693, nmi_ref=0.244),
    "breast_cancer": DatasetInfo(uci_id=14, pretty_name="Breast-cancer", acc_ref=0.685, nmi_ref=0.021),
    "heart-c": DatasetInfo(uci_id=45, pretty_name="Heart-c", acc_ref=0.832, nmi_ref=0.292),
    "heart-statlog": DatasetInfo(uci_id=145, pretty_name="Heart-statlog", acc_ref=0.837, nmi_ref=0.270),
    "hepatitis": DatasetInfo(uci_id=46, pretty_name="Hepatitis", acc_ref=0.845, nmi_ref=0.107),
    "ionosphere": DatasetInfo(uci_id=52, pretty_name="Ionosphere", acc_ref=0.883, nmi_ref=0.128),
    # "liver-disorders": DatasetInfo(uci_id=60, pretty_name="Liver-disorders", acc_ref=0.580, nmi_ref=0.0),
    # "lung-cancer": DatasetInfo(uci_id=62, pretty_name="Lung-cancer", acc_ref=0.500, nmi_ref=0.279),
    "lymphography": DatasetInfo(uci_id=63, pretty_name="Lymphography", acc_ref=0.838, nmi_ref=0.132),
    # "monks-problems-2-train":
    "sonar": DatasetInfo(uci_id=151, pretty_name="Sonar", acc_ref=0.755, nmi_ref=0.007),
    # "soybean":
    "wdbc": DatasetInfo(uci_id=17, pretty_name="Wdbc", acc_ref=0.977, nmi_ref=0.611),
    "wine": DatasetInfo(uci_id=109, pretty_name="Wine", acc_ref=0.989, nmi_ref=0.835),
    "zoo": DatasetInfo(uci_id=111, pretty_name="Zoo", acc_ref=0.960, nmi_ref=0.752),
}

US_CENSUS_DATASET = DatasetInfo(uci_id=116, pretty_name="US Census Data (1990)", acc_ref=None, nmi_ref=None)

BASELINE_METHODS = [
    "SVD-entropy", "LS", "SPEC", "USFSM", "FSFS", "RRFS", "UDFS", "NDFS", "UFSACO", "MGSACO", "DSRMR", "LLC-fs",
    "DGUFS", "Li et al.", "LS-WNCH-BE"]

CLASSIFICATION_RESULTS = {
    "automobile": [0.498, 0.478, 0.478, 0.595, 0.659, 0.668, 0.507, 0.566, 0.678, 0.668, 0.561, 0.517, 0.683, 0.693,
                   0.673],
    "breast_cancer": [0.710, 0.724, 0.703, 0.696, 0.717, 0.692, 0.706, 0.710, 0.713, 0.696, 0.699, 0.703, 0.713, 0.678,
                      0.685],
    "heart-c": [0.822, 0.818, 0.835, 0.845, 0.792, 0.802, 0.815, 0.812, 0.785, 0.818, 0.809, 0.802, 0.759, 0.819,
                0.815],
    "heart-statlog": [0.830, 0.778, 0.826, 0.763, 0.778, 0.826, 0.778, 0.822, 0.833, 0.796, 0.830, 0.804, 0.807, 0.793,
                      0.726],
    "hepatitis": [0.852, 0.852, 0.858, 0.832, 0.839, 0.826, 0.858, 0.871, 0.839, 0.839, 0.845, 0.845, 0.819, 0.852,
                  0.845],
    "ionosphere": [0.866, 0.832, 0.789, 0.889, 0.852, 0.866, 0.869, 0.880, 0.889, 0.866, 0.886, 0.866, 0.889, 0.874,
                   0.852],
    # "liver-disorders": [0.580, 0.580, 0.580, 0.580, 0.580, 0.580, 0.580, 0.580, 0.580, 0.580, 0.580, 0.580, 0.580, 0.580, 0.580],
    # "lung-cancer": [0.529, 0.476, 0.467, 0.533, 0.471, 0.505, 0.533, 0.476, 0.471, 0.529, 0.443, 0.538, 0.433, 0.510, 0.476],
    "lymphography": [0.784, 0.777, 0.831, 0.845, 0.764, 0.689, 0.804, 0.837, 0.817, 0.845, 0.804, 0.845, 0.797, 0.838,
                     0.616],
    # "monks-problems-2-train": [0.621, 0.621, 0.598, 0.621, 0.610, 0.598, 0.621, 0.610, 0.621, 0.598, 0.610, 0.610, 0.621, 0.621, 0.621]
    "sonar": [0.798, 0.789, 0.755, 0.759, 0.760, 0.746, 0.774, 0.774, 0.779, 0.784, 0.765, 0.765, 0.779, 0.702, 0.755],
    # "soybean": [0.871, 0.880, 0.892, 0.898, 0.898, 0.862, 0.873, 0.868, 0.889, 0.921, 0.896, 0.925, 0.902, 0.912, 0.937]
    "wdbc": [0.944, 0.952, 0.965, 0.952, 0.958, 0.963, 0.951, 0.967, 0.965, 0.961, 0.956, 0.975, 0.972, 0.949, 0.935],
    "wine": [0.955, 0.955, 0.955, 0.949, 0.933, 0.921, 0.926, 0.966, 0.938, 0.944, 0.938, 0.972, 0.938, 0.910, 0.961],
    "zoo": [0.941, 0.940, 0.891, 0.950, 0.871, 0.831, 0.921, 0.921, 0.881, 0.921, 0.920, 0.930, 0.921, 0.960, 0.960],
}

CLUSTERING_RESULTS = {
    "automobile": [0.188, 0.166, 0.181, 0.264, 0.249, 0.183, 0.257, 0.189, 0.245, 0.192, 0.162, 0.180, 0.266, 0.244,
                   0.213],
    "breast_cancer": [0.024, 0.010, 0.028, 0.015, 0.004, 0.007, 0.031, 0.006, 0.008, 0.022, 0.004, 0.009, 0.013, 0.003,
                      0.021],
    "heart-c": [0.246, 0.272, 0.256, 0.176, 0.117, 0.268, 0.279, 0.255, 0.240, 0.247, 0.202, 0.247, 0.175, 0.156,
                0.147],
    "heart-statlog": [0.254, 0.220, 0.190, 0.230, 0.172, 0.270, 0.190, 0.254, 0.097, 0.168, 0.205, 0.024, 0.103, 0.190,
                      0.266],
    "hepatitis": [0.126, 0.130, 0.192, 0.157, 0.106, 0.107, 0.072, 0.071, 0.056, 0.100, 0.177, 0.077, 0.094, 0.195,
                  0.107],
    "ionosphere": [0.126, 0.126, 0.108, 0.112, 0.183, 0.082, 0.131, 0.131, 0.102, 0.141, 0.130, 0.129, 0.095, 0.092,
                   0.126],
    # "liver-disorders": [0.001, 0.001, 0.003, 0.000, 0.001, 0.002, 0.001, 0.003, 0.010, 0.002, 0.001, 0.001, 0.000, 0.000, 0.001],
    # "lung-cancer": [0.282, 0.271, 0.176, 0.243, 0.179, 0.147, 0.274, 0.217, 0.214, 0.236, 0.224, 0.203, 0.180, 0.241, 0.221],
    "lymphography": [0.144, 0.122, 0.138, 0.111, 0.148, 0.055, 0.125, 0.117, 0.185, 0.149, 0.119, 0.131, 0.103, 0.137,
                     0.042],
    # "monks-problems-2-train": [0.002, 0.017, 0.008, 0.003, 0.004, 0.008, 0.012, 0.007, 0.017, 0.007, 0.017, 0.003, 0.012, 0.001, 0.001]
    "sonar": [0.007, 0.018, 0.010, 0.013, 0.019, 0.023, 0.007, 0.008, 0.042, 0.012, 0.012, 0.005, 0.022, 0.002, 0.007],
    # "soybean": [0.711, 0.695, 0.724, 0.700, 0.495, 0.481, 0.707, 0.731, 0.612, 0.625, 0.663, 0.577, 0.487, 0.607, 0.645]
    "wdbc": [0.625, 0.619, 0.660, 0.641, 0.719, 0.753, 0.611, 0.605, 0.686, 0.611, 0.538, 0.672, 0.579, 0.581, 0.584],
    "wine": [0.846, 0.846, 0.783, 0.743, 0.697, 0.800, 0.835, 0.889, 0.798, 0.706, 0.815, 0.787, 0.760, 0.704, 0.808],
    "zoo": [0.882, 0.881, 0.667, 0.645, 0.748, 0.425, 0.901, 0.738, 0.803, 0.759, 0.817, 0.703, 0.658, 0.773, 0.752],
}


def load_classification_results(use_ratio: bool):
    df = pd.DataFrame(data=CLASSIFICATION_RESULTS, index=BASELINE_METHODS)
    if use_ratio:
        baseline = pd.Series(
            {dataset: info.acc_ref for dataset, info in DATASET_INFO.items()}
        )
        df = df.div(baseline)
    return df


def load_clustering_results(use_ratio: bool):
    df = pd.DataFrame(data=CLUSTERING_RESULTS, index=BASELINE_METHODS)
    if use_ratio:
        baseline = pd.Series(
            {dataset: info.nmi_ref for dataset, info in DATASET_INFO.items()}
        )
        df = df.div(baseline)
    return df


def get_features_and_targets(dataset) -> (np.ndarray, pd.DataFrame, np.ndarray):
    cat_cols = {col: "category" for col in dataset.variables.loc[
        (dataset.variables["type"].isin({"Binary", "Categorical"})) & (dataset.variables["role"] == "Feature")][
        "name"].tolist()}
    X_orig = dataset.data.features.astype(cat_cols)
    y, _ = pd.factorize(dataset.data.targets.squeeze())
    X_data = pd.get_dummies(X_orig).astype(float)
    X_data = SimpleImputer().fit_transform(X_data)
    return X_data, X_orig, y


def load_dataset(dataset_id: int, use_cache: bool = True):
    cache_folder = os.path.join(DATA_CACHE_LOCATION, str(dataset_id))
    if use_cache and os.path.exists(cache_folder) and len(os.listdir(cache_folder)) > 0:
        X_data = np.load(os.path.join(cache_folder, "preprocessed_data.npy"))
        X_orig = pd.read_json(os.path.join(cache_folder, "original_data.json"), orient='split')
        y = np.load(os.path.join(cache_folder, "targets.npy"))
    else:
        dataset = fetch_ucirepo(id=dataset_id)
        X_data, X_orig, y = get_features_and_targets(dataset)
        if use_cache:
            os.makedirs(cache_folder, exist_ok=True)
            np.save(os.path.join(cache_folder, "preprocessed_data.npy"), X_data)
            X_orig.to_json(os.path.join(cache_folder, "original_data.json"), orient='split')
            np.save(os.path.join(cache_folder, "targets.npy"), y)
    return X_data, X_orig, y

#
# if __name__ == "__main__":
#     print(load_clustering_results(True))
