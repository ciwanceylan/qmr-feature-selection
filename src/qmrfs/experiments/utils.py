import os
import dataclasses as dc
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
from scipy.io import loadmat

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
    "heart-c": DatasetInfo(uci_id=45, pretty_name="Heart-c", acc_ref=0.832, nmi_ref=0.292),
    "heart-statlog": DatasetInfo(uci_id=145, pretty_name="Heart-statlog", acc_ref=0.837, nmi_ref=0.270),
    "sonar": DatasetInfo(uci_id=151, pretty_name="Sonar", acc_ref=0.755, nmi_ref=0.007),
    "wine": DatasetInfo(uci_id=109, pretty_name="Wine", acc_ref=0.989, nmi_ref=0.835),
}

US_CENSUS_DATASET = DatasetInfo(uci_id=116, pretty_name="US Census Data (1990)", acc_ref=None, nmi_ref=None)


def fix_data_dtypes(dataset):
    cat_cols = {col: "category" for col in dataset.variables.loc[
        (dataset.variables["type"].isin({"Binary", "Categorical"})) & (dataset.variables["role"] == "Feature")][
        "name"].tolist()}
    X_orig = dataset.data.features.astype(cat_cols)
    return X_orig


def get_features(X_orig, use_factorize_categorical: bool) -> np.ndarray:
    X_orig = X_orig.loc[:, ~X_orig.columns.duplicated()].copy()
    X_orig = X_orig.loc[:, ~X_orig.isna().all(axis=0)].copy()
    if use_factorize_categorical:
        X_data = X_orig.copy()
        for col in X_data.columns:
            if X_data[col].dtype == 'category':
                factorized, _ = pd.factorize(X_data[col])
                X_data[col] = factorized.astype(float)
    else:
        X_data = pd.get_dummies(X_orig).astype(float)
    X_data = SimpleImputer().fit_transform(X_data.astype(float))
    return X_data


def load_dataset(dataset_id: int, use_cache: bool = True, use_factorize_categorical: bool = False):
    cache_folder = os.path.join(DATA_CACHE_LOCATION, str(dataset_id))
    os.makedirs(cache_folder, exist_ok=True)
    path_to_data_with_dtypes = os.path.join(cache_folder, "original_data.pkl")
    path_factorize_cat = os.path.join(cache_folder, "preprocessed_data_factorized.npy")
    path_dummy_cat = os.path.join(cache_folder, "preprocessed_data.npy")
    if use_cache and os.path.exists(path_to_data_with_dtypes):
        X_orig = pd.read_pickle(path_to_data_with_dtypes)
        y = np.load(os.path.join(cache_folder, "targets.npy"))
    else:
        dataset = fetch_ucirepo(id=dataset_id)
        X_orig = fix_data_dtypes(dataset)
        targets = dataset.data.targets
        if targets is not None:
            y, _ = pd.factorize(targets.squeeze())
        else:
            y = np.asarray([])
        if use_cache:
            X_orig.to_pickle(path_to_data_with_dtypes)
            np.save(os.path.join(cache_folder, "targets.npy"), y)

    if use_cache and use_factorize_categorical and os.path.exists(path_factorize_cat):
        X_data = np.load(path_factorize_cat)
    elif use_cache and not use_factorize_categorical and os.path.exists(path_dummy_cat):
        X_data = np.load(path_dummy_cat)
    else:
        X_data = get_features(X_orig, use_factorize_categorical=use_factorize_categorical)

        if use_cache:
            os.makedirs(cache_folder, exist_ok=True)
            if use_factorize_categorical:
                np.save(path_factorize_cat, X_data)
            else:
                np.save(path_dummy_cat, X_data)

    return X_data, X_orig, y


def load_isolet():
    data = loadmat("datasets/isolet/Isolet1.mat")
    X_data = data['X']
    y = data['Y'].squeeze()
    return X_data, y
