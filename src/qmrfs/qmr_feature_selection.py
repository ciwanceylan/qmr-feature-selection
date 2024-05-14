import numpy as np
import torch
import qmrfs.core as core


class QMRFeatureSelector:

    def __init__(self, tolerance: float):
        self.tolerance = tolerance

    def fit_transform(self, features: torch.Tensor):
        columns_to_keep_mask = core.get_keep_columns_qr(features, tolerance=self.tolerance)
        pruned_features = features[:, columns_to_keep_mask]
        return pruned_features, columns_to_keep_mask


def qmr_fs(features, tolerance: float):
    features = torch.from_numpy(features)
    selector = QMRFeatureSelector(tolerance=tolerance)
    selected_features, columns_to_keep_mask = selector.fit_transform(features)
    return selected_features.numpy()

