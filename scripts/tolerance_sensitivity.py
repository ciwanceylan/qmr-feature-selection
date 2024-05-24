from typing import Optional
import os
import pandas as pd
import numpy as np
import tqdm

from qmrfs.experiments.classification import run_classification_experiment
from qmrfs.experiments.clustering import run_clustering_experiment


def main_classification(*, sorting_strategy, seed: int, feature_order_seed: Optional[int],
                        use_factorize_categorical: bool):
    if use_factorize_categorical:
        folder = f"results/data/tol_sens/factorize/{sorting_strategy}"
    else:
        folder = f"results/data/tol_sens/dummy/{sorting_strategy}"

    if 'random' in sorting_strategy:
        folder += f"_{feature_order_seed}"

    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, "classification.json")
    tolerances = np.arange(0.9, 0.075, -0.05).tolist() + [0.075, 0.05, 1e-2, 5e-3, 1e-3, 1e-4]
    all_results = []
    for tol in tqdm.tqdm(tolerances):
        abs_results, _ = run_classification_experiment(tolerance=tol, sorting_strategy=sorting_strategy,
                                                       seed=seed, feature_order_seed=feature_order_seed,
                                                       use_factorize_categorical=use_factorize_categorical)
        all_results.append(abs_results)

    all_results = pd.concat(all_results, axis=0, ignore_index=True)
    all_results.to_json(save_path, orient='records', indent=2)


def main_clustering(*, sorting_strategy, seed: int, feature_order_seed: Optional[int],
                    use_factorize_categorical: bool):
    if use_factorize_categorical:
        folder = f"results/data/tol_sens/factorize/{sorting_strategy}"
    else:
        folder = f"results/data/tol_sens/dummy/{sorting_strategy}"

    if 'random' in sorting_strategy:
        folder += f"_{feature_order_seed}"
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, "clustering.json")
    tolerances = np.arange(0.9, 0.05, -0.05).tolist() + [0.05, 1e-2, 5e-3, 1e-3, 1e-4]
    all_results = []
    for tol in tqdm.tqdm(tolerances):
        abs_results, _ = run_clustering_experiment(tolerance=tol, sorting_strategy=sorting_strategy,
                                                   seed=seed, feature_order_seed=feature_order_seed,
                                                   use_factorize_categorical=use_factorize_categorical)
        all_results.append(abs_results)

    all_results = pd.concat(all_results, axis=0, ignore_index=True)
    all_results.to_json(save_path, orient='records', indent=2)


if __name__ == "__main__":
    seed = 6342312
    feature_order_seed_1 = 534643
    feature_order_seed_2 = 6782343
    feature_order_seed_3 = 92384234
    feature_order_seed_4 = 12352312
    feature_order_seed_5 = 8239423

    main_classification(sorting_strategy='random', seed=seed, feature_order_seed=feature_order_seed_1,
                        use_factorize_categorical=True)
    main_classification(sorting_strategy='random', seed=seed, feature_order_seed=feature_order_seed_2,
                        use_factorize_categorical=True)
    main_classification(sorting_strategy='random', seed=seed, feature_order_seed=feature_order_seed_3,
                        use_factorize_categorical=True)
    main_classification(sorting_strategy='random', seed=seed, feature_order_seed=feature_order_seed_4,
                        use_factorize_categorical=True)
    main_classification(sorting_strategy='random', seed=seed, feature_order_seed=feature_order_seed_5,
                        use_factorize_categorical=True)
    main_classification(sorting_strategy='default', seed=seed, feature_order_seed=None,
                        use_factorize_categorical=True)
    main_classification(sorting_strategy='entropy_high2low', seed=seed, feature_order_seed=None,
                        use_factorize_categorical=True)

    main_clustering(sorting_strategy='random', seed=seed, feature_order_seed=feature_order_seed_1,
                    use_factorize_categorical=True)
    main_clustering(sorting_strategy='random', seed=seed, feature_order_seed=feature_order_seed_2,
                    use_factorize_categorical=True)
    main_clustering(sorting_strategy='random', seed=seed, feature_order_seed=feature_order_seed_3,
                    use_factorize_categorical=True)
    main_clustering(sorting_strategy='random', seed=seed, feature_order_seed=feature_order_seed_4,
                    use_factorize_categorical=True)
    main_clustering(sorting_strategy='random', seed=seed, feature_order_seed=feature_order_seed_5,
                    use_factorize_categorical=True)
    main_clustering(sorting_strategy='default', seed=seed, feature_order_seed=None,
                    use_factorize_categorical=True)
    main_clustering(sorting_strategy='entropy_high2low', seed=seed, feature_order_seed=None,
                    use_factorize_categorical=True)
