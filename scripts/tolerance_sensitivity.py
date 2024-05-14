import os
import pandas as pd
import tqdm

from qmrfs.experiments.classification import run_classification_experiment
from qmrfs.experiments.clustering import run_clustering_experiment


def main_classification(seed: int):
    folder = "results/data/tol_sens"
    os.makedirs("results/data/tol_sens", exist_ok=True)
    save_path = os.path.join(folder, "classification.json")
    tolerances = [0.7, 0.6, 0.5, 0.3, 0.2, 0.1, 0.075, 0.05, 1e-2, 5e-3, 1e-3, 1e-4]
    all_results = []
    for tol in tqdm.tqdm(tolerances):
        abs_results, _ = run_classification_experiment(tolerance=tol, seed=seed)
        all_results.append(abs_results)

    all_results = pd.concat(all_results, axis=0, ignore_index=True)
    all_results.to_json(save_path, orient='records', indent=2)


def main_clustering(seed: int):
    folder = "results/data/tol_sens"
    os.makedirs("results/data/tol_sens", exist_ok=True)
    save_path = os.path.join(folder, "clustering.json")
    tolerances = [0.7, 0.6, 0.5, 0.3, 0.2, 0.1, 0.075, 0.05, 1e-2, 5e-3, 1e-3, 1e-4]
    all_results = []
    for tol in tqdm.tqdm(tolerances):
        abs_results, _ = run_clustering_experiment(tolerance=tol, seed=seed)
        all_results.append(abs_results)

    all_results = pd.concat(all_results, axis=0, ignore_index=True)
    all_results.to_json(save_path, orient='records', indent=2)


if __name__ == "__main__":
    seed = 6342312

    main_classification(seed=seed)
    main_clustering(seed=seed)
