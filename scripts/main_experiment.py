import os

import qmrfs.experiments.classification as classification
import qmrfs.experiments.clustering as clustering


def main_classification(use_factorize_categorical: bool, seed: int):
    if use_factorize_categorical:
        folder = f"results/data/main_experiment/factorize/"
    else:
        folder = f"results/data/main_experiment/dummy/"
    os.makedirs(folder, exist_ok=True)
    df = classification.run_classification_evaluation_on_precomputed_features(
        num_reps=5,
        seed=seed,
        use_factorize_categorical=use_factorize_categorical,
        verbose=True
    )
    df.to_json(os.path.join(folder, "classification.json"), orient='records', indent=2)


def main_clustering(use_factorize_categorical: bool, seed: int):
    if use_factorize_categorical:
        folder = f"results/data/main_experiment/factorize/"
    else:
        folder = f"results/data/main_experiment/dummy/"
    os.makedirs(folder, exist_ok=True)
    df = clustering.run_clustering_evaluation_on_precomputed_features(
        num_reps=25,
        seed=seed,
        use_factorize_categorical=use_factorize_categorical,
        verbose=True
    )
    df.to_json(os.path.join(folder, "clustering.json"), orient='records', indent=2)


if __name__ == "__main__":
    seed = 6342312

    main_classification(use_factorize_categorical=True, seed=seed)
    main_clustering(use_factorize_categorical=True, seed=seed)
