import os

from qmrfs.experiments.utils import load_classification_results, load_clustering_results
from qmrfs.experiments.classification import run_classification_experiment
from qmrfs.experiments.clustering import run_clustering_experiment


def main_classification(tolerance: float, sorting_strategy, seed: int, feature_order_seed: int):
    folder = "results/data/main_experiment/classification"
    os.makedirs(folder, exist_ok=True)

    abs_results, rel_results = run_classification_experiment(tolerance=tolerance, sorting_strategy=sorting_strategy,
                                                             seed=seed, feature_order_seed=feature_order_seed)
    full_rel_df = load_classification_results(use_ratio=True)
    full_rel_df.loc["qmr"] = rel_results

    ranks = full_rel_df.rank(axis=0, ascending=False)
    ranks_avg = ranks.mean(axis=1)
    ranks_std = ranks.std(axis=1)
    full_rel_df["avg_rank"] = ranks_avg
    full_rel_df["rank_std"] = ranks_std
    full_rel_df.to_json(os.path.join(folder, "rel_and_rank.json"), indent=2)
    abs_results.to_json(os.path.join(folder, "full_data.json"), orient='records', indent=2)


def main_clustering(tolerance: float, sorting_strategy, seed: int, feature_order_seed: int):
    folder = "results/data/main_experiment/clustering"
    os.makedirs(folder, exist_ok=True)

    abs_results, rel_results = run_clustering_experiment(tolerance=tolerance, sorting_strategy=sorting_strategy,
                                                         seed=seed, feature_order_seed=feature_order_seed)
    full_rel_df = load_clustering_results(use_ratio=True)
    full_rel_df.loc["qmr"] = rel_results

    ranks = full_rel_df.rank(axis=0, ascending=False)
    ranks_avg = ranks.mean(axis=1)
    ranks_std = ranks.std(axis=1)
    full_rel_df["avg_rank"] = ranks_avg
    full_rel_df["rank_std"] = ranks_std
    full_rel_df.to_json(os.path.join(folder, "rel_and_rank.json"), indent=2)
    abs_results.to_json(os.path.join(folder, "full_data.json"), orient='records', indent=2)


if __name__ == "__main__":
    tolerance = 1e-1
    seed = 6342312
    main_classification(tolerance=tolerance, seed=seed)
    main_clustering(tolerance=tolerance, seed=seed)
