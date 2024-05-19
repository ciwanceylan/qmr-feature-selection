from typing import Optional, Union, Literal
import os

from qmrfs.experiments.utils import load_classification_results, load_clustering_results, DATASET2THETA
import qmrfs.experiments.classification as classification
import qmrfs.experiments.clustering as clustering
from qmrfs.qmr_feature_selection import TranslationMode


# def main_classification(tolerance: Union[float, Literal['auto']], sorting_strategy, seed: int,
#                         feature_order_seed: Optional[int], use_factorize_categorical: bool,
#                         feature_translation: TranslationMode):
#     if use_factorize_categorical:
#         folder = f"results/data/main_experiment/factorize/classification/{feature_translation}/{sorting_strategy}"
#     else:
#         folder = f"results/data/main_experiment/dummy/classification/{feature_translation}/{sorting_strategy}"
#     if tolerance == 'auto':
#         folder += "_auto"
#     else:
#         folder += f"_{tolerance:.2e}".replace('.', '-')
#     os.makedirs(folder, exist_ok=True)
#
#     abs_results, rel_results = run_classification_experiment(tolerance=tolerance, sorting_strategy=sorting_strategy,
#                                                              seed=seed, feature_order_seed=feature_order_seed,
#                                                              use_factorize_categorical=use_factorize_categorical,
#                                                              feature_translation=feature_translation)
#     full_rel_df = load_classification_results(use_ratio=True)
#     full_rel_df.loc["qmr"] = rel_results
#
#     ranks = full_rel_df.rank(axis=0, ascending=False)
#     ranks_avg = ranks.mean(axis=1)
#     ranks_std = ranks.std(axis=1)
#     full_rel_df["avg_rank"] = ranks_avg
#     full_rel_df["rank_std"] = ranks_std
#     full_rel_df.to_json(os.path.join(folder, "rel_and_rank.json"), indent=2)
#     abs_results.to_json(os.path.join(folder, "full_data.json"), orient='records', indent=2)
#
#
# def main_clustering(tolerance: Union[float, Literal['auto']], sorting_strategy, seed: int,
#                     feature_order_seed: Optional[int], use_factorize_categorical: bool,
#                     feature_translation: TranslationMode):
#     if use_factorize_categorical:
#         folder = f"results/data/main_experiment/factorize/clustering/{feature_translation}/{sorting_strategy}"
#     else:
#         folder = f"results/data/main_experiment/dummy/clustering/{feature_translation}/{sorting_strategy}"
#     if tolerance == 'auto':
#         folder += "_auto"
#     else:
#         folder += f"_{tolerance:.2e}".replace('.', '-')
#
#     os.makedirs(folder, exist_ok=True)
#
#     abs_results, rel_results = clustering.run_clustering_experiment(tolerance=tolerance, sorting_strategy=sorting_strategy,
#                                                          use_factorize_categorical=use_factorize_categorical,
#                                                          seed=seed, feature_order_seed=feature_order_seed,
#                                                          feature_translation=feature_translation)
#     full_rel_df = load_clustering_results(use_ratio=True)
#     full_rel_df.loc["qmr"] = rel_results
#
#     ranks = full_rel_df.rank(axis=0, ascending=False)
#     ranks_avg = ranks.mean(axis=1)
#     ranks_std = ranks.std(axis=1)
#     full_rel_df["avg_rank"] = ranks_avg
#     full_rel_df["rank_std"] = ranks_std
#     full_rel_df.to_json(os.path.join(folder, "rel_and_rank.json"), indent=2)
#     abs_results.to_json(os.path.join(folder, "full_data.json"), orient='records', indent=2)

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
    # tolerance = 'auto'
    seed = 6342312

    main_classification(use_factorize_categorical=True, seed=seed)
    main_clustering(use_factorize_categorical=True, seed=seed)
    # main_classification(use_factorize_categorical=False)
    # main_clustering(use_factorize_categorical=False)

    # main_classification(tolerance=tolerance, sorting_strategy='entropy_high2low',
    #                     seed=seed, feature_order_seed=None, use_factorize_categorical=True)
    # main_clustering(tolerance=tolerance, sorting_strategy='entropy_high2low',
    #                 seed=seed, feature_order_seed=None, use_factorize_categorical=True)

    # main_classification(tolerance=tolerance, sorting_strategy='entropy_high2low',
    #                     seed=seed, feature_order_seed=None, use_factorize_categorical=False)
    # main_clustering(tolerance=tolerance, sorting_strategy='entropy_high2low',
    #                 seed=seed, feature_order_seed=None, use_factorize_categorical=False)

    # tolerance = 0.0
    # main_classification(tolerance=tolerance, sorting_strategy='entropy_high2low',
    #                     seed=seed, feature_order_seed=None, use_factorize_categorical=True)
    # main_clustering(tolerance=tolerance, sorting_strategy='entropy_high2low',
    #                 seed=seed, feature_order_seed=None, use_factorize_categorical=True)
