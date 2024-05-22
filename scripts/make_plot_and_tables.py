import os
from typing import Literal, Set, Collection, List
import pandas as pd
import numpy as np
import qmrfs.results.results_processing as resproc
from qmrfs.qmr_feature_selection import TranslationMode, SortingStrategy
import glob
import matplotlib.pyplot as plt
import seaborn as sns


def load_tol_sens_data(mode: Literal['classification', 'clustering'], cat_pp_mode: Literal['factorize', 'dummy'],
                       translate_mode: TranslationMode):
    results = []
    for path in glob.glob(f"./results/data/tol_sens/{cat_pp_mode}/{translate_mode}/*/", recursive=False):
        df = pd.read_json(os.path.join(path, mode + '.json'), orient='records')
        results.append(df)
    results = pd.concat(results, axis=0, ignore_index=True)
    return results


def make_dim_ratio_comparison_plots(mode: Literal['classification', 'clustering'],
                                    cat_pp_mode: Literal['factorize', 'dummy'],
                                    methods: List[str]):
    y_lims = {
        'classification': {
            'heart-statlog': (0.6, 0.92),
            'wine': (0.7, 1.05)
        },
        'clustering': {
            'automobile': (0.1, 0.3),
            'heart-c': (0.05, 0.25),
            'sonar': (-0.01, 0.088),
            'wine': (0.25, 1.05)
        }
    }
    data = pd.read_json(f"./results/data/main_experiment/{cat_pp_mode}/{mode}.json")

    # data = data.loc[(data['dim_ratio'] < 0.99) & (data['dim_ratio'] > 0.18)].copy()
    data['method'] = data['method'].map(lambda x: x[0] if isinstance(x, list) else x)
    data['percent_dim_kept'] = 100 * data['dim_ratio']

    y = "accuracy" if mode == "classification" else "nmi"
    y_label = "Accuracy" if mode == "classification" else "NMI"

    datasets = data["dataset"].unique()
    for dataset_name in datasets:
        save_folder = f"results/figures/comparison/{cat_pp_mode}/{dataset_name}/"
        plt_data = data.loc[data["dataset"] == dataset_name]

        if dataset_name == 'isolet':
            fig, ax = resproc.lineplot(
                plt_data, x="red_dim", x_label=r"\# features kept", y=y, y_label=y_label, x_lim=(19, 105),
                hue="method", errorbar="sd", legend_order=methods, fontsize_legend=34, fontsize=48,
                save_path=os.path.join(save_folder, mode), seperate_legend=True, baseline_method='baseline_full'
            )
            plt.close(fig)
        else:
            fig, ax = resproc.lineplot(
                plt_data, x="percent_dim_kept", x_label=r"\% features kept", y=y, y_label=y_label, x_lim=(18, 100),
                y_lim=y_lims[mode].get(dataset_name, None), hue="method", errorbar="sd", legend_order=methods,
                fontsize_legend=34, fontsize=48,
                save_path=os.path.join(save_folder, mode), seperate_legend=True, baseline_method='baseline_full'
            )
            plt.close(fig)


def plot_tol_sens_single_sorting_model(mode: Literal['classification', 'clustering'],
                                       cat_pp_mode: Literal['factorize', 'dummy'],
                                       sorting_model: SortingStrategy,
                                       translate_mode: TranslationMode,
                                       exclude_datasets: Collection[str] = None):
    if exclude_datasets is None:
        exclude_datasets = set()
    save_folder = f"results/figures/{sorting_model}/{cat_pp_mode}/{translate_mode}/{mode}"
    os.makedirs(save_folder, exist_ok=True)
    df = load_tol_sens_data(mode=mode, cat_pp_mode=cat_pp_mode, translate_mode=translate_mode)
    y_label = "Acc. ratio" if mode == 'classification' else "NMI ratio"

    df["sorting_model"] = df["sorting_strategy"]
    df.loc[df["sorting_strategy"].str.contains('random'), "sorting_model"] += df["feature_order_seed"].astype(str)

    df = df.loc[~df["dataset"].isin(exclude_datasets)].copy()
    df = df.loc[df["sorting_model"] == sorting_model].copy()

    resproc.lineplot(
        df, x="tolerance", x_label=r"$\theta$", y="rel_score", y_label=y_label,
        x_scale="log", y_lim=(0.0, 1.5), hue="dataset", errorbar=None,
        save_path=os.path.join(save_folder, "theta_score_" + sorting_model)
    )

    resproc.lineplot(
        df, x="tolerance", x_label=r"$\theta$", y="dim_ratio", y_label="Dim. ratio",
        y_lim=(0., 1.01), hue="dataset", errorbar='sd',
        save_path=os.path.join(save_folder, "theta_dim_ratio_" + sorting_model)
    )

    resproc.lineplot(
        df, x="dim_ratio", x_label="Dim. ratio", y="rel_score", y_label=y_label,
        hue="dataset", errorbar=None, y_lim=(0.0, 1.5),
        save_path=os.path.join(save_folder, "dim_ratio_score_" + sorting_model)
    )

    # resproc.scatterplot(
    #     df, x="dim_ratio", x_label=r"Dim. ratio", y="rel_score", y_label=y_label, hue="dataset",
    #     save_path=os.path.join(save_folder, "dim_ratio_" + sorting_model)
    # )

    # resproc.lmplot(
    #     df, x="dim_ratio", x_label=r"Dim. ratio", y="rel_score", y_label=y_label, hue="dataset",
    #     save_path=os.path.join(save_folder, "dim_ratio_" + sorting_model)
    # )


def plot_tol_sens_sorting(
        mode: Literal['classification', 'clustering'],
        cat_pp_mode: Literal['factorize', 'dummy'],
        sorting_strategies: Collection[SortingStrategy],
        translate_mode: TranslationMode,
        include_datasets: Collection[str],
        exclude_datasets: Collection[str]
):
    if exclude_datasets is None:
        exclude_datasets = set()
    save_folder = f"results/figures/sorting_strat/{cat_pp_mode}/{translate_mode}/{mode}"
    os.makedirs(save_folder, exist_ok=True)
    df = load_tol_sens_data(mode, cat_pp_mode=cat_pp_mode, translate_mode=translate_mode)
    y_label = "Acc. ratio" if mode == 'classification' else "NMI ratio"

    df["sorting_model"] = df["sorting_strategy"]
    df.loc[df["sorting_strategy"].str.contains('random'), "sorting_model"] += df["feature_order_seed"].map(
        lambda x: "" if np.isnan(x) else str(int(x)))

    df = df.loc[df["dataset"].isin(include_datasets)].copy()
    df = df.loc[~df["dataset"].isin(exclude_datasets)].copy()
    df = df.loc[df["sorting_strategy"].isin(sorting_strategies)].copy()

    resproc.lineplot(
        df, x="tolerance", x_label=r"$\theta$", y="rel_score", y_label=y_label,
        x_scale="log",
        # y_lim=(0.0, 1.5),
        hue="sorting_model", errorbar='se',
        save_path=os.path.join(save_folder, "sorting_theta"), legend_order=sorted(df["sorting_model"].unique())
    )

    # resproc.scatterplot(
    #     df, x="tolerance", x_label=r"$\theta$", y="max_rel_error", y_label="Max rel. error",
    #     x_scale="log", y_scale="log", x_lim=(1e-8, 1), y_lim=(1e-8, 1), hue="sorting_model",
    #     save_path=os.path.join(save_folder, "error_check")
    # )


# def plot_tolerance_sensitivity_cc_combined():
#     save_folder = "results/figures/"
#     os.makedirs(save_folder, exist_ok=True)
#     df_clf = pd.read_json(f"results/data/tol_sens/classification.json")
#     df_clf["mode"] = "Classification (Score=Accuracy)"
#     df_clstr = pd.read_json(f"results/data/tol_sens/clustering.json", orient='records')
#     df_clstr["mode"] = "Clustering (Score=NMI)"
#     df = pd.concat((df_clf, df_clstr), axis=0, ignore_index=True)
#
#     df["percent_dims_removed"] = 100 * (1. - df["dim_ratio"])
#
#     y_label = "Score ratio"
#
#     resproc.lineplot(
#         df, x="tolerance", x_label=r"$\theta$", y="rel_score", y_label=y_label,
#         x_scale="log", y_lim=(0.5, 1.2), hue="mode",
#         save_path=os.path.join(save_folder, "tolerance_sensitivity")
#     )
#
#     resproc.lineplot(
#         df, x="tolerance", x_label=r"$\theta$", y="percent_dims_removed", y_label=r"\% features removed",
#         x_scale="log", y_lim=(0., 100.), hue=None,
#         save_path=os.path.join(save_folder, "dimensions")
#     )


if __name__ == "__main__":
    datasets = ["automobile",
                # "breast_cancer",
                "heart-c",
                "heart-statlog",
                # "hepatitis",
                # "ionosphere",
                # "lymphography",
                "sonar",
                # "wdbc",
                "wine",
                # "zoo",
                "isolet"]

    methods = ["baseline_full", "qmrfs", "svd_entropy", "ls", "spec", "usfsm", "udfs", "ndfs", "cnafs", "fmiufs"]
    make_dim_ratio_comparison_plots(mode='classification', cat_pp_mode='factorize', methods=methods)
    make_dim_ratio_comparison_plots(mode='clustering', cat_pp_mode='factorize', methods=methods)
    #

    methods = ["qmrfs", "svd_entropy", "ls", "spec", "usfsm", "udfs", "ndfs", "cnafs", "fmiufs"]
    resproc.make_comparison_table(datasets=datasets, methods=methods)

    # plot_tol_sens_sorting('classification', cat_pp_mode='factorize', translate_mode='const-vector',
    #                       sorting_strategies={'entropy_high2low', 'random', 'default'},
    #                       include_datasets=set(datasets), exclude_datasets={})
    # plot_tol_sens_sorting('clustering', cat_pp_mode='factorize', translate_mode='const-vector',
    #                       sorting_strategies={'entropy_high2low', 'random', 'default'},
    #                       include_datasets=set(datasets), exclude_datasets={})
