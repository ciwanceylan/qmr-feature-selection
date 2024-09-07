import os
from typing import Literal, Collection, List
import pandas as pd
import numpy as np
import qmrfs.results.results_processing as resproc
from qmrfs.qmr_feature_selection import SortingStrategy
import glob
import matplotlib.pyplot as plt


def load_tol_sens_data(mode: Literal['classification', 'clustering'], cat_pp_mode: Literal['factorize', 'dummy']):
    results = []
    for path in glob.glob(f"./results/data/tol_sens/{cat_pp_mode}/*/", recursive=False):
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
    datafilter = (((data["dataset"] == 'isolet') & (data['red_dim'] >= 19)) | ((data["dataset"] != 'isolet') & (data['dim_ratio'] >= 0.18)))
    data = data.loc[datafilter].copy()
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
                hue="method", errorbar="se", legend_order=methods, fontsize_legend=34, fontsize=48,
                save_path=os.path.join(save_folder, mode), seperate_legend=True, baseline_method='baseline_full'
            )
            plt.close(fig)
        else:
            fig, ax = resproc.lineplot(
                plt_data, x="percent_dim_kept", x_label=r"\% features kept", y=y, y_label=y_label, x_lim=(18, 100),
                y_lim=y_lims[mode].get(dataset_name, None), hue="method", errorbar="se", legend_order=methods,
                fontsize_legend=34, fontsize=48,
                save_path=os.path.join(save_folder, mode), seperate_legend=True, baseline_method='baseline_full'
            )
            plt.close(fig)


def plot_tol_sens_sorting(
        mode: Literal['classification', 'clustering'],
        cat_pp_mode: Literal['factorize', 'dummy'],
        sorting_strategies: Collection[SortingStrategy],
        include_datasets: Collection[str],
        exclude_datasets: Collection[str]
):
    if exclude_datasets is None:
        exclude_datasets = set()
    save_folder = f"results/figures/sorting_strat/{cat_pp_mode}/{mode}"
    os.makedirs(save_folder, exist_ok=True)
    df = load_tol_sens_data(mode, cat_pp_mode=cat_pp_mode)
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


if __name__ == "__main__":
    datasets = ["automobile",
                "heart-c",
                "heart-statlog",
                "sonar",
                "wine",
                "isolet"]

    methods = ["baseline_full", "qrpfs", "qmrfs", "svd_entropy", "ls", "spec", "usfsm", "udfs", "ndfs", "cnafs", "fmiufs"]
    make_dim_ratio_comparison_plots(mode='classification', cat_pp_mode='factorize', methods=methods)
    make_dim_ratio_comparison_plots(mode='clustering', cat_pp_mode='factorize', methods=methods)

    methods = ["qrpfs", "qmrfs", "svd_entropy", "ls", "spec", "usfsm", "udfs", "ndfs", "cnafs", "fmiufs"]
    resproc.make_comparison_table(datasets=datasets, methods=methods)

    # plot_tol_sens_sorting('classification', cat_pp_mode='factorize', translate_mode='const-vector',
    #                       sorting_strategies={'entropy_high2low', 'random', 'default'},
    #                       include_datasets=set(datasets), exclude_datasets={})
    # plot_tol_sens_sorting('clustering', cat_pp_mode='factorize', translate_mode='const-vector',
    #                       sorting_strategies={'entropy_high2low', 'random', 'default'},
    #                       include_datasets=set(datasets), exclude_datasets={})
