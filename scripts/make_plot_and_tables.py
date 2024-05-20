import os
from typing import Literal, Set, Collection, List
import pandas as pd
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
    data = pd.read_json(f"./results/data/main_experiment/{cat_pp_mode}/{mode}.json")
    baseline_results = data.loc[data["method"] == 'baseline_full'].copy()
    baseline_results_copy = baseline_results.copy()
    baseline_results_copy['dim_ratio'] = 0.2
    baseline_results['dim_ratio'] = 0.92
    baseline_results = pd.concat((baseline_results_copy, baseline_results), axis=0, ignore_index=True)
    data = data.loc[data["method"] != 'baseline_full'].copy()
    data = data.loc[(data['dim_ratio'] < 0.99) & (data['dim_ratio'] > 0.18)].copy()
    data['method'] = data['method'].map(lambda x: x[0] if isinstance(x, list) else x)

    y = "accuracy" if mode == "classification" else "nmi"
    y_label = "Accuracy" if mode == "classification" else "NMI"

    datasets = data["dataset"].unique()
    for dataset_name in datasets:
        save_folder = f"results/figures/comparison/{cat_pp_mode}/{dataset_name}/"
        plt_data = data.loc[data["dataset"] == dataset_name]
        bl_plt_data = baseline_results.loc[baseline_results["dataset"] == dataset_name]

        fig, ax = resproc.lineplot(
            plt_data, x="dim_ratio", x_label=r"Dim. ratio", y=y, y_label=y_label,
            hue="method", errorbar="sd", legend_order=methods,
            save_path=os.path.join(save_folder, mode), seperate_legend=True
        )
        sns.lineplot(data=bl_plt_data, x="dim_ratio", y=y, hue="method", errorbar='sd', markers=False, ax=ax,
                     palette=['k'])
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
        exclude_datasets: Collection[str] = None
):
    if exclude_datasets is None:
        exclude_datasets = set()
    save_folder = f"results/figures/sorting_strat/{cat_pp_mode}/{translate_mode}/{mode}"
    os.makedirs(save_folder, exist_ok=True)
    df = load_tol_sens_data(mode, cat_pp_mode=cat_pp_mode, translate_mode=translate_mode)
    y_label = "Acc. ratio" if mode == 'classification' else "NMI ratio"

    df["sorting_model"] = df["sorting_strategy"]
    df.loc[df["sorting_strategy"].str.contains('random'), "sorting_model"] += df["feature_order_seed"].astype(str)

    df = df.loc[~df["dataset"].isin(exclude_datasets)].copy()
    df = df.loc[df["sorting_strategy"].isin(sorting_strategies)].copy()

    resproc.lineplot(
        df, x="tolerance", x_label=r"$\theta$", y="rel_score", y_label=y_label,
        x_scale="log", y_lim=(0.0, 1.5), hue="sorting_model", errorbar='se',
        save_path=os.path.join(save_folder, "sorting_theta")
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
    datasets = ["automobile", "breast_cancer", "heart-c", "heart-statlog",
                # "hepatitis",
                # "ionosphere",
                "lymphography", "sonar", "wdbc", "wine", "zoo", "isolet"]

    methods = ["qmrfs", "svd_entropy", "ls", "spec", "usfsm", "udfs", "ndfs", "cnafs", "fmiufs"]

    make_dim_ratio_comparison_plots(mode='classification', cat_pp_mode='factorize', methods=methods)
    make_dim_ratio_comparison_plots(mode='clustering', cat_pp_mode='factorize', methods=methods)
    # make_dim_ratio_comparison_plots(mode='classification', cat_pp_mode='dummy')
    # make_dim_ratio_comparison_plots(mode='clustering', cat_pp_mode='dummy')

    # out = resproc.create_comparison_table_data(datasets=datasets, methods=methods)

    # resproc.make_comparison_table("classification")
    # resproc.make_comparison_table("clustering")
    # resproc.make_full_res_table()

    # plot_tolerance_sensitivity()

    # for translate_mode in ['none', 'centre', 'non-negative']:
    #     for cat_pp_mode in ['factorize', 'dummy']:
    #         plot_tol_sens_sorting('classification', cat_pp_mode=cat_pp_mode, translate_mode=translate_mode,
    #                               sorting_strategies={'entropy_high2low', 'random', 'default'})
    #         plot_tol_sens_sorting('clustering', cat_pp_mode=cat_pp_mode, translate_mode=translate_mode,
    #                               sorting_strategies={'entropy_high2low', 'random', 'default'})
    #
    #         # plot_tol_sens_single_sorting_model('clustering', sorting_model='entropy_low2high', exclude_datasets={'sonar'})
    #
    #         plot_tol_sens_single_sorting_model(
    #             'classification', sorting_model='entropy_high2low',
    #             cat_pp_mode=cat_pp_mode, translate_mode=translate_mode,
    #             exclude_datasets=None
    #         )
    #
    #         plot_tol_sens_single_sorting_model(
    #             'clustering', sorting_model='entropy_high2low',
    #             cat_pp_mode=cat_pp_mode, translate_mode=translate_mode,
    #             exclude_datasets=None
    #         )

    #
    # plot_tol_sens_single_sorting_model(
    #     'clustering', sorting_model='entropy_high2low',
    #     exclude_datasets={'sonar'}
    # )

    # plot_tol_sens_single_sorting_model('clustering', sorting_model='default', exclude_datasets={'sonar'})
