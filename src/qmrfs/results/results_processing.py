from typing import Dict, List, Collection
import os
import pandas as pd
import numpy as np
import warnings
import copy
from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns
import pylab
from cycler import cycler

colors_cb_github = {'qual': ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c',
                             '#dede00']}

from qmrfs.experiments.utils import DATASET_INFO

PRETTY_METHOD_NAMES = {
    'baseline_full': "Baseline",
    "qmrfs": "QMR-FS",
    "svd_entropy": "SVD Ent.",
    "ls": "LS",
    "spec": "SPEC",
    "usfsm": "USFSM",
    "udfs": "UDFS",
    "ndfs": "NDFS",
    "cnafs": "CNAFS",
    "fmiufs": "FMIUFS",
}


def setup_matplotlib(fontsize=32, fontsize_legend=26):
    rc_extra = {
        "font.size": fontsize,
        'legend.fontsize': fontsize_legend,
        'figure.figsize': (12, 9),
        'legend.frameon': True,
        'legend.edgecolor': '1',
        'legend.facecolor': 'inherit',
        'legend.framealpha': 0.6,
        'legend.markerscale': 1.5,
        # 'text.latex.preview': True,
        'text.usetex': True,
        'svg.fonttype': 'none',
        'text.latex.preamble': r'\usepackage{libertine}',
        'font.family': 'Linux Libertine',
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'libertine',
        'mathtext.it': 'libertine:italic',
        'mathtext.bf': 'libertine:bold',
        'axes.prop_cycle': cycler('color', colors_cb_github['qual']),
        'patch.facecolor': '#0072B2',
        'figure.autolayout': True,
        'lines.linewidth': 3,
    }

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(rc_extra)


def save_all_formats(figure: plt.Figure, save_path: str):
    main_dir = os.path.dirname(save_path)
    name = os.path.basename(save_path)
    save_dir_png = os.path.join(main_dir, "png")
    save_dir_pdf = os.path.join(main_dir, "pdf")
    os.makedirs(save_dir_png, exist_ok=True)
    os.makedirs(save_dir_pdf, exist_ok=True)

    figure.savefig(os.path.join(save_dir_png, f"{name}.png"))
    figure.savefig(os.path.join(save_dir_pdf, f"{name}.pdf"))


def make_legend_pretty(legend_obj, pretty_names: Dict[str, str]):
    legend_obj.set_title(None)
    for legend_text in legend_obj.texts:
        text = legend_text.get_text()
        legend_text.set_text(pretty_names.get(text, text))


def handle_legend(ax: plt.Axes, seperate_legend: bool, legend_order: List[str], figsize=(36, 2), ncols=10):
    pretty_names = PRETTY_METHOD_NAMES

    legend_order = {label: i for i, label in enumerate(legend_order)}
    if not seperate_legend:
        make_legend_pretty(ax.legend(ncol=1), pretty_names=pretty_names)
        figlegend = None
    else:
        ax.legend().set_visible(False)
        figlegend = pylab.figure(figsize=figsize)
        handles, labels = ax.get_legend_handles_labels()
        handles_and_labels = [(handle, label) for handle, label in zip(handles, labels)]
        handles_and_labels = sorted(handles_and_labels, key=lambda x: legend_order[x[1]])
        sorted_handles, sorted_labels = zip(*handles_and_labels)
        # ncol = len(lgd_handles[1]) // 2
        labels = [pretty_names[name] for name in sorted_labels]
        figlegend.legend(sorted_handles, labels, loc='center', ncol=ncols)
    return figlegend


def lineplot(pltdata, *, x: str, y: str,
             x_label: str = None, y_label: str = None,
             x_lim: tuple[float, float] = None, y_lim: tuple[float, float] = None,
             x_scale: str = None, y_scale: str = None, hue=None, errorbar,
             save_path: str = None, fontsize: int = 32, fontsize_legend: int = 26,
             seperate_legend: bool = False, legend_order: List[str], ax=None, baseline_method: str = None):
    setup_matplotlib(fontsize=fontsize, fontsize_legend=fontsize_legend)
    if x_label is None:
        x_label = x
    if y_label is None:
        y_label = y

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    plt.tight_layout(pad=0.2)

    if baseline_method is not None:
        baseline_results = pltdata.loc[pltdata["method"] == baseline_method].copy()
        pltdata = pltdata.loc[pltdata["method"] != baseline_method].copy()
        baseline_results_copy = baseline_results.copy()
        baseline_results_copy[x] = x_lim[0] if x_lim is not None else pltdata[x].min()
        baseline_results[x] = x_lim[1] if x_lim is not None else pltdata[x].max()
        baseline_results = pd.concat((baseline_results_copy, baseline_results), axis=0, ignore_index=True)
        sns.lineplot(data=baseline_results, x=x, y=y, hue=hue, errorbar=errorbar, markers=False, ax=ax, palette=['k'])
        plt_order = copy.deepcopy(legend_order)
        plt_order.remove(baseline_method)
    else:
        plt_order = legend_order

    sns.lineplot(data=pltdata, x=x, y=y, hue=hue, style=hue, errorbar=errorbar, markers=True, ax=ax,
                 hue_order=plt_order, style_order=plt_order, palette='colorblind', markersize=22)
    ax.set_xlabel(x_label, fontdict={'fontsize': int(1.1 * fontsize)})
    ax.set_ylabel(y_label, fontdict={'fontsize': int(1.1 * fontsize)})
    if x_scale is not None:
        ax.set_xscale('log', base=2) if x_scale == "log2" else ax.set_xscale(x_scale)
    if y_scale is not None:
        ax.set_yscale('log', base=2) if y_scale == "log2" else ax.set_yscale(y_scale)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    plt.tight_layout(pad=0.2)

    lgd_obj = ax.legend()
    lgd_obj.set_title(None)

    if save_path is not None:
        save_all_formats(fig, save_path)
    figlegend = handle_legend(ax, seperate_legend=seperate_legend, legend_order=legend_order)
    if save_path is not None:
        save_all_formats(fig, save_path)
        if figlegend is not None:
            save_all_formats(figlegend, save_path + "_legend")

    return fig, ax


def scatterplot(pltdata, x: str, y: str,
                x_label: str = None, y_label: str = None,
                x_lim: tuple[float, float] = None, y_lim: tuple[float, float] = None,
                x_scale: str = None, y_scale: str = None, hue=None,
                save_path: str = None, fontsize: int = 32, fontsize_legend: int = 26):
    setup_matplotlib(fontsize=fontsize, fontsize_legend=fontsize_legend)
    if x_label is None:
        x_label = x
    if y_label is None:
        y_label = y

    fig, ax = plt.subplots()
    sns.scatterplot(data=pltdata, x=x, y=y, hue=hue, style=hue)
    ax.set_xlabel(x_label, fontdict={'fontsize': int(1.1 * fontsize)})
    ax.set_ylabel(y_label, fontdict={'fontsize': int(1.1 * fontsize)})
    if x_scale is not None:
        ax.set_xscale('log', base=2) if x_scale == "log2" else ax.set_xscale(x_scale)
    if y_scale is not None:
        ax.set_yscale('log', base=2) if y_scale == "log2" else ax.set_yscale(y_scale)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    lgd_obj = ax.legend()
    lgd_obj.set_title(None)

    if save_path is not None:
        save_all_formats(fig, save_path)

    return fig, ax


def create_multicol_table_start_nc(table_data):
    top_titles = [method_name for method_name in table_data.columns]

    layout = r"{@{}l|" + len(top_titles) * "p{0.885cm}" + "@{}}"
    out = "\\begin{{tabular}}{layout}".format(layout=layout)

    titles = " &".join(top_titles)

    out += "& " + titles + r"\\" + "\n" + r"\midrule" + "\n"
    return out


def get_thresholds_top_k(df, k_val: int):
    thresholds = {}
    for col in df.columns:
        best_val = df[col].max()
        if col in {"Avg. rank", "Avg. rank std."}:
            topk_method = df[col].sort_values(ascending=False).index[-k_val]
        else:
            topk_method = df[col].sort_values().index[-k_val]
        best_k_val = df[col][topk_method]

        thresholds[col] = best_k_val

    return thresholds


def tbl_elm(value, std, is_best, num_decimal=3):
    element = f"{np.round(value, decimals=num_decimal):.{num_decimal}f} \\pm {np.round(std, decimals=num_decimal):.{num_decimal}f}"
    element = "$\\mathbf{}$".format("{" + element + "}") if is_best else "${}$".format(element)
    return element


def tbl_elm_without_std(value, num_decimal=3, is_best=False):
    element = f"{np.round(value, decimals=num_decimal):.{num_decimal}f}"
    element = "$\\mathbf{}$".format("{" + element + "}") if is_best else "${}$".format(element)
    return element


def percentage_formatting(value: float, num_digits: int):
    rounded = np.round(100 * value, decimals=num_digits)
    if rounded < 10 ** (-num_digits):
        out = "0.0"
    elif rounded >= 10 ** (num_digits - 1):
        out = f"{int(rounded)}"
    else:
        out = f"{rounded:.1f}"
    return out


def crate_full_res_tbl_elm(mean_value, std_value, num_digits: int):
    out = r"$"
    out += percentage_formatting(mean_value, num_digits=num_digits)
    out += r" \pm "
    out += percentage_formatting(std_value, num_digits=num_digits)
    out += r"$"
    return out


def create_full_res_tex_code(cls_results: pd.DataFrame, clstr_results: pd.DataFrame):
    out = r"""\begin{tabular}{@{}l|c|p{0.80cm}p{0.80cm}p{0.80cm}|p{0.80cm}p{0.80cm}p{0.80cm}@{}}
        \toprule
        \multicolumn{1}{c}{}  & \multicolumn{1}{c}{}  & \multicolumn{3}{c}{Classification Acc.} & \multicolumn{3}{c}{Clustering NMI}\\
        \midrule
        Dataset & \% F.S. & QMR-FS & Base-line & Base-line\cite{solorio2020review} & QMR-FS & Base-line & Base-line\cite{solorio2020review} \\
        \midrule
    """
    datasets = cls_results.index

    for dataset in datasets:
        cls_df_row = cls_results.loc[dataset]
        clstr_df_row = clstr_results.loc[dataset]
        the_row = [f"{DATASET_INFO[dataset].pretty_name}"]

        percentage_kept_dims = percentage_formatting(cls_df_row['dim_ratio'], num_digits=2)
        the_row.append(f"{percentage_kept_dims}")

        the_row.append(crate_full_res_tbl_elm(cls_df_row['red_mean'], cls_df_row['red_std'], num_digits=2))
        the_row.append(crate_full_res_tbl_elm(cls_df_row['full_mean'], cls_df_row['full_std'], num_digits=2))
        the_row.append(percentage_formatting(DATASET_INFO[dataset].acc_ref, num_digits=3))

        the_row.append(crate_full_res_tbl_elm(clstr_df_row['red_mean'], clstr_df_row['red_std'], num_digits=2))
        the_row.append(crate_full_res_tbl_elm(clstr_df_row['full_mean'], clstr_df_row['full_std'], num_digits=2))
        the_row.append(percentage_formatting(DATASET_INFO[dataset].nmi_ref, num_digits=3))

        the_row = " & ".join(the_row)
        the_row += r"\\" + "\n"
        out += the_row
    out += "\\bottomrule\n\\end{tabular}"
    return out


def create_comparison_table_tex_code(table_data):
    out = create_multicol_table_start_nc(table_data)
    # out += create_supervised_row(table_data, pretty_names_obj)
    # out += r"\midrule"

    thresholds = get_thresholds_top_k(table_data.T, 1)

    for dataset in table_data.index:
        if dataset == "Avg. rank":
            out += r"\midrule" + "\n"
        the_full_row = [f"{DATASET_INFO[dataset].pretty_name if dataset in DATASET_INFO else dataset} "]
        for col in table_data.columns:
            val = table_data.loc[dataset, col].item()
            if dataset == "Avg. rank":
                # std = table_data.loc["rank_std", col].item()
                the_full_row.append(tbl_elm_without_std(val, is_best=val <= thresholds[dataset]))
            elif dataset == "Avg. rank std.":
                the_full_row.append(tbl_elm_without_std(val, is_best=False))
            else:
                the_full_row.append(tbl_elm_without_std(val, is_best=val >= thresholds[dataset]))
        out += " & ".join(the_full_row)
        out += r"\\" + "\n"

    # if c < len(algs_dict) - 1:
    out += "\\bottomrule\n\\end{tabular}"
    return out


def create_filter_mask(data: pd.DataFrame, dataset: str, tol: float, p: float, dim_tol: int, dim: int):
    p_min = (1. - tol) * p
    p_max = (1. + tol) * p
    dim_min = dim - dim_tol
    dim_max = dim + dim_tol
    if dataset == 'isolet':
        mask = (data["red_dim"] > dim_min) & (data["red_dim"] < dim_max) & (data["dataset"] == 'isolet')
    else:
        mask = (data["dim_ratio"] > p_min) & (data["dim_ratio"] < p_max) & (data["dataset"] == dataset)
    return mask


def filter_data_by_dim_ratio(data: pd.DataFrame, p: float, dim: int, methods: List[str]):
    datasets = data['dataset'].unique()
    methods = set(methods)
    tol = {d: 0.01 for d in datasets}
    dim_tol = 1

    finished_data = dict()

    for dataset in datasets:

        mask = create_filter_mask(data, dataset=dataset, tol=tol[dataset], p=p, dim_tol=dim_tol, dim=dim)
        filtered_data = data.loc[mask]

        count = 0
        while set(filtered_data["method"].unique()) != methods:
            if count >= 10:
                warnings.warn(
                    f"Method difference {methods - set(filtered_data['method'].unique())} for tolerance of {p} for dataset {dataset}")
                # raise RuntimeError(f"Could not find dim ratio with in tolerance of {p} for dataset {dataset}")

            if dataset == "isolet":
                dim_tol += 1
            else:
                tol[dataset] += 0.02
            mask = create_filter_mask(data, dataset=dataset, tol=tol[dataset], p=p, dim_tol=dim_tol, dim=dim)
            filtered_data = data.loc[mask]
            count += 1

        finished_data[dataset] = filtered_data

    finished_data = pd.concat(finished_data.values(), ignore_index=True)
    return finished_data


def _get_avg_rank(data: pd.DataFrame, p: float, dim: int, methods: List[str], score_name: str):
    filtered_data = filter_data_by_dim_ratio(data, p=p, dim=dim, methods=methods)
    avg_data = filtered_data.groupby(["dataset", "method"], as_index=False).agg("mean")
    avg_data['rank'] = avg_data.groupby("dataset")[score_name].rank(method='dense', ascending=False)
    avg_rank = avg_data.groupby("method", as_index=True)['rank'].agg(['mean', 'std'])
    return avg_rank


def create_comparison_table_data(datasets: Collection[str], methods: List[str], dim_ratios: List[tuple[float, int]]):
    cls_data = pd.read_json("results/data/main_experiment/factorize/classification.json", orient='records')
    cls_data = cls_data.loc[(cls_data['dataset'].isin(set(datasets))) & (cls_data['method'].isin(set(methods)))].copy()

    clstr_data = pd.read_json("results/data/main_experiment/factorize/clustering.json", orient='records')
    isolet_durations = clstr_data.loc[
        clstr_data["dataset"] == "isolet"
        ].groupby(["method"], as_index=True)["duration"].agg("mean")
    clstr_data = clstr_data.loc[
        (clstr_data['dataset'].isin(set(datasets))) & (clstr_data['method'].isin(set(methods)))].copy()

    cls_avg_rank_data = {}
    clstr_avg_rank_data = {}
    for p, isolet_dim in dim_ratios:
        cls_avg_rank_data[int(100 * p)] = _get_avg_rank(cls_data, p=p, dim=isolet_dim,
                                                        methods=methods, score_name='accuracy')
        clstr_avg_rank_data[int(100 * p)] = _get_avg_rank(clstr_data, p=p, dim=isolet_dim,
                                                          methods=methods, score_name='nmi')

    return isolet_durations, cls_avg_rank_data, clstr_avg_rank_data


def get_start_method_comparison(methods):
    top_titles = [PRETTY_METHOD_NAMES[method_name] for method_name in methods]
    layout = r"{@{}l|" + len(top_titles) * "c" + "@{}}"
    out = "\\begin{{tabular}}{layout}".format(layout=layout) + "\n"
    titles = " &".join(top_titles)
    out += "& " + titles + r"\\" + "\n" + r"\midrule" + "\n"
    return out


def method_complexities(methods):
    # complexities = {
    #     "qmrfs": r"$\bigO(n d^2)$",
    #     "svd_entropy": r"$\bigO(n d^3)$",
    #     "ls": r"$\bigO(n^2 d)$",
    #     "spec": r"$\bigO(n^2 d)$",
    #     "usfsm": r"$\bigO(n^3 d + n^2 d^2)$",
    #     "udfs": r"$\bigO(n^2 d + d^3)^*$",
    #     "ndfs": r"$\bigO(n^2 d + n d^2 + d^3)^*$",
    #     "cnafs": r"$\bigO(n^2 d + n d^2 + d^3)^*$",
    #     "fmiufs": r"$\bigO(n^2 d + n d^2)$"
    # }

    complexities = {
        "qmrfs": r"$\bigO(n d^2)$",
        "svd_entropy": r"$\bigO(n d^3)$",
        "ls": r"$\bigO(n^2 d)$",
        "spec": r"$\bigO(n^2 d)$",
        "usfsm": r"$\bigO(n^3 d)$",
        "udfs": r"$\bigO(n^2 d)^*$",
        "ndfs": r"$\bigO(n^2 d)^*$",
        "cnafs": r"$\bigO(n^2 d)^*$",
        "fmiufs": r"$\bigO(n^2 d)$"
    }

    out = r"Time complexity &" + " &".join([complexities[method] for method in methods]) + r"\\" + "\n"
    return out


def timeformat(duration: float):
    td_duration = timedelta(seconds=duration)
    hours, remainder = divmod(int(td_duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    if duration < 1:
        out = f"{td_duration.microseconds // 1000}ms"
    elif duration < 60:
        out = f"{seconds}s"
    elif duration < 3600:
        out = f"{minutes}m {seconds}s"
    elif hours < 5:
        out = f"{hours}h {minutes}m"
    else:
        out = f"{hours}h"
    return out


def get_isolet_times(isolet_durations: pd.Series, methods: List[str]):
    out = "Runtime (Isolet)"
    qmrfs_time = isolet_durations['qmrfs'].item()
    for method in methods:
        if method == "qmrfs":
            out += f" & {timeformat(qmrfs_time)}"
        else:
            ratio = int(np.round(isolet_durations[method].item() / qmrfs_time))
            out += f" & " + r"$\times" + f"{ratio}$"
    out += r"\\" + "\n"
    return out


def format_rank_data(rank_data: pd.DataFrame, methods: List[str], row_name: str):
    out = row_name
    for method in methods:
        elm = tbl_elm(value=rank_data.loc[method, 'mean'], std=rank_data.loc[method, 'std'], num_decimal=1,
                      is_best=False)
        out += f" & {elm}"
    out += r"\\" + "\n"
    return out


def make_comparison_table(datasets: Collection[str], methods: List[str]):
    main_save_dir = "results/tables/main_res"
    dim_ratios = [
        (0.4, 50),
        # (0.7, 70),
        (0.6, 100),
        # (0.9, 100)
    ]
    isolet_durations, cls_rank_data, clstr_rank_data = create_comparison_table_data(
        datasets=datasets, methods=methods, dim_ratios=dim_ratios)

    out = get_start_method_comparison(methods)
    out += method_complexities(methods)
    out += get_isolet_times(isolet_durations, methods=methods)
    out += r"\midrule" + "\n"
    out += format_rank_data(cls_rank_data[40], methods=methods, row_name=r"Clsif. avg. rank ($40$\%)")
    out += format_rank_data(cls_rank_data[60], methods=methods, row_name=r"Clsif. avg. rank ($60$\%)")
    out += r"\midrule" + "\n"
    out += format_rank_data(clstr_rank_data[40], methods=methods, row_name=r"Clstr. avg. rank ($40$\%)")
    out += format_rank_data(clstr_rank_data[60], methods=methods, row_name=r"Clstr. avg. rank ($60$\%)")

    out += r"\bottomrule" + "\n" r"\end{tabular}"

    os.makedirs(main_save_dir, exist_ok=True)
    save_filename = os.path.join(main_save_dir, f"main_res.tex")
    print(f"Saving to {save_filename}")
    with open(save_filename, "w") as fp:
        fp.write(out)


def make_full_res_table(results_folder: str = "entropy_high2low_1-00e-01"):
    main_save_dir = "results/tables/full_res"

    cls_results = pd.read_json(f"results/data/main_experiment/classification/{results_folder}/full_data.json",
                               orient='records').set_index("dataset")
    clstr_results = pd.read_json(f"results/data/main_experiment/clustering/{results_folder}/full_data.json",
                                 orient='records').set_index("dataset")

    tex_code = create_full_res_tex_code(cls_results=cls_results, clstr_results=clstr_results)

    os.makedirs(main_save_dir, exist_ok=True)
    save_filename = os.path.join(main_save_dir, f"full_res.tex")
    print(f"Saving to {save_filename}")
    with open(save_filename, "w") as fp:
        fp.write(tex_code)


def make_comparison_table_old(mode):
    main_save_dir = "results/tables/rel_and_rank"
    qmr_res = pd.read_json(f"results/data/main_experiment/{mode}/rel_and_rank.json")
    qmr_res = qmr_res.reindex(index=['qmr'] + qmr_res.index[:-1].tolist())

    qmr_res = qmr_res.rename(
        index={
            "qmr": "QMR",
            "SVD-entropy": "SVD-ent.",
            "LS-WNCH-BE": "L.W.B.",
            "UFSACO": "UFS-ACO",
            "MGSACO": "MGS-ACO",
            "DSRMR": "DSR-MR",
            "Li et al.": "Li et.al"
        },
        columns={
            "avg_rank": "Avg. rank",
            "rank_std": "Avg. rank std."
        })

    tex_code = create_comparison_table_tex_code(qmr_res.T)

    os.makedirs(main_save_dir, exist_ok=True)
    save_filename = os.path.join(main_save_dir, f"{mode}.tex")
    print(f"Saving to {save_filename}")
    with open(save_filename, "w") as fp:
        fp.write(tex_code)
