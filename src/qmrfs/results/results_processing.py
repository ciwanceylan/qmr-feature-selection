import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

colors_cb_github = {'qual': ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c',
                             '#dede00']}

from qmrfs.experiments.utils import DATASET_INFO


def setup_matplotlib(fontsize=32, fontsize_legend=26):
    rc_extra = {
        "font.size": fontsize,
        'legend.fontsize': fontsize_legend,
        'figure.figsize': (12, 9),
        'legend.frameon': True,
        'legend.edgecolor': '1',
        'legend.facecolor': 'inherit',
        'legend.framealpha': 0.6,
        'legend.markerscale': 2.3,
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


def lineplot(pltdata, x: str, y: str,
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
    sns.lineplot(data=pltdata, x=x, y=y, hue=hue, style=hue, errorbar=("ci", 95))
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


def make_nc_multicol_table(table_data):
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


def make_comparison_table(mode):
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

    tex_code = make_nc_multicol_table(qmr_res.T)

    os.makedirs(main_save_dir, exist_ok=True)
    save_filename = os.path.join(main_save_dir, f"{mode}.tex")
    print(f"Saving to {save_filename}")
    with open(save_filename, "w") as fp:
        fp.write(tex_code)


def make_qmr_results_table(df: pd.DataFrame):
    pass
