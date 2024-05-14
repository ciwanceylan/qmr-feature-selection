import os
import pandas as pd
import qmrfs.results.results_processing as resproc


def plot_tolerance_sensitivity():
    save_folder = "results/figures/"
    os.makedirs(save_folder, exist_ok=True)
    df_clf = pd.read_json(f"results/data/tol_sens/classification.json")
    df_clf["mode"] = "Classification (Score=Accuracy)"
    df_clstr = pd.read_json(f"results/data/tol_sens/clustering.json")
    df_clstr["mode"] = "Clustering (Score=NMI)"
    df = pd.concat((df_clf, df_clstr), axis=0, ignore_index=True)

    df["percent_dims_removed"] = 100 * (1. - df["dim_ratio"])

    y_label = "Score ratio"

    resproc.lineplot(
        df, x="tolerance", x_label=r"$\theta$", y="rel_score", y_label=y_label,
        x_scale="log", y_lim=(0.5, 1.2),  hue="mode",
        save_path=os.path.join(save_folder, "tolerance_sensitivity")
    )

    resproc.lineplot(
        df, x="tolerance", x_label=r"$\theta$", y="percent_dims_removed", y_label=r"\% features removed",
        x_scale="log", y_lim=(0., 100.), hue=None,
        save_path=os.path.join(save_folder, "dimensions")
    )


if __name__ == "__main__":
    # resproc.make_comparison_table("classification")
    # resproc.make_comparison_table("clustering")

    plot_tolerance_sensitivity()
