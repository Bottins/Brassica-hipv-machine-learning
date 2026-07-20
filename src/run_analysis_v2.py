"""Robustness-focused analysis for the receiver-plant study.

This version leaves the original analysis untouched and adds:

* repeated stratified five-fold cross-validation;
* permutation-derived null distributions;
* leave-one-experimental-run-out sensitivity analysis;
* out-of-fold SHAP feature importance;
* uncertainty intervals for feature-set ablation.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import run_analysis as original


SEED = 42
N_REPEATS = 30
N_ABLATION_REPEATS = 5
N_PERMUTATIONS = 30
PERMUTATION_CV_REPEATS = 1
N_SHAP_REPEATS = 3
N_JOBS = 4

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "processed" / "receiver_plant_dataset.xlsx"
OUT_DIR = PROJECT_DIR / "results_v2"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

ID_COLS = original.ID_COLS
CLASS_ORDER = original.CLASS_ORDER
FOCUS_CLASSES = original.FOCUS_CLASSES
GB_PARAMS_4CLASS = original.GB_PARAMS_4CLASS
GB_PARAMS_FOCUSED = original.GB_PARAMS_FOCUSED


def evaluate_splits(X, y, class_order, params, splits, positive_label=None):
    """Return out-of-fold metrics and predictions for an explicit split list."""
    predictions = np.empty(len(y), dtype=object)
    probabilities = np.zeros((len(y), len(class_order)), dtype=float)
    seen = np.zeros(len(y), dtype=bool)
    feature_importances = []

    for train_idx, validation_idx in splits:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(X[train_idx])
        x_validation = scaler.transform(X[validation_idx])
        model = GradientBoostingClassifier(**params)
        model.fit(x_train, y[train_idx])

        fold_predictions = model.predict(x_validation)
        fold_probabilities = model.predict_proba(x_validation)
        class_to_index = {label: idx for idx, label in enumerate(model.classes_)}
        ordered_probabilities = np.column_stack(
            [fold_probabilities[:, class_to_index[label]] for label in class_order]
        )

        predictions[validation_idx] = fold_predictions
        probabilities[validation_idx] = ordered_probabilities
        seen[validation_idx] = True
        feature_importances.append(model.feature_importances_)

    y_seen = y[seen]
    pred_seen = predictions[seen]
    result = {
        "preds": predictions,
        "probs": probabilities,
        "seen": seen,
        "cm": confusion_matrix(y_seen, pred_seen, labels=class_order),
        "accuracy": accuracy_score(y_seen, pred_seen),
        "f1": (
            f1_score(y_seen, pred_seen, pos_label=positive_label)
            if positive_label
            else f1_score(y_seen, pred_seen, average="macro")
        ),
        "feature_importance_mean": np.mean(feature_importances, axis=0),
    }
    if positive_label:
        positive_index = class_order.index(positive_label)
        binary_y = (y_seen == positive_label).astype(int)
        result["auc"] = roc_auc_score(binary_y, probabilities[seen, positive_index])
    else:
        result["auc"] = np.nan
    return result


def evaluate_seed(X, y, class_order, params, seed, positive_label=None):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(splitter.split(X, y))
    result = evaluate_splits(X, y, class_order, params, splits, positive_label)
    return {
        "repeat": seed,
        "accuracy": result["accuracy"],
        "f1": result["f1"],
        "auc": result["auc"],
        "cm": result["cm"],
        "feature_importance": result["feature_importance_mean"],
    }


def repeated_cv(X, y, class_order, params, positive_label=None, n_repeats=N_REPEATS):
    rows = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_seed)(
            X,
            y,
            class_order,
            params,
            SEED + repeat,
            positive_label,
        )
        for repeat in range(n_repeats)
    )
    metrics = pd.DataFrame(
        [
            {
                "repeat": repeat,
                "accuracy": row["accuracy"],
                "f1": row["f1"],
                "auc": row["auc"],
            }
            for repeat, row in enumerate(rows, start=1)
        ]
    )
    return {
        "metrics": metrics,
        "cms": np.stack([row["cm"] for row in rows]),
        "feature_importance_mean": np.mean(
            np.stack([row["feature_importance"] for row in rows]), axis=0
        ),
    }


def permutation_worker(
    permutation,
    X,
    y,
    class_order,
    params,
    positive_label=None,
):
    rng = np.random.default_rng(SEED + 10_000 + permutation)
    permuted_y = rng.permutation(y)
    repeated_rows = [
        evaluate_seed(
            X,
            permuted_y,
            class_order,
            params,
            SEED + repeat,
            positive_label,
        )
        for repeat in range(PERMUTATION_CV_REPEATS)
    ]
    return {
        "permutation": permutation + 1,
        "accuracy": float(np.median([row["accuracy"] for row in repeated_rows])),
        "f1": float(np.median([row["f1"] for row in repeated_rows])),
        "auc": (
            float(np.median([row["auc"] for row in repeated_rows]))
            if positive_label
            else np.nan
        ),
    }


def permutation_null(X, y, class_order, params, positive_label=None):
    rows = Parallel(n_jobs=N_JOBS)(
        delayed(permutation_worker)(
            permutation,
            X,
            y,
            class_order,
            params,
            positive_label,
        )
        for permutation in range(N_PERMUTATIONS)
    )
    return pd.DataFrame(rows)


def summarize_metric(task, metric, observed, null):
    values = observed[metric].dropna().to_numpy()
    null_values = null[metric].dropna().to_numpy()
    estimate = float(np.median(values))
    permutation_observed = float(values[0])
    return {
        "task": task,
        "metric": metric,
        "estimate": estimate,
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "null_median": float(np.median(null_values)),
        "null_q025": float(np.quantile(null_values, 0.025)),
        "null_q975": float(np.quantile(null_values, 0.975)),
        "permutation_observed": permutation_observed,
        "permutation_p": float(
            (1 + np.sum(null_values >= permutation_observed)) / (len(null_values) + 1)
        ),
    }


def save_performance_with_null(summary, null_4, null_focus):
    labels = ["Accuracy", "Macro-F1", "Accuracy", "F1", "AUC"]
    x = np.arange(len(labels))
    estimates = summary["estimate"].to_numpy()
    lower = estimates - summary["q025"].to_numpy()
    upper = summary["q975"].to_numpy() - estimates
    null_values = [
        null_4["accuracy"].to_numpy(),
        null_4["f1"].to_numpy(),
        null_focus["accuracy"].to_numpy(),
        null_focus["f1"].to_numpy(),
        null_focus["auc"].to_numpy(),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    violin = ax.violinplot(
        null_values,
        positions=x,
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor("#C9CED3")
        body.set_edgecolor("#6C757D")
        body.set_alpha(0.75)
    violin["cmedians"].set_color("#59656F")
    violin["cmedians"].set_linewidth(1.2)

    colors = ["#2C7FB8", "#74A9CF", "#C94C4C", "#E07A5F", "#3B8C5A"]
    for position, estimate, low, high, color in zip(x, estimates, lower, upper, colors):
        ax.errorbar(
            position,
            estimate,
            yerr=np.array([[low], [high]]),
            fmt="o",
            markersize=8,
            capsize=4,
            linewidth=1.5,
            color=color,
            ecolor=color,
            zorder=5,
        )
    for position, estimate in zip(x, estimates):
        ax.text(position, estimate + 0.035, f"{estimate:.2f}", ha="center", fontweight="bold")

    ax.axvline(1.5, color="#AAB2B8", linewidth=1.0)
    ax.text(0.5, 1.01, "Four treatment groups", ha="center", transform=ax.get_xaxis_transform())
    ax.text(3.0, 1.01, "Herbivore-challenged receivers", ha="center", transform=ax.get_xaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Cross-validated score")
    ax.legend(
        handles=[
            mpatches.Patch(color="#C9CED3", label="Permuted-label null distribution"),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="#20262E",
                linestyle="",
                label="Median and empirical 95% interval",
            ),
        ],
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_model_performance_v2.png", bbox_inches="tight")
    plt.close(fig)


def mean_row_normalized_confusion(cms):
    normalized = []
    for cm in cms:
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized.append(
            np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        )
    return np.mean(normalized, axis=0)


def save_repeated_confusion(result_4, result_focus):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    specifications = [
        (
            axes[0],
            mean_row_normalized_confusion(result_4["cms"]),
            CLASS_ORDER,
            "#2C7FB8",
            "Four treatment groups",
        ),
        (
            axes[1],
            mean_row_normalized_confusion(result_focus["cms"]),
            FOCUS_CLASSES,
            "#C94C4C",
            "Herbivore-challenged receivers",
        ),
    ]
    for ax, matrix, classes, color, title in specifications:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1%",
            cmap=sns.light_palette(color, as_cmap=True),
            vmin=0,
            vmax=1,
            xticklabels=[original.TREAT_LABEL[label] for label in classes],
            yticklabels=[original.TREAT_LABEL[label] for label in classes],
            ax=ax,
            cbar=False,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted treatment")
        ax.set_ylabel("Observed treatment")
        ax.tick_params(axis="x", rotation=24)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Mean row-normalised confusion matrices across repeated cross-validation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_confusion_matrices_v2.png", bbox_inches="tight")
    plt.close(fig)


def compute_ablation(df, feature_cols):
    feature_sets = original.get_feature_sets(feature_cols)
    focus_df = df[df["labels"].isin(FOCUS_CLASSES)].copy()
    rows = []
    for set_name, columns in feature_sets.items():
        tasks = [
            (
                "4-class",
                df[columns].to_numpy(dtype=np.float32),
                df["labels"].to_numpy(dtype=object),
                CLASS_ORDER,
                GB_PARAMS_4CLASS,
                None,
            ),
            (
                "focused h0h1 vs h1h1",
                focus_df[columns].to_numpy(dtype=np.float32),
                focus_df["labels"].to_numpy(dtype=object),
                FOCUS_CLASSES,
                GB_PARAMS_FOCUSED,
                "h1h1",
            ),
        ]
        for task, X, y, classes, params, positive_label in tasks:
            result = repeated_cv(
                X,
                y,
                classes,
                params,
                positive_label,
                n_repeats=N_ABLATION_REPEATS,
            )
            metric = "auc" if positive_label else "f1"
            values = result["metrics"][metric].to_numpy()
            rows.append(
                {
                    "task": task,
                    "feature_set": set_name,
                    "n_samples": len(y),
                    "n_features": len(columns),
                    "metric": metric,
                    "estimate": float(np.median(values)),
                    "q025": float(np.quantile(values, 0.025)),
                    "q975": float(np.quantile(values, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def save_ablation(ablation):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    task_specs = [
        ("4-class", "Four treatment groups", "Macro-F1"),
        ("focused h0h1 vs h1h1", "Herbivore-challenged receivers", "AUC"),
    ]
    colors = {
        "all": "#59656F",
        "gas+voc_no_leaf": "#8C6BB1",
        "gas_only": "#2C7FB8",
        "voc_only": "#C94C4C",
        "leaf_only": "#3B8C5A",
    }
    for ax, (task, title, ylabel) in zip(axes, task_specs):
        subset = ablation[ablation["task"] == task].sort_values("estimate", ascending=False)
        positions = np.arange(len(subset))
        estimates = subset["estimate"].to_numpy()
        lower = estimates - subset["q025"].to_numpy()
        upper = subset["q975"].to_numpy() - estimates
        bars = ax.bar(
            positions,
            estimates,
            color=[colors[name] for name in subset["feature_set"]],
            edgecolor="white",
        )
        ax.errorbar(
            positions,
            estimates,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            color="#20262E",
            capsize=3,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [original.FEATURE_SET_LABEL[name] for name in subset["feature_set"]],
            rotation=25,
            ha="right",
        )
        ax.set_ylim(0, 0.95)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for bar, estimate in zip(bars, estimates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                estimate + 0.035,
                f"{estimate:.2f}",
                ha="center",
                fontweight="bold",
            )
    fig.suptitle("Performance by measurement domain")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_feature_set_ablation_v2.png", bbox_inches="tight")
    plt.close(fig)


def compute_oof_shap(df, feature_cols):
    focus_df = df[df["labels"].isin(FOCUS_CLASSES)].copy()
    X = focus_df[feature_cols].to_numpy(dtype=np.float32)
    y = focus_df["labels"].to_numpy(dtype=object)
    absolute_sum = np.zeros(len(feature_cols), dtype=float)
    signed_sum = np.zeros(len(feature_cols), dtype=float)
    n_explanations = 0

    for repeat in range(N_SHAP_REPEATS):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + repeat)
        for train_idx, validation_idx in splitter.split(X, y):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(X[train_idx])
            x_validation = scaler.transform(X[validation_idx])
            model = GradientBoostingClassifier(**GB_PARAMS_FOCUSED)
            model.fit(x_train, y[train_idx])
            values = shap.TreeExplainer(model).shap_values(x_validation)
            if isinstance(values, list):
                values = values[FOCUS_CLASSES.index("h1h1")]
            elif np.ndim(values) == 3:
                values = values[:, :, FOCUS_CLASSES.index("h1h1")]
            absolute_sum += np.abs(values).sum(axis=0)
            signed_sum += values.sum(axis=0)
            n_explanations += len(validation_idx)

    return (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "mean_abs_shap": absolute_sum / n_explanations,
                "mean_shap": signed_sum / n_explanations,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def save_oof_shap(shap_df):
    named = shap_df.copy()
    named["feature_label"] = named["feature"].map(original.clean_name)
    named["domain"] = named["feature"].map(original.feature_domain)
    top = named.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.barh(
        top["feature_label"],
        top["mean_abs_shap"],
        color=[original.DOMAIN_COLORS[domain] for domain in top["domain"]],
        edgecolor="white",
    )
    ax.set_xlabel("Mean absolute out-of-fold SHAP value")
    ax.set_title("Out-of-fold feature contributions in the focused classifier")
    ax.legend(
        handles=[
            mpatches.Patch(color=color, label=domain)
            for domain, color in original.DOMAIN_COLORS.items()
        ],
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_focused_shap_features_v2.png", bbox_inches="tight")
    plt.close(fig)


def leave_one_run_out(X, y, groups, class_order, params, positive_label=None):
    splitter = LeaveOneGroupOut()
    splits = list(splitter.split(X, y, groups))
    return evaluate_splits(X, y, class_order, params, splits, positive_label)


def save_grouped_probabilities(result, y):
    positive_index = FOCUS_CLASSES.index("h1h1")
    probability = result["probs"][:, positive_index]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bins = np.linspace(0, 1, 12)
    for label in FOCUS_CLASSES:
        mask = y == label
        ax.hist(
            probability[mask],
            bins=bins,
            alpha=0.72,
            color=original.TREAT_COLORS[label],
            edgecolor="white",
            label=f"Observed {original.TREAT_LABEL[label]}",
        )
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1.4)
    ax.set_xlabel("Leave-one-run-out probability of rHIPV + herbivore")
    ax.set_ylabel("Number of samples")
    ax.set_title("Predicted probabilities with experimental runs held out")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_probability_separation_v2.png", bbox_inches="tight")
    plt.close(fig)


def main():
    original.configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    original.FIG_DIR = FIG_DIR
    original.TABLE_DIR = TABLE_DIR

    df = pd.read_excel(DATA_FILE)
    feature_cols = [column for column in df.columns if column not in ID_COLS]
    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df["labels"].to_numpy(dtype=object)
    groups_all = df["run"].to_numpy()

    focus_mask = np.isin(y_all, FOCUS_CLASSES)
    X_focus = X_all[focus_mask]
    y_focus = y_all[focus_mask]
    groups_focus = groups_all[focus_mask]

    repeated_4 = repeated_cv(X_all, y_all, CLASS_ORDER, GB_PARAMS_4CLASS)
    repeated_focus = repeated_cv(
        X_focus,
        y_focus,
        FOCUS_CLASSES,
        GB_PARAMS_FOCUSED,
        positive_label="h1h1",
    )
    null_4 = permutation_null(X_all, y_all, CLASS_ORDER, GB_PARAMS_4CLASS)
    null_focus = permutation_null(
        X_focus,
        y_focus,
        FOCUS_CLASSES,
        GB_PARAMS_FOCUSED,
        positive_label="h1h1",
    )

    summary = pd.DataFrame(
        [
            summarize_metric(
                "Four treatment groups",
                "accuracy",
                repeated_4["metrics"],
                null_4,
            ),
            summarize_metric(
                "Four treatment groups",
                "f1",
                repeated_4["metrics"],
                null_4,
            ),
            summarize_metric(
                "Herbivore-challenged receivers",
                "accuracy",
                repeated_focus["metrics"],
                null_focus,
            ),
            summarize_metric(
                "Herbivore-challenged receivers",
                "f1",
                repeated_focus["metrics"],
                null_focus,
            ),
            summarize_metric(
                "Herbivore-challenged receivers",
                "auc",
                repeated_focus["metrics"],
                null_focus,
            ),
        ]
    )

    grouped_4 = leave_one_run_out(
        X_all,
        y_all,
        groups_all,
        CLASS_ORDER,
        GB_PARAMS_4CLASS,
    )
    grouped_focus = leave_one_run_out(
        X_focus,
        y_focus,
        groups_focus,
        FOCUS_CLASSES,
        GB_PARAMS_FOCUSED,
        positive_label="h1h1",
    )
    grouped_summary = pd.DataFrame(
        [
            {
                "task": "Four treatment groups",
                "accuracy": grouped_4["accuracy"],
                "f1": grouped_4["f1"],
                "auc": np.nan,
            },
            {
                "task": "Herbivore-challenged receivers",
                "accuracy": grouped_focus["accuracy"],
                "f1": grouped_focus["f1"],
                "auc": grouped_focus["auc"],
            },
        ]
    )

    ablation = compute_ablation(df, feature_cols)
    shap_df = compute_oof_shap(df, feature_cols)

    original.save_dataset_overview(df, feature_cols)
    save_performance_with_null(summary, null_4, null_focus)
    save_repeated_confusion(repeated_4, repeated_focus)
    save_ablation(ablation)
    save_oof_shap(shap_df)
    save_grouped_probabilities(grouped_focus, y_focus)
    original.save_pca_views(df, feature_cols, X_all, y_all)
    original.save_top_feature_boxplots(df, shap_df)

    summary.to_csv(TABLE_DIR / "robust_performance_summary.csv", index=False)
    repeated_4["metrics"].assign(task="four_class").to_csv(
        TABLE_DIR / "repeated_cv_four_class.csv", index=False
    )
    repeated_focus["metrics"].assign(task="focused_h0h1_vs_h1h1").to_csv(
        TABLE_DIR / "repeated_cv_focused.csv", index=False
    )
    pd.concat(
        [
            null_4.assign(task="four_class"),
            null_focus.assign(task="focused_h0h1_vs_h1h1"),
        ],
        ignore_index=True,
    ).to_csv(TABLE_DIR / "permutation_null_distributions.csv", index=False)
    grouped_summary.to_csv(TABLE_DIR / "leave_one_run_out_summary.csv", index=False)
    ablation.to_csv(TABLE_DIR / "feature_set_ablation_repeated.csv", index=False)
    shap_df.to_csv(TABLE_DIR / "oof_shap_feature_importance.csv", index=False)

    print("Robust Scientific Reports analysis completed.")
    print(summary.to_string(index=False))
    print("\nLeave-one-run-out sensitivity analysis")
    print(grouped_summary.to_string(index=False))


if __name__ == "__main__":
    main()
