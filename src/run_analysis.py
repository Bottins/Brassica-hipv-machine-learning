"""Reproduce the analyses and article figures for the receiver-plant study.

The script reads the publication-ready dataset from ``data/processed`` and
writes figures and summary tables to ``results``. All reported model scores,
the feature-set ablation and SHAP rankings are recomputed from the dataset.
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
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


SEED = 42
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "processed" / "receiver_plant_dataset.xlsx"
OUT_DIR = PROJECT_DIR / "results"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

ID_COLS = ["merge_key", "run", "id sample", "labels"]
CLASS_ORDER = ["h0h0", "h0h1", "h1h0", "h1h1"]
FOCUS_CLASSES = ["h0h1", "h1h1"]

TREAT_LABEL = {
    "h0h0": "rC",
    "h0h1": "rC + herbivore",
    "h1h0": "rHIPV",
    "h1h1": "rHIPV + herbivore",
}

TREAT_LABEL_SHORT = {
    "h0h0": "rC",
    "h0h1": "rC + herb.",
    "h1h0": "rHIPV",
    "h1h1": "rHIPV + herb.",
}

TREAT_COLORS = {
    "h0h0": "#3B8C5A",  # leaf green
    "h0h1": "#2C7FB8",  # measured blue
    "h1h0": "#E6A23C",  # volatile amber
    "h1h1": "#C94C4C",  # herbivory red
}

DOMAIN_COLORS = {
    "Gas exchange and fluorescence": "#2C7FB8",
    "VOC profile (GC-MS)": "#C94C4C",
    "Leaf damage": "#3B8C5A",
}

FEATURE_NAME = {
    "gas_Photo": "Net photosynthesis",
    "gas_Cond": "Stomatal conductance",
    "gas_Ci": "Intercellular CO2",
    "gas_Fo": "Minimum fluorescence, dark",
    "gas_Fm": "Maximum fluorescence, dark",
    "gas_Fo'": "Minimum fluorescence, light",
    "gas_Fm'": "Maximum fluorescence, light",
    "gas_Fs": "Steady-state fluorescence",
    "gas_Fv/Fm": "Maximum PSII efficiency",
    "gas_Fv'/Fm'": "Light-adapted PSII efficiency",
    "gas_PhiPS2": "Actual PSII efficiency",
    "gas_PhiCO2": "CO2 fixation efficiency",
    "gas_qP": "Photochemical quenching",
    "gas_qN": "Non-photochemical quenching coeff.",
    "gas_NPQ": "Non-photochemical quenching",
    "gas_ETR": "Electron transport rate",
    "gas_Trmmol": "Transpiration rate",
    "gas_VpdL": "Leaf-to-air VPD",
    "gas_Tleaf": "Leaf temperature",
    "gas_Ci/Ca": "Ci/Ca ratio",
    "voc_a-pinene": "alpha-Pinene",
    "voc_6methyl5hepten2one": "6-methyl-5-hepten-2-one",
    "voc_b-myrcene": "beta-Myrcene",
    "voc_limonene": "Limonene",
    "voc_e-b-ocimene": "(E)-beta-Ocimene",
    "voc_linalool": "Linalool",
    "voc_camphor": "Camphor",
    "voc_a-copaene": "alpha-Copaene",
    "voc_transcaryophyllene": "trans-Caryophyllene",
    "voc_a-guaiene": "alpha-Guaiene",
    "voc_a-humulene": "alpha-Humulene",
    "voc_geranylacetone": "Geranylacetone",
    "voc_a-farnesene": "(E,E)-alpha-Farnesene",
    "voc_d-cadinene": "delta-Cadinene",
    "voc_dimethyldisulfide": "Dimethyl disulfide",
    "voc_cis3hexen1ol": "cis-3-Hexen-1-ol",
    "voc_1octen3ol": "1-Octen-3-ol",
    "voc_cis3hexenylacetate": "cis-3-Hexenyl acetate",
    "voc_nonanal": "Nonanal",
    "voc_methylsalicylate": "Methyl salicylate",
    "leaf_damage_cm2": "Leaf area consumed",
}

FEATURE_SET_LABEL = {
    "all": "All features",
    "gas+voc_no_leaf": "Gas + VOC",
    "gas_only": "Gas only",
    "voc_only": "VOC only",
    "leaf_only": "Leaf damage only",
}

GB_PARAMS_4CLASS = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.85,
    "random_state": SEED,
}

GB_PARAMS_FOCUSED = {
    "n_estimators": 150,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 1.0,
    "random_state": SEED,
}


def configure_style():
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    sns.set_theme(style="whitegrid", rc={"grid.color": "#D8DED8", "grid.linewidth": 0.7})


def clean_name(feature):
    return FEATURE_NAME.get(feature, feature.replace("gas_", "").replace("voc_", "").replace("_", " "))


def feature_domain(feature):
    if feature.startswith("gas_"):
        return "Gas exchange and fluorescence"
    if feature.startswith("voc_"):
        return "VOC profile (GC-MS)"
    return "Leaf damage"


def domain_color(feature):
    return DOMAIN_COLORS[feature_domain(feature)]


def normalize_confusion(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)


def confusion_annotations(cm):
    cm_norm = normalize_confusion(cm)
    labels = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            labels[i, j] = f"{cm[i, j]}\n{cm_norm[i, j]:.0%}"
    return labels


def evaluate_gb_cv(X, y, class_order, params, positive_label=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preds_all = np.empty(len(y), dtype=object)
    probs_all = np.zeros((len(y), len(class_order)), dtype=float)
    feature_importances = []
    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(X[train_idx])
        x_val = scaler.transform(X[val_idx])
        y_train = y[train_idx]
        y_val = y[val_idx]

        model = GradientBoostingClassifier(**params)
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        probs = model.predict_proba(x_val)

        class_to_idx = {label: idx for idx, label in enumerate(model.classes_)}
        ordered_probs = np.zeros((len(x_val), len(class_order)), dtype=float)
        for class_idx, class_name in enumerate(class_order):
            ordered_probs[:, class_idx] = probs[:, class_to_idx[class_name]]

        preds_all[val_idx] = preds
        probs_all[val_idx] = ordered_probs
        feature_importances.append(model.feature_importances_)

        fold_rows.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(y_val, preds),
                "f1": f1_score(y_val, preds, pos_label=positive_label)
                if positive_label
                else f1_score(y_val, preds, average="macro"),
            }
        )

    cm = confusion_matrix(y, preds_all, labels=class_order)
    result = {
        "preds": preds_all,
        "probs": probs_all,
        "cm": cm,
        "accuracy": accuracy_score(y, preds_all),
        "f1": f1_score(y, preds_all, pos_label=positive_label)
        if positive_label
        else f1_score(y, preds_all, average="macro"),
        "feature_importance_mean": np.mean(feature_importances, axis=0),
        "feature_importance_std": np.std(feature_importances, axis=0),
        "folds": pd.DataFrame(fold_rows),
    }

    if positive_label:
        positive_idx = class_order.index(positive_label)
        y_binary = (y == positive_label).astype(int)
        result["auc"] = roc_auc_score(y_binary, probs_all[:, positive_idx])
    else:
        result["auc"] = np.nan
    return result


def get_feature_sets(feature_cols):
    """Return the feature groups used in the ablation analysis."""
    gas_cols = [col for col in feature_cols if col.startswith("gas_")]
    voc_cols = [col for col in feature_cols if col.startswith("voc_")]
    leaf_cols = [col for col in feature_cols if col == "leaf_damage_cm2"]
    return {
        "all": feature_cols,
        "gas+voc_no_leaf": gas_cols + voc_cols,
        "gas_only": gas_cols,
        "voc_only": voc_cols,
        "leaf_only": leaf_cols,
    }


def compute_ablation(df, feature_cols):
    """Evaluate each measurement domain with the same CV used in the paper."""
    feature_sets = get_feature_sets(feature_cols)
    y_all = df["labels"].to_numpy(dtype=object)
    focus_df = df[df["labels"].isin(FOCUS_CLASSES)].copy()
    y_focus = focus_df["labels"].to_numpy(dtype=object)
    rows = []

    for set_name, columns in feature_sets.items():
        result = evaluate_gb_cv(
            df[columns].to_numpy(dtype=np.float32),
            y_all,
            CLASS_ORDER,
            GB_PARAMS_4CLASS,
        )
        rows.append(
            {
                "task": "4-class",
                "feature_set": set_name,
                "n_samples": len(y_all),
                "n_features": len(columns),
                "accuracy": result["accuracy"],
                "f1": result["f1"],
                "auc": np.nan,
            }
        )

    for set_name, columns in feature_sets.items():
        focused_result = evaluate_gb_cv(
            focus_df[columns].to_numpy(dtype=np.float32),
            y_focus,
            FOCUS_CLASSES,
            GB_PARAMS_FOCUSED,
            positive_label="h1h1",
        )
        rows.append(
            {
                "task": "focused h0h1 vs h1h1",
                "feature_set": set_name,
                "n_samples": len(y_focus),
                "n_features": len(columns),
                "accuracy": focused_result["accuracy"],
                "f1": focused_result["f1"],
                "auc": focused_result["auc"],
            }
        )

    return pd.DataFrame(rows)


def compute_shap_ranking(df, feature_cols):
    """Fit the binary model and calculate global SHAP feature importance."""
    focus_df = df[df["labels"].isin(FOCUS_CLASSES)].copy()
    x_focus = focus_df[feature_cols].to_numpy(dtype=np.float32)
    y_focus = focus_df["labels"].to_numpy(dtype=object)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_focus)
    model = GradientBoostingClassifier(**GB_PARAMS_FOCUSED)
    model.fit(x_scaled, y_focus)

    shap_values = shap.TreeExplainer(model).shap_values(x_scaled)
    if isinstance(shap_values, list):
        values = shap_values[FOCUS_CLASSES.index("h1h1")]
    elif np.ndim(shap_values) == 3:
        values = shap_values[:, :, FOCUS_CLASSES.index("h1h1")]
    else:
        values = shap_values

    return (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "mean_abs_shap": np.abs(values).mean(axis=0),
                "mean_shap": values.mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def save_dataset_overview(df, feature_cols):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})

    counts = df["labels"].value_counts().reindex(CLASS_ORDER)
    axes[0].bar(
        [TREAT_LABEL[c] for c in CLASS_ORDER],
        counts.values,
        color=[TREAT_COLORS[c] for c in CLASS_ORDER],
        edgecolor="white",
        linewidth=1.0,
    )
    axes[0].set_title("Receiver Plant Treatment Groups")
    axes[0].set_ylabel("Number of samples")
    axes[0].tick_params(axis="x", rotation=18)
    for i, value in enumerate(counts.values):
        axes[0].text(i, value + 0.8, str(value), ha="center", fontweight="bold")

    domain_counts = pd.Series([feature_domain(c) for c in feature_cols]).value_counts()
    domain_counts = domain_counts.reindex(DOMAIN_COLORS.keys())
    axes[1].barh(
        domain_counts.index,
        domain_counts.values,
        color=[DOMAIN_COLORS[d] for d in domain_counts.index],
        edgecolor="white",
        linewidth=1.0,
    )
    axes[1].set_title("Structured Data Domains")
    axes[1].set_xlabel("Number of variables")
    for i, value in enumerate(domain_counts.values):
        axes[1].text(value + 0.3, i, str(value), va="center", fontweight="bold")
    axes[1].set_xlim(0, max(domain_counts.values) + 4)

    fig.suptitle("Heterogeneous Dataset Used for Plant-Plant Communication Detection", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_dataset_overview.png", bbox_inches="tight")
    plt.close(fig)


def save_performance_summary(result_4, result_focus):
    metrics = pd.DataFrame(
        [
            {"task": "Four treatment groups", "metric": "Accuracy", "score": result_4["accuracy"]},
            {"task": "Four treatment groups", "metric": "Macro-F1", "score": result_4["f1"]},
            {"task": "Herbivore-exposed receivers", "metric": "Accuracy", "score": result_focus["accuracy"]},
            {"task": "Herbivore-exposed receivers", "metric": "F1", "score": result_focus["f1"]},
            {"task": "Herbivore-exposed receivers", "metric": "AUC", "score": result_focus["auc"]},
        ]
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    x = np.arange(len(metrics))
    colors = ["#2C7FB8", "#74A9CF", "#C94C4C", "#E07A5F", "#3B8C5A"]
    bars = ax.bar(x, metrics["score"], color=colors, edgecolor="white", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["metric"], rotation=0)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Cross-validated score")
    ax.set_title("Machine Learning Performance on Structured Plant Phenotypes")
    ax.axhline(0.5, color="#59656F", linestyle=":", linewidth=1.2, label="Binary chance level")
    for bar, score in zip(bars, metrics["score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.025, f"{score:.2f}", ha="center", fontweight="bold")
    handles = [
        mpatches.Patch(color="#2C7FB8", label="Four treatment groups"),
        mpatches.Patch(color="#C94C4C", label="rC + herbivore vs rHIPV + herbivore"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_model_performance.png", bbox_inches="tight")
    plt.close(fig)

    return metrics


def save_confusion_matrices(result_4, result_focus):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))

    sns.heatmap(
        normalize_confusion(result_4["cm"]),
        annot=confusion_annotations(result_4["cm"]),
        fmt="",
        cmap=sns.light_palette("#2C7FB8", as_cmap=True),
        vmin=0,
        vmax=1,
        xticklabels=[TREAT_LABEL[c] for c in CLASS_ORDER],
        yticklabels=[TREAT_LABEL[c] for c in CLASS_ORDER],
        ax=axes[0],
        cbar=False,
        annot_kws={"fontsize": 8.5},
    )
    axes[0].set_title("Four-Class Classifier")
    axes[0].set_xlabel("Predicted treatment")
    axes[0].set_ylabel("Observed treatment")
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        normalize_confusion(result_focus["cm"]),
        annot=confusion_annotations(result_focus["cm"]),
        fmt="",
        cmap=sns.light_palette("#C94C4C", as_cmap=True),
        vmin=0,
        vmax=1,
        xticklabels=[TREAT_LABEL[c] for c in FOCUS_CLASSES],
        yticklabels=[TREAT_LABEL[c] for c in FOCUS_CLASSES],
        ax=axes[1],
        cbar=False,
        annot_kws={"fontsize": 10},
    )
    axes[1].set_title("Focused Communication-Detection Task")
    axes[1].set_xlabel("Predicted treatment")
    axes[1].set_ylabel("Observed treatment")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].tick_params(axis="y", rotation=0)

    fig.suptitle("Cross-Validated Confusion Matrices", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def save_ablation(ablation_df):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    task_specs = [
        ("4-class", "f1", "Four Treatment Groups", "Macro-F1"),
        ("focused h0h1 vs h1h1", "auc", "Focused Receiver Comparison", "AUC"),
    ]

    feature_set_colors = {
        "all": "#59656F",
        "gas+voc_no_leaf": "#8C6BB1",
        "gas_only": "#2C7FB8",
        "voc_only": "#C94C4C",
        "leaf_only": "#3B8C5A",
    }

    for ax, (task, metric, title, ylabel) in zip(axes, task_specs):
        sub = ablation_df[ablation_df["task"] == task].copy()
        sub = sub.sort_values(metric, ascending=False)
        labels = [FEATURE_SET_LABEL.get(v, v) for v in sub["feature_set"]]
        bars = ax.bar(
            range(len(sub)),
            sub[metric],
            color=[feature_set_colors.get(v, "#59656F") for v in sub["feature_set"]],
            edgecolor="white",
            linewidth=1.0,
        )
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 0.85)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.axhline(0.5, color="#59656F", linestyle=":", linewidth=1.2)
        for bar, score in zip(bars, sub[metric]):
            if not np.isnan(score):
                ax.text(bar.get_x() + bar.get_width() / 2, score + 0.025, f"{score:.2f}", ha="center", fontweight="bold")

    fig.suptitle("Ablation Shows Value of Integrated Heterogeneous Measurements", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_feature_set_ablation.png", bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(result_focus, feature_cols, shap_df):
    fi_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "gb_importance": result_focus["feature_importance_mean"],
            "gb_importance_sd": result_focus["feature_importance_std"],
        }
    )
    fi_df["feature_label"] = fi_df["feature"].map(clean_name)
    fi_df["domain"] = fi_df["feature"].map(feature_domain)

    shap_named = shap_df.copy()
    shap_named["feature_label"] = shap_named["feature"].map(clean_name)
    shap_named["domain"] = shap_named["feature"].map(feature_domain)

    top = shap_named.sort_values("mean_abs_shap", ascending=False).head(14).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.barh(
        top["feature_label"],
        top["mean_abs_shap"],
        color=[DOMAIN_COLORS[d] for d in top["domain"]],
        edgecolor="white",
        linewidth=1.0,
    )
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Features Driving the Focused Plant-Plant Communication Classifier")
    ax.grid(True, axis="x", alpha=0.35)
    handles = [mpatches.Patch(color=color, label=domain) for domain, color in DOMAIN_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_focused_shap_features.png", bbox_inches="tight")
    plt.close(fig)

    merged = fi_df.merge(
        shap_named[["feature", "mean_abs_shap", "mean_shap"]],
        on="feature",
        how="left",
    ).sort_values("mean_abs_shap", ascending=False)
    merged.to_csv(TABLE_DIR / "focused_feature_ranking.csv", index=False)
    return merged


def save_probability_separation(result_focus, y_focus):
    positive_idx = FOCUS_CLASSES.index("h1h1")
    p_positive = result_focus["probs"][:, positive_idx]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bins = np.linspace(0, 1, 12)
    for label in FOCUS_CLASSES:
        mask = y_focus == label
        ax.hist(
            p_positive[mask],
            bins=bins,
            alpha=0.72,
            color=TREAT_COLORS[label],
            edgecolor="white",
            label=f"Observed {TREAT_LABEL[label]}",
        )
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1.4, label="Decision threshold")
    ax.set_xlabel("Predicted probability of rHIPV + herbivore")
    ax.set_ylabel("Number of samples")
    ax.set_title("Focused Classifier Probability Separation")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_probability_separation.png", bbox_inches="tight")
    plt.close(fig)


def save_pca_views(df, feature_cols, X_all, y_all):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X_all)
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(x_scaled)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for label in CLASS_ORDER:
        mask = y_all == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=54,
            alpha=0.82,
            color=TREAT_COLORS[label],
            edgecolor="white",
            linewidth=0.6,
            label=TREAT_LABEL[label],
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Multidimensional Phenotype Space")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_pca_multidimensional_space.png", bbox_inches="tight")
    plt.close(fig)

    focus_mask = df["labels"].isin(FOCUS_CLASSES).values
    x_focus = X_all[focus_mask]
    y_focus = y_all[focus_mask]
    scaler_focus = StandardScaler()
    x_focus_scaled = scaler_focus.fit_transform(x_focus)
    pca_focus = PCA(n_components=2, random_state=SEED)
    coords_focus = pca_focus.fit_transform(x_focus_scaled)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for label in FOCUS_CLASSES:
        mask = y_focus == label
        ax.scatter(
            coords_focus[mask, 0],
            coords_focus[mask, 1],
            s=62,
            alpha=0.86,
            color=TREAT_COLORS[label],
            edgecolor="white",
            linewidth=0.7,
            label=TREAT_LABEL[label],
        )
    ax.set_xlabel(f"PC1 ({pca_focus.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_focus.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Focused Receiver Phenotype Space")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_pca_focused_receiver_space.png", bbox_inches="tight")
    plt.close(fig)


def save_top_feature_boxplots(df, top_features):
    selected = top_features["feature"].head(6).tolist()
    long_df = df[["labels"] + selected].melt(id_vars="labels", var_name="feature", value_name="value")
    long_df["feature_label"] = long_df["feature"].map(clean_name)
    long_df["treatment"] = long_df["labels"].map(TREAT_LABEL_SHORT)
    treatment_order = [TREAT_LABEL_SHORT[c] for c in CLASS_ORDER]

    g = sns.catplot(
        data=long_df,
        kind="box",
        x="treatment",
        hue="treatment",
        y="value",
        col="feature_label",
        col_wrap=3,
        sharey=False,
        height=3.1,
        aspect=1.05,
        palette=[TREAT_COLORS[c] for c in CLASS_ORDER],
        order=treatment_order,
        hue_order=treatment_order,
        legend=False,
        linewidth=0.9,
        fliersize=2.0,
    )
    g.set_axis_labels("", "Observed value")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=18, labelsize=8)
    g.fig.suptitle("Top Classifier Features Across Treatment Groups", y=1.02, fontweight="bold")
    g.fig.tight_layout()
    g.fig.savefig(FIG_DIR / "09_top_feature_distributions.png", bbox_inches="tight")
    plt.close(g.fig)


def main():
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_FILE}. "
            "Run 'python src/prepare_data.py' first."
        )

    df = pd.read_excel(DATA_FILE)
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    x_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df["labels"].to_numpy(dtype=object)

    focus_mask = np.isin(y_all, FOCUS_CLASSES)
    x_focus = x_all[focus_mask]
    y_focus = y_all[focus_mask]

    result_4 = evaluate_gb_cv(x_all, y_all, CLASS_ORDER, GB_PARAMS_4CLASS)
    result_focus = evaluate_gb_cv(x_focus, y_focus, FOCUS_CLASSES, GB_PARAMS_FOCUSED, positive_label="h1h1")

    ablation_df = compute_ablation(df, feature_cols)
    shap_df = compute_shap_ranking(df, feature_cols)

    save_dataset_overview(df, feature_cols)
    metrics = save_performance_summary(result_4, result_focus)
    save_confusion_matrices(result_4, result_focus)
    save_ablation(ablation_df)
    top_features = save_feature_importance(result_focus, feature_cols, shap_df)
    save_probability_separation(result_focus, y_focus)
    save_pca_views(df, feature_cols, x_all, y_all)
    save_top_feature_boxplots(df, top_features)

    metrics.to_csv(TABLE_DIR / "model_performance_summary.csv", index=False)
    ablation_df.to_csv(TABLE_DIR / "feature_set_ablation.csv", index=False)
    shap_df.to_csv(TABLE_DIR / "shap_feature_importance.csv", index=False)
    pd.concat(
        [
            result_4["folds"].assign(task="four_class"),
            result_focus["folds"].assign(task="focused_h0h1_vs_h1h1"),
        ],
        ignore_index=True,
    ).to_csv(TABLE_DIR / "cross_validation_folds.csv", index=False)

    focused_predictions = pd.DataFrame(
        {
            "observed": y_focus,
            "predicted": result_focus["preds"],
            "probability_rHIPV_herbivore": result_focus["probs"][:, FOCUS_CLASSES.index("h1h1")],
        }
    )
    focused_predictions.to_csv(TABLE_DIR / "focused_cross_validated_predictions.csv", index=False)

    print("Scientific Reports focused analysis completed.")
    print(f"Figures: {FIG_DIR}")
    print(f"Tables:  {TABLE_DIR}")
    print(
        "Focused task: "
        f"accuracy={result_focus['accuracy']:.3f}, "
        f"F1={result_focus['f1']:.3f}, "
        f"AUC={result_focus['auc']:.3f}"
    )


if __name__ == "__main__":
    main()
