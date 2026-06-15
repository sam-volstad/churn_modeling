"""
Telco Customer Churn Analysis

Script version of the notebook workflow for reproducible local/Docker execution.

Outputs are written to the outputs/ directory:
- cleaning_summary.csv
- model_results.csv
- cluster_summary.csv
- cluster_revenue.csv
- cluster_avg_charges.csv
- cluster_avg_tenure.csv
- driver_table.csv
- driver_category_summary.csv
- elbow_silhouette.png
- cluster_pca_scatter.png
- roc_curve_comparison.html
- precision_recall_curve_comparison.html
- cluster_churn_rate.html
- cluster_revenue.html
- cluster_avg_charges.html
- cluster_avg_tenure.html
- permutation_importance_heatmap.html
- permutation_importance_heatmap.png
- README-ready images in churn_images/
- shap_beeswarm.png
- shap_bar.png
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import kagglehub
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


RANDOM_STATE = 123
OUTPUT_DIR = Path("outputs")
IMAGE_DIR = Path("churn_images")
DATASET_NAME = "blastchar/telco-customer-churn"
CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def ensure_output_dirs(
    output_dir: Path = OUTPUT_DIR,
    image_dir: Path = IMAGE_DIR,
) -> None:
    """Create output directories if they do not already exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)


def save_matplotlib_figure(
    fig: Figure
    output_path: Path,
    readme_image_path: Path | None = None,
) -> None:
    """Save a Matplotlib figure to outputs and, optionally, README image path."""
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if readme_image_path is not None:
        fig.savefig(readme_image_path, dpi=300, bbox_inches="tight")


def save_plotly_html(fig: go.Figure, output_path: Path) -> None:
    """Save an interactive Plotly figure as HTML."""
    fig.write_html(output_path)


def load_telco_data() -> pd.DataFrame:
    """
    Load the Telco churn dataset.

    Priority:
    1. If TELCO_CSV_PATH is set, load that local file.
    2. Otherwise, download/load the KaggleHub dataset.

    Note:
    Docker runs may need internet access the first time KaggleHub downloads the dataset.
    """
    local_csv_path = os.getenv("TELCO_CSV_PATH")

    if local_csv_path:
        csv_path = Path(local_csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"TELCO_CSV_PATH was set, but file was not found: {csv_path}")
        print(f"Loading local dataset from: {csv_path}")
        return pd.read_csv(csv_path)

    print(f"Downloading/loading KaggleHub dataset: {DATASET_NAME}")
    dataset_path = kagglehub.dataset_download(DATASET_NAME)
    csv_path = Path(dataset_path) / CSV_NAME

    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found at: {csv_path}")

    return pd.read_csv(csv_path)


def clean_and_encode_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Clean raw Telco data and prepare encoded model features.

    Returns:
    - cleaned raw dataframe
    - x_raw feature matrix
    - y target vector
    """
    df = df.copy()

    original_rows = len(df)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_totalcharges = df["TotalCharges"].isna().sum()

    df = df.dropna()
    final_rows = len(df)

    cleaning_summary = pd.DataFrame(
        {
            "Metric": [
                "Rows Before Cleaning",
                "Rows After Cleaning",
                "Rows Removed",
                "Missing TotalCharges",
            ],
            "Value": [
                original_rows,
                final_rows,
                original_rows - final_rows,
                missing_totalcharges,
            ],
        }
    )
    cleaning_summary.to_csv(OUTPUT_DIR / "cleaning_summary.csv", index=False)

    print("\nData cleaning summary:")
    print(cleaning_summary.to_string(index=False))

    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    categorical_cols = categorical_cols.drop("customerID")

    df_encoded = pd.get_dummies(df, columns=list(categorical_cols), drop_first=True)

    x_raw = df_encoded.drop("Churn", axis=1)

    # customerID is an identifier and should not be used as a model feature.
    # TotalCharges is removed to reduce lifecycle/revenue leakage-like dominance
    # while still preserving tenure and MonthlyCharges as business-readable predictors.
    x_raw = x_raw.drop(["customerID", "TotalCharges"], axis=1)

    # Ensure boolean dummy columns behave cleanly with sklearn/shap.
    x_raw = x_raw.astype(float)

    y = df["Churn"].astype(int)

    return df, x_raw, y


def run_clustering_analysis(
    df: pd.DataFrame,
    x_raw: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    image_dir: Path = IMAGE_DIR,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Run PCA + KMeans clustering for customer segmentation.

    PCA is used only for clustering visualization/segmentation space,
    not as a modeling input for churn prediction.
    """
    print("Running clustering analysis...")

    cluster_features = StandardScaler().fit_transform(x_raw)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    x_pca = pca.fit_transform(cluster_features)

    k_range = range(2, 10)
    wcss = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=RANDOM_STATE,
        )
        labels = kmeans.fit_predict(x_pca)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(x_pca, labels))

    # Save elbow/silhouette chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(list(k_range), wcss, marker="o")
    ax1.set_title("Elbow Method (WCSS vs K)")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("WCSS / Inertia")

    ax2.plot(list(k_range), silhouette_scores, marker="o")
    ax2.set_title("Silhouette Method (Score vs K)")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Average Silhouette Score")

    fig.tight_layout()
    save_matplotlib_figure(
        fig,
        output_dir / "elbow_silhouette.png",
        image_dir / "Elb_Sil.png",
    )
    plt.close(fig)

    # Final selected k
    kmeans = KMeans(
        n_clusters=3,
        init="k-means++",
        n_init=10,
        random_state=RANDOM_STATE,
    )
    cluster_labels = kmeans.fit_predict(x_pca)

    cluster_df = df.copy()
    cluster_df["Cluster"] = cluster_labels

    cluster_summary = (
        cluster_df.groupby("Cluster")
        .agg(
            Customers=("Churn", "count"),
            ChurnRate=("Churn", "mean"),
        )
        .reset_index()
    )
    cluster_summary["ChurnRate"] *= 100
    cluster_summary.to_csv(output_dir / "cluster_summary.csv", index=False)

    rev_summary = (
        cluster_df.groupby("Cluster")
        .agg(Total_Revenue=("TotalCharges", "sum"))
        .reset_index()
    )
    rev_summary["Revenue_Label"] = (
        (rev_summary["Total_Revenue"] / 1000).round(1).astype(str) + "K"
    )
    rev_summary.to_csv(output_dir / "cluster_revenue.csv", index=False)

    avg_charges_summary = (
        cluster_df.groupby("Cluster")
        .agg(Avg_Charges=("MonthlyCharges", "mean"))
        .reset_index()
    )
    avg_charges_summary.to_csv(output_dir / "cluster_avg_charges.csv", index=False)

    avg_tenure_summary = (
        cluster_df.groupby("Cluster")
        .agg(Avg_Tenure=("tenure", "mean"))
        .reset_index()
    )
    avg_tenure_summary.to_csv(output_dir / "cluster_avg_tenure.csv", index=False)

    # Save PCA cluster scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        x_pca[:, 0],
        x_pca[:, 1],
        c=cluster_labels,
        s=50,
        cmap="viridis",
    )
    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=200,
        c="red",
        marker="X",
        label="Cluster Centers",
    )
    ax.set_title("KMeans Customer Clusters in PCA Space")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    save_matplotlib_figure(
        fig,
        output_dir / "cluster_pca_scatter.png",
        image_dir / "PCA_Clust.png",
    )
    plt.close(fig)

    # Save Plotly cluster charts
    fig_churn = px.bar(
        cluster_summary,
        x="Cluster",
        y="ChurnRate",
        text="ChurnRate",
        title="Customer Churn Rate by Cluster",
    )
    fig_churn.update_traces(texttemplate="%{text:.1f}%", textposition="auto")
    fig_churn.update_layout(
        yaxis_range=[0, cluster_summary["ChurnRate"].max() * 1.15],
        yaxis_title="Churn Rate (%)",
        xaxis_title="Cluster",
    )
    save_plotly_html(fig_churn, output_dir / "cluster_churn_rate.html")

    fig_revenue = px.bar(
        rev_summary,
        x="Cluster",
        y="Total_Revenue",
        text="Revenue_Label",
        title="Total Revenue by Cluster",
    )
    fig_revenue.update_traces(textposition="auto")
    fig_revenue.update_layout(
        yaxis_range=[0, rev_summary["Total_Revenue"].max() * 1.15],
        yaxis_title="Total Revenue ($)",
        xaxis_title="Cluster",
    )
    save_plotly_html(fig_revenue, output_dir / "cluster_revenue.html")

    fig_charges = px.bar(
        avg_charges_summary,
        x="Cluster",
        y="Avg_Charges",
        text="Avg_Charges",
        title="Average Monthly Charges by Cluster",
    )
    fig_charges.update_traces(texttemplate="$%{text:,.2f}", textposition="auto")
    fig_charges.update_layout(
        yaxis_range=[0, avg_charges_summary["Avg_Charges"].max() * 1.15],
        yaxis_title="Average Charges ($)",
        xaxis_title="Cluster",
    )
    save_plotly_html(fig_charges, output_dir / "cluster_avg_charges.html")

    fig_tenure = px.bar(
        avg_tenure_summary,
        x="Cluster",
        y="Avg_Tenure",
        text="Avg_Tenure",
        title="Average Tenure by Cluster",
    )
    fig_tenure.update_traces(texttemplate="%{text:.1f} months", textposition="auto")
    fig_tenure.update_layout(
        yaxis_range=[0, avg_tenure_summary["Avg_Tenure"].max() * 1.15],
        yaxis_title="Average Tenure (months)",
        xaxis_title="Cluster",
    )
    save_plotly_html(fig_tenure, output_dir / "cluster_avg_tenure.html")

    return cluster_df, cluster_labels


def build_models(
    x: pd.DataFrame,
    y: pd.Series,
    models: List[Tuple[str, Pipeline]],
    output_dir: Path = OUTPUT_DIR,
    image_dir: Path = IMAGE_DIR,
    gen_plot: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, Pipeline]]:
    """
    Train and evaluate multiple models using stratified cross-validation.

    Returns:
    - model comparison table
    - permutation feature importances
    - fitted models
    """
    print("Training and evaluating models...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    feature_importances = {}
    fitted_models = {}

    roc_fig = go.Figure()
    pr_fig = go.Figure()
    roc_curve_data = []
    pr_curve_data = []

    baseline = y.mean()

    for model_name, model in models:
        print(f"Evaluating: {model_name}")

        scores = cross_validate(
            model,
            x,
            y,
            cv=skf,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=-1,
        )

        y_proba = cross_val_predict(
            model,
            x,
            y,
            cv=skf,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        auc = roc_auc_score(y, y_proba)
        fpr, tpr, _ = roc_curve(y, y_proba)
        precision, recall, _ = precision_recall_curve(y, y_proba)
        ap_score = average_precision_score(y, y_proba)

        fitted_model = model.fit(x, y)
        fitted_models[model_name] = fitted_model

        # Permutation importance is computed on the full dataset for
        # interpretability, not as a cross-validated performance estimate.
        pi = permutation_importance(
            fitted_model,
            x,
            y,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
            n_jobs=-1,
        )

        perm_importance = pd.Series(
            pi["importances_mean"],
            index=x.columns,
        ).sort_values(ascending=False)

        feature_importances[model_name] = perm_importance

        results.append(
            {
                "model": model_name,
                "accuracy": scores["test_accuracy"].mean(),
                "precision": scores["test_precision"].mean(),
                "recall": scores["test_recall"].mean(),
                "f1": scores["test_f1"].mean(),
                "auc": scores["test_roc_auc"].mean(),
                "auc_std": scores["test_roc_auc"].std(),
                "ap_score": ap_score,
            }
        )

        if gen_plot:
            roc_curve_data.append((model_name, fpr, tpr, auc))
            pr_curve_data.append((model_name, recall, precision, ap_score))

            roc_fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{model_name} (AUC={auc:.3f})",
                )
            )

            pr_fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"{model_name} (AP={ap_score:.3f})",
                )
            )

    model_results = pd.DataFrame(results).sort_values("auc", ascending=False)
    model_results.index = pd.Index(range(1, len(model_results) + 1))
    model_results.index.name = "Rank"
    model_results.to_csv(output_dir / "model_results.csv")

    if gen_plot:
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Baseline",
                line=dict(dash="dash"),
            )
        )
        roc_fig.update_layout(
            title="ROC Curve Comparison (Cross-Validated)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        save_plotly_html(roc_fig, output_dir / "roc_curve_comparison.html")

        roc_png, roc_ax = plt.subplots(figsize=(8, 6))
        for model_name, fpr, tpr, auc in roc_curve_data:
            roc_ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
        roc_ax.plot([0, 1], [0, 1], linestyle="--", label="Random Baseline")
        roc_ax.set_title("ROC Curve Comparison (Cross-Validated)")
        roc_ax.set_xlabel("False Positive Rate")
        roc_ax.set_ylabel("True Positive Rate")
        roc_ax.legend()
        roc_png.tight_layout()
        save_matplotlib_figure(
            roc_png,
            output_dir / "roc_curve_comparison.png",
            image_dir / "ROC_AUC.png",
        )
        plt.close(roc_png)

        pr_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[baseline, baseline],
                mode="lines",
                name=f"Baseline ({baseline:.2f})",
                line=dict(dash="dash"),
            )
        )
        pr_fig.update_layout(
            title="Precision-Recall Curve Comparison (Cross-Validated)",
            xaxis_title="Recall",
            yaxis_title="Precision",
        )
        save_plotly_html(pr_fig, output_dir / "precision_recall_curve_comparison.html")

        pr_png, pr_ax = plt.subplots(figsize=(8, 6))
        for model_name, recall, precision, ap_score in pr_curve_data:
            pr_ax.plot(recall, precision, label=f"{model_name} (AP={ap_score:.3f})")
        pr_ax.plot([0, 1], [baseline, baseline], linestyle="--", label=f"Baseline ({baseline:.2f})")
        pr_ax.set_title("Precision-Recall Curve Comparison (Cross-Validated)")
        pr_ax.set_xlabel("Recall")
        pr_ax.set_ylabel("Precision")
        pr_ax.legend()
        pr_png.tight_layout()
        save_matplotlib_figure(
            pr_png,
            output_dir / "precision_recall_curve_comparison.png",
            image_dir / "Precision-Recall_Curve.png",
        )
        plt.close(pr_png)

    return model_results, feature_importances, fitted_models


def map_driver(feature: str) -> str:
    """Map model feature names to business-readable driver categories."""
    if "InternetService" in feature:
        return "Service Type Risk"
    if "Contract" in feature:
        return "Contract Stickiness"
    if "PaymentMethod" in feature or "MonthlyCharges" in feature:
        return "Pricing Sensitivity"
    if "tenure" in feature or "TotalCharges" in feature:
        return "Customer Lifecycle"
    if "TechSupport" in feature or "OnlineSecurity" in feature:
        return "Support Engagement"
    return "Other"


def run_feature_importance_analysis(
    feature_importances: Dict[str, pd.Series],
    output_dir: Path = OUTPUT_DIR,
    image_dir: Path = IMAGE_DIR,
    top_n: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create permutation importance heatmap and business driver summaries."""
    print("Running feature importance analysis...")

    fi_df = pd.DataFrame(feature_importances)

    top_features = (
        fi_df.mean(axis=1)
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    fi_top = fi_df.loc[top_features]
    fi_norm = fi_top.apply(lambda col: col / col.sum() if col.sum() != 0 else col, axis=0)

    fig = px.imshow(
        fi_norm,
        color_continuous_scale="Viridis",
        aspect="auto",
        title="Permutation Feature Importance (Normalized)",
    )
    fig.update_layout(height=700)
    save_plotly_html(fig, output_dir / "permutation_importance_heatmap.html")

    heatmap_fig, heatmap_ax = plt.subplots(figsize=(10, 8))
    heatmap = heatmap_ax.imshow(fi_norm.values, aspect="auto")
    heatmap_ax.set_title("Permutation Feature Importance (Normalized)")
    heatmap_ax.set_xticks(range(len(fi_norm.columns)))
    heatmap_ax.set_xticklabels(fi_norm.columns, rotation=45, ha="right")
    heatmap_ax.set_yticks(range(len(fi_norm.index)))
    heatmap_ax.set_yticklabels(fi_norm.index)
    heatmap_fig.colorbar(heatmap, ax=heatmap_ax, label="Normalized Importance")
    heatmap_fig.tight_layout()
    save_matplotlib_figure(
        heatmap_fig,
        output_dir / "permutation_importance_heatmap.png",
        image_dir / "Feature_Importance_Heatmap.png",
    )
    plt.close(heatmap_fig)

    driver_strength = fi_norm.mean(axis=1).sort_values(ascending=False)
    top_drivers = driver_strength.head(10)

    driver_table = top_drivers.reset_index()
    driver_table.columns = ["Feature", "Importance"]
    driver_table["Category"] = driver_table["Feature"].apply(map_driver)
    driver_table.to_csv(output_dir / "driver_table.csv", index=False)

    summary = (
        driver_table.groupby("Category")
        .agg(
            Total_Importance=("Importance", "sum"),
            Feature_Count=("Feature", "count"),
        )
        .sort_values("Total_Importance", ascending=False)
        .reset_index()
    )
    summary.to_csv(output_dir / "driver_category_summary.csv", index=False)

    return driver_table, summary


def run_shap_analysis(
    best_model: Pipeline,
    x_raw: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    image_dir: Path = IMAGE_DIR,
) -> None:
    """
    Run SHAP analysis for the best model.

    This implementation is intended for Logistic Regression pipeline:
    scaler -> model.

    If another model wins, SHAP is skipped to avoid misleading output.
    """
    print("Running SHAP analysis...")

    if "scaler" not in best_model.named_steps:
        print("Skipping SHAP: best model does not contain a scaler step.")
        return

    lr_model = best_model.named_steps.get("model")

    if not isinstance(lr_model, LogisticRegression):
        print("Skipping SHAP: best model is not Logistic Regression.")
        return

    feature_names = x_raw.columns.tolist()
    x_shap = x_raw.astype(float)

    scaler = best_model.named_steps["scaler"]
    x_scaled = scaler.transform(x_shap)

    # LinearExplainer is appropriate for the fitted logistic regression model.
    explainer = shap.LinearExplainer(
        lr_model,
        x_scaled,
        feature_names=feature_names,
    )

    shap_values = explainer(x_scaled)
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    current_fig = plt.gcf()
    save_matplotlib_figure(
        current_fig,
        output_dir / "shap_beeswarm.png",
        image_dir / "SHAP_Beeswarm.png",
    )
    plt.close()

    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    current_fig = plt.gcf()
    save_matplotlib_figure(
        current_fig,
        output_dir / "shap_bar.png",
        image_dir / "SHAP_Bar.png",
    )
    plt.close()


def main() -> None:
    """Run full churn modeling workflow."""
    ensure_output_dirs()

    df = load_telco_data()
    df, x_raw, y = clean_and_encode_data(df)

    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Model feature matrix shape: {x_raw.shape}")
    print(f"Churn rate: {y.mean():.2%}")

    run_clustering_analysis(df, x_raw)

    models = [
        (
            "Logistic Regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=5000)),
                ]
            ),
        ),
        (
            "Random Forest",
            Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=50,
                            max_depth=10,
                            min_samples_leaf=50,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    )
                ]
            ),
        ),
        (
            "XGBoost",
            Pipeline(
                [
                    (
                        "model",
                        XGBClassifier(
                            max_depth=6,
                            learning_rate=0.05,
                            n_estimators=300,
                            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                            random_state=RANDOM_STATE,
                            eval_metric="logloss",
                            n_jobs=-1,
                        ),
                    )
                ]
            ),
        ),
    ]

    model_results, feature_importances, fitted_models = build_models(
        x_raw,
        y,
        models,
        gen_plot=True,
    )

    best_model_name = model_results.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    print("\nModel results:")
    print(model_results.round(3))
    print(f"\nBest model by AUC: {best_model_name}")

    driver_table, driver_summary = run_feature_importance_analysis(feature_importances)

    print("\nTop model drivers:")
    print(driver_table.round(4))

    print("\nDriver category summary:")
    print(driver_summary.round(4))

    run_shap_analysis(best_model, x_raw)

    print(f"\nAnalysis complete. Outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
