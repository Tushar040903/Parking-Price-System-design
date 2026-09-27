"""
Model Explainability Module
============================

Provides tools for explaining ML model predictions:
- SHAP (SHapley Additive exPlanations) values
- Feature importance analysis
- Partial Dependence Plots (PDP)
- Individual prediction explanations

Upgrade 8: Feature Importance & Explainability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────
# SHAP Analysis
# ─────────────────────────────────────────────────────────────────────

def compute_shap_values(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_samples: int = 500,
) -> Tuple:
    """
    Compute SHAP values for model explanations.

    Uses TreeExplainer for tree-based models (XGBoost, RF)
    and KernelExplainer for others.

    Parameters
    ----------
    model : trained model
        The ML model to explain.
    X_train : pd.DataFrame
        Training features (used for background in KernelExplainer).
    X_test : pd.DataFrame
        Test features to explain.
    max_samples : int
        Maximum number of samples to explain (for speed).

    Returns
    -------
    tuple
        (shap_values, explainer, X_explain)
    """
    import shap

    X_explain = X_test.iloc[:max_samples]

    # Try TreeExplainer first (faster for tree models)
    model_type = type(model).__name__
    try:
        if "XGB" in model_type or "RandomForest" in model_type or "GradientBoosting" in model_type:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)
        else:
            # Use KernelExplainer for linear/other models
            background = shap.sample(X_train, min(100, len(X_train)))
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_explain, nsamples=100)
    except Exception:
        # Fallback to KernelExplainer
        background = shap.sample(X_train, min(50, len(X_train)))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_explain, nsamples=50)

    print(f"   ✅ SHAP values computed for {len(X_explain)} samples")
    return shap_values, explainer, X_explain


def plot_shap_summary(
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Create a SHAP summary plot (beeswarm plot).

    Shows the impact of each feature on model predictions.
    Red = high feature value, Blue = low feature value.
    """
    import shap

    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_explain, show=False, plot_size=(12, 8))
    plt.title("SHAP Feature Impact Summary", fontweight="bold", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_shap_bar(
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Create a SHAP bar plot showing mean absolute SHAP values.
    """
    import shap

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_explain, plot_type="bar", show=False)
    plt.title("Mean |SHAP Value| (Feature Importance)", fontweight="bold", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    feature: str,
    interaction_feature: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Create a SHAP dependence plot for a specific feature.

    Shows how a feature's value affects its SHAP value (model output).
    """
    import shap

    fig = plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature, shap_values, X_explain,
        interaction_index=interaction_feature,
        show=False,
    )
    plt.title(f"SHAP Dependence: {feature}", fontweight="bold", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Feature Importance (Model-Based)
# ─────────────────────────────────────────────────────────────────────

def get_feature_importance(
    model,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    Supports tree-based models (feature_importances_) and
    linear models (coefficients).
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        importance_type = "Gini/Gain Importance"
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_)
        importance_type = "Absolute Coefficient"
    else:
        raise ValueError("Model does not have feature_importances_ or coef_")

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "type": importance_type,
    }).sort_values("importance", ascending=False)

    return imp_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Optional[str] = None,
):
    """
    Plot feature importance as a horizontal bar chart.
    """
    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))

    ax.barh(range(len(top)), top["importance"].values, color=colors,
            edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontweight="bold")
    ax.set_title(f"Top {top_n} Feature Importances ({top.iloc[0]['type']})",
                 fontweight="bold", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Partial Dependence Plots
# ─────────────────────────────────────────────────────────────────────

def plot_partial_dependence(
    model,
    X: pd.DataFrame,
    features: List[str],
    n_cols: int = 3,
    save_path: Optional[str] = None,
):
    """
    Create Partial Dependence Plots (PDP) for specified features.

    PDPs show the marginal effect of a feature on the predicted outcome.
    """
    from sklearn.inspection import PartialDependenceDisplay

    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_features == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Get feature indices
    feature_indices = [list(X.columns).index(f) for f in features if f in X.columns]
    available_features = [f for f in features if f in X.columns]

    display = PartialDependenceDisplay.from_estimator(
        model, X, available_features,
        kind="both",  # Shows ICE curves + average
        subsample=min(200, len(X)),
        n_jobs=-1,
        ax=axes.ravel()[:len(available_features)],
        random_state=42,
    )

    # Hide unused axes
    for j in range(len(available_features), n_rows * n_cols):
        axes.ravel()[j].set_visible(False)

    plt.suptitle("Partial Dependence Plots", fontweight="bold", fontsize=16, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Individual Prediction Explanations
# ─────────────────────────────────────────────────────────────────────

def explain_single_prediction(
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    index: int = 0,
    save_path: Optional[str] = None,
):
    """
    Explain a single prediction using a SHAP waterfall/force plot.
    """
    import shap

    fig = plt.figure(figsize=(14, 4))

    # Use force plot
    expected_value = shap_values.mean()  # approximation
    shap.force_plot(
        expected_value,
        shap_values[index],
        X_explain.iloc[index],
        matplotlib=True,
        show=False,
    )

    plt.title(f"SHAP Explanation for Record #{index}", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def create_explainability_report(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    top_features: int = 6,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Generate a complete explainability report:
    1. Feature importance (model-based)
    2. SHAP summary plot
    3. SHAP bar plot
    4. SHAP dependence plots for top features
    5. Partial Dependence Plots

    Returns a summary dictionary.
    """
    results = {}

    # 1. Feature Importance
    print("   📊 Computing feature importance...")
    try:
        imp_df = get_feature_importance(model, feature_names)
        results["feature_importance"] = imp_df
        save = f"{save_dir}/feature_importance.png" if save_dir else None
        plot_feature_importance(imp_df, save_path=save)
    except Exception as e:
        print(f"   ⚠️ Feature importance failed: {e}")

    # 2. SHAP Values
    print("   🔮 Computing SHAP values...")
    try:
        shap_values, explainer, X_explain = compute_shap_values(
            model, X_train, X_test, max_samples=300
        )
        results["shap_values"] = shap_values
        results["X_explain"] = X_explain

        # Summary plot
        save = f"{save_dir}/shap_summary.png" if save_dir else None
        plot_shap_summary(shap_values, X_explain, save_path=save)

        # Bar plot
        save = f"{save_dir}/shap_bar.png" if save_dir else None
        plot_shap_bar(shap_values, X_explain, save_path=save)

        # Dependence plots for top features
        top_features_list = imp_df.head(min(top_features, len(imp_df)))["feature"].tolist()
        for feat in top_features_list:
            if feat in X_explain.columns:
                save = f"{save_dir}/shap_dep_{feat}.png" if save_dir else None
                plot_shap_dependence(shap_values, X_explain, feat, save_path=save)

    except Exception as e:
        print(f"   ⚠️ SHAP analysis failed: {e}")

    # 3. Partial Dependence Plots
    print("   📈 Computing Partial Dependence Plots...")
    try:
        pdp_features = imp_df.head(min(6, len(imp_df)))["feature"].tolist()
        save = f"{save_dir}/pdp.png" if save_dir else None
        plot_partial_dependence(model, X_test, pdp_features, save_path=save)
    except Exception as e:
        print(f"   ⚠️ PDP failed: {e}")

    print("   ✅ Explainability report complete!")
    return results
