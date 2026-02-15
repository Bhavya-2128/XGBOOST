"""
shap_explainer.py
──────────────────
SHAP (SHapley Additive exPlanations) analysis for XGBoost models.
Produces:
  - Global beeswarm summary plot
  - Bar summary plot (mean |SHAP|)
  - Dependence plots for top features
  - Waterfall plot for a single prediction
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger("sqcyl")


def explain_model(model, X: np.ndarray, feature_names: list, target_name: str, cfg: dict):
    """
    Compute SHAP values and generate explanation plots.
    Falls back gracefully if shap not installed.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed; skipping SHAP analysis. pip install shap")
        return

    plots_dir = Path(cfg["paths"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # ── 1. Beeswarm summary ──────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          show=False, plot_size=None)
        plt.title(f"SHAP Summary (beeswarm) — {target_name}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"shap_beeswarm_{target_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"Beeswarm plot failed: {e}")

    # ── 2. Bar summary ───────────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          plot_type="bar", show=False, plot_size=None)
        plt.title(f"SHAP Mean |Values| — {target_name}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"shap_bar_{target_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"Bar plot failed: {e}")

    # ── 3. Dependence plots for top 2 features ───────────────────────────
    try:
        mean_abs = np.abs(shap_values).mean(axis=0)
        top2 = np.argsort(mean_abs)[::-1][:2]
        for fi in top2:
            fig, ax = plt.subplots(figsize=(7, 4))
            shap.dependence_plot(fi, shap_values, X,
                                 feature_names=feature_names,
                                 show=False, ax=ax)
            plt.title(f"SHAP Dependence: {feature_names[fi]} — {target_name}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"shap_dep_{target_name}_{feature_names[fi]}.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as e:
        logger.warning(f"Dependence plots failed: {e}")

    logger.info(f"SHAP plots saved for {target_name}")


def explain_all(models_dict: dict, X_test_dict: dict, feature_names: list, cfg: dict):
    """Run SHAP analysis on all trained models."""
    for target, entry in models_dict.items():
        model = entry.get("model")
        X = X_test_dict.get(target)
        if model is None or X is None:
            continue
        explain_model(model, X, feature_names, target, cfg)
