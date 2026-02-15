"""
visualizer.py
─────────────
All matplotlib plots for hydrodynamic analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import logging

logger = logging.getLogger("sqcyl")


def _save(fig, name: str, cfg: dict, dpi: int = 150):
    plot_dir = Path(cfg["paths"]["plots_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved → {path}")
    return path


# ─── 1. Coeff vs Re ───────────────────────────────────────────────────────

def plot_coeff_vs_Re(df_db: pd.DataFrame, coeff: str, LD_values: list, cfg: dict):
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.tab10
    for i, ld in enumerate(LD_values):
        sub = df_db[np.isclose(df_db["LD"], ld, atol=0.1)]
        if sub.empty:
            continue
        sub = sub.sort_values("Re")
        ax.semilogx(sub["Re"], sub[coeff], label=f"L/D={ld:.1f}",
                    color=cmap(i % 10), lw=1.8)

    # Drag crisis indicator
    Re_cr = cfg["physics"]["Re_drag_crisis"]
    ax.axvline(Re_cr, color="red", ls="--", lw=1.2, alpha=0.7, label="Re_cr")

    ax.set_xlabel("Re", fontsize=12)
    ax.set_ylabel(coeff, fontsize=12)
    ax.set_title(f"{coeff} vs Re — Square Cylinder Tandem", fontsize=13)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"{coeff}_vs_Re", cfg)


# ─── 2. Coeff vs L/D ─────────────────────────────────────────────────────

def plot_coeff_vs_LD(df_db: pd.DataFrame, coeff: str, Re_values: list, cfg: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    for i, re in enumerate(Re_values):
        sub = df_db[np.isclose(np.log10(df_db["Re"]), np.log10(re), atol=0.15)]
        if sub.empty:
            continue
        sub = sub[sub["alpha"] < 1].sort_values("LD")
        ax.plot(sub["LD"], sub[coeff],
                label=f"Re={re:.0e}", color=cmap(i / max(len(Re_values)-1, 1)), lw=2)

    ax.set_xlabel("L/D", fontsize=12)
    ax.set_ylabel(coeff, fontsize=12)
    ax.set_title(f"{coeff} vs L/D — Square Cylinder", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"{coeff}_vs_LD", cfg)


# ─── 3. Bi-stable St branches ────────────────────────────────────────────

def plot_bistable_st(df_db: pd.DataFrame, LD_val: float, cfg: dict):
    sub = df_db[np.isclose(df_db["LD"], LD_val, atol=0.1)].sort_values("Re")
    if sub.empty or "St_upper" not in sub.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(sub["Re"], sub["St_upper"], "b-",  lw=2, label="St upper branch")
    ax.semilogx(sub["Re"], sub["St_lower"], "b--", lw=2, label="St lower branch")
    ax.set_xlabel("Re"); ax.set_ylabel("St")
    ax.set_title(f"Bi-stable St at L/D = {LD_val} — Square Cylinder")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"bistable_St_LD{LD_val:.2f}", cfg)


# ─── 4. Regime Map ────────────────────────────────────────────────────────

def plot_regime_map(df_db: pd.DataFrame, cfg: dict):
    pivot = df_db[df_db["alpha"] < 1].pivot_table(
        values="regime", index="LD", columns="Re", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = mcolors.ListedColormap(["#4dac26", "#d7191c", "#2c7bb6"])
    im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values,
                       cmap=cmap, vmin=0, vmax=2)
    cbar = fig.colorbar(im, ax=ax, ticks=[0.33, 1.0, 1.66])
    cbar.ax.set_yticklabels(["Extended-body", "Reattachment", "Co-shedding"])
    ax.set_xscale("log")
    ax.set_xlabel("Re", fontsize=12); ax.set_ylabel("L/D", fontsize=12)
    ax.set_title("Flow Regime Map — Square Cylinder Tandem", fontsize=13)
    fig.tight_layout()
    return _save(fig, "regime_map", cfg)


# ─── 5. Drag Crisis Zoom ──────────────────────────────────────────────────

def plot_drag_crisis(df_db: pd.DataFrame, cfg: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df_db[(df_db["Re"] > 1e4) & (df_db["Re"] < 1e6) & (df_db["alpha"] < 1)]
    LD_vals = [1.5, 3.0, 5.0]
    cmap = plt.cm.Set1
    for i, ld in enumerate(LD_vals):
        s = sub[np.isclose(sub["LD"], ld, atol=0.15)].sort_values("Re")
        if "CD1" in s.columns:
            ax.semilogx(s["Re"], s["CD1"], color=cmap(i), lw=2, label=f"CD1 L/D={ld}")
    Re_cr = cfg["physics"]["Re_drag_crisis"]
    ax.axvspan(2e5, Re_cr, alpha=0.12, color="orange", label="Drag crisis zone")
    ax.set_xlabel("Re"); ax.set_ylabel("CD1")
    ax.set_title("Drag Crisis Region — Square Cylinder"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "drag_crisis_zoom", cfg)


# ─── 6. Feature Importance ────────────────────────────────────────────────

def plot_feature_importance(model, target_name: str, feature_names: list, cfg: dict):
    fi = model.feature_importances_
    idx = np.argsort(fi)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(fi)), fi[idx], color="#2196F3", edgecolor="k", lw=0.5)
    ax.set_xticks(range(len(fi)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=35, ha="right")
    ax.set_ylabel("Feature Importance (gain)")
    ax.set_title(f"Feature Importance — {target_name}")
    fig.tight_layout()
    return _save(fig, f"feature_importance_{target_name}", cfg)


# ─── 7. Actual vs Predicted ───────────────────────────────────────────────

def plot_actual_vs_predicted(y_true, y_pred, target_name: str, cfg: dict):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=18, alpha=0.6, color="#1565C0", edgecolors="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    ax.set_xlabel(f"Actual {target_name}"); ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title(f"Actual vs Predicted — {target_name}"); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"actual_vs_pred_{target_name}", cfg)


# ─── 8. Residuals ────────────────────────────────────────────────────────

def plot_residuals(y_true, y_pred, target_name: str, cfg: dict):
    resid = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_pred, resid, s=14, alpha=0.5, color="#E65100")
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residuals vs Fitted — {target_name}")
    axes[1].hist(resid, bins=30, color="#7B1FA2", edgecolor="w", lw=0.4)
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    fig.tight_layout()
    return _save(fig, f"residuals_{target_name}", cfg)


# ─── 9. Data Coverage ─────────────────────────────────────────────────────

def plot_data_coverage(df_raw: pd.DataFrame, cfg: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, target, color in zip(axes, ["CD1", "St"], ["#0277BD", "#2E7D32"]):
        sub = df_raw.dropna(subset=[target])
        ax.scatter(sub["Re"], sub["LD"], s=12, c=color, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("Re"); ax.set_ylabel("L/D")
        ax.set_title(f"Data Coverage — {target} ({len(sub)} points)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "data_coverage", cfg)


# ─── 10. CD1 vs CD2 comparison ───────────────────────────────────────────

def plot_cd1_vs_cd2(df_db: pd.DataFrame, cfg: dict):
    if "CD1" not in df_db.columns or "CD2" not in df_db.columns:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.tab10
    LD_vals = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
    for i, ld in enumerate(LD_vals):
        sub = df_db[np.isclose(df_db["LD"], ld, atol=0.1) & (df_db["alpha"] < 1)]
        sub = sub.sort_values("Re")
        ax.semilogx(sub["Re"], sub["CD1"], "-",  color=cmap(i), lw=2, label=f"CD1 L/D={ld}")
        ax.semilogx(sub["Re"], sub["CD2"], "--", color=cmap(i), lw=2, label=f"CD2 L/D={ld}")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Re"); ax.set_ylabel("CD")
    ax.set_title("CD1 (solid) vs CD2 (dashed) — Square Cylinder Tandem")
    ax.legend(ncol=3, fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "CD1_vs_CD2_comparison", cfg)


def make_all_plots(df_db: pd.DataFrame, df_raw: pd.DataFrame,
                   models_dict: dict, cfg: dict):
    """Generate all plots in one call."""
    from src.preprocessing import FEATURE_COLS

    ph = cfg["physics"]
    LD_samples = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
    Re_samples = [1e3, 2e4, 1e5, 5e5]

    plot_data_coverage(df_raw, cfg)
    plot_regime_map(df_db, cfg)
    plot_drag_crisis(df_db, cfg)
    plot_cd1_vs_cd2(df_db, cfg)

    for coeff in ["CD1", "CD2", "CL"]:
        if coeff in df_db.columns:
            plot_coeff_vs_Re(df_db, coeff, LD_samples, cfg)
            plot_coeff_vs_LD(df_db, coeff, Re_samples, cfg)

    if "St_upper" in df_db.columns:
        plot_bistable_st(df_db, 1.125, cfg)
        plot_bistable_st(df_db, 4.0, cfg)
        plot_coeff_vs_Re(df_db, "St_upper", LD_samples, cfg)
        plot_coeff_vs_LD(df_db, "St_upper", Re_samples, cfg)

    for target, entry in models_dict.items():
        model = entry.get("model")
        if model is None:
            continue
        plot_feature_importance(model, target, FEATURE_COLS, cfg)
