"""
database.py
───────────
Generate a dense-grid hydrodynamic coefficient database
over the full (Re, L/D, alpha) parameter space and export
to CSV + multi-sheet Excel.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger("sqcyl")


def generate_prediction_grid(cfg: dict) -> pd.DataFrame:
    """
    Create a meshgrid of (Re, L/D, alpha) covering the full parameter space.
    Returns a dataframe ready for feature-engineering.
    """
    db_cfg = cfg["database"]
    ph_cfg = cfg["physics"]

    Re_arr    = np.logspace(np.log10(ph_cfg["Re_min"]),
                            np.log10(ph_cfg["Re_max"]),
                            db_cfg["Re_points"])
    LD_arr    = np.linspace(ph_cfg["LD_min"], ph_cfg["LD_max"], db_cfg["LD_points"])
    alpha_arr = np.linspace(ph_cfg["alpha_min"], ph_cfg["alpha_max"], db_cfg["alpha_points"])

    grid = np.array(np.meshgrid(Re_arr, LD_arr, alpha_arr)).T.reshape(-1, 3)
    df = pd.DataFrame(grid, columns=["Re", "LD", "alpha"])
    logger.info(f"Generated prediction grid: {len(df):,} points")
    return df


def build_full_database(models_dict: dict, cfg: dict) -> pd.DataFrame:
    """
    Run the prediction grid through all saved models,
    returning a merged database dataframe.
    """
    from src.preprocessing import build_features, FEATURE_COLS
    from src.flow_physics import add_regime_column
    from src.bistable import predict_bistable_st

    df_grid = generate_prediction_grid(cfg)
    df_grid = build_features(df_grid)
    df_grid = add_regime_column(df_grid, cfg)

    X = df_grid[FEATURE_COLS].values
    LD_arr = df_grid["LD"].values

    targets_done = []

    for target in ["CD1", "CD2", "CL"]:
        key = target
        if key in models_dict and models_dict[key]["model"] is not None:
            df_grid[target] = models_dict[key]["model"].predict(X)
            targets_done.append(target)

    # St: use upper/lower branch models
    m_up = models_dict.get("St_upper", {}).get("model")
    m_lo = models_dict.get("St_lower", {}).get("model")
    if m_up and m_lo:
        st_up, st_lo = predict_bistable_st(X, LD_arr, m_up, m_lo, cfg)
        df_grid["St_upper"] = st_up
        df_grid["St_lower"] = st_lo
        df_grid["St"] = st_up   # default to upper branch

    logger.info(f"Database populated for: {targets_done + ['St']}")
    return df_grid


def export_database(df: pd.DataFrame, cfg: dict):
    """Save the database to CSV (per-target) and a combined Excel workbook."""
    out_dir = Path(cfg["paths"]["database_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Individual CSVs
    for col in ["CD1", "CD2", "CL", "St_upper", "St_lower"]:
        if col in df.columns:
            subset = df[["Re", "LD", "alpha", "regime_label", col]].dropna(subset=[col])
            path = out_dir / f"database_{col}.csv"
            subset.to_csv(path, index=False)
            logger.info(f"Saved {path} ({len(subset):,} rows)")

    # Full Excel workbook (multi-sheet)
    try:
        excel_path = out_dir / "full_database.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Full_Grid", index=False)
            for col in ["CD1", "CD2", "CL", "St_upper", "St_lower"]:
                if col in df.columns:
                    pivot = df.pivot_table(values=col, index="Re", columns="LD",
                                           aggfunc="mean")
                    pivot.to_excel(writer, sheet_name=f"Pivot_{col}")
        logger.info(f"Excel workbook saved → {excel_path}")
    except Exception as e:
        logger.warning(f"Excel export failed: {e}. CSV files saved successfully.")
