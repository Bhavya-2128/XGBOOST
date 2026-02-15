"""
bistable.py
───────────
Two strategies to handle bi-stable flow behaviour:

  PREPROCESSING  (for St) — separate upper/lower branch datasets,
                             train two independent models.

  POSTPROCESSING (for CD2) — polynomial curve fitting within each
                              branch after ML prediction.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger("sqcyl")


# ─── PREPROCESSING: Branch Separation (for St) ───────────────────────────

def separate_bistable_branches(
    df: pd.DataFrame,
    target: str,
    cfg: dict
) -> tuple:
    """
    Split dataset into upper-branch and lower-branch subsets.
    Rows with bistable == 1 → upper; bistable == 2 → lower.
    Rows outside bi-stable zones (bistable == 0) go into BOTH groups
    (they represent single-valued behaviour).
    """
    if "bistable" not in df.columns:
        logger.warning("No 'bistable' column found; treating all rows as single-valued")
        return df.copy(), df.copy()

    single = df[df["bistable"] == 0]
    upper  = df[df["bistable"] == 1]
    lower  = df[df["bistable"] == 2]

    df_upper = pd.concat([single, upper], ignore_index=True)
    df_lower = pd.concat([single, lower], ignore_index=True)

    logger.info(
        f"Bistable split for '{target}': "
        f"upper={len(df_upper)}, lower={len(df_lower)}, "
        f"single-valued={len(single)}"
    )
    return df_upper, df_lower


def predict_bistable_st(
    X: np.ndarray,
    LD_arr: np.ndarray,
    model_upper,
    model_lower,
    cfg: dict
) -> tuple:
    """
    Predict St for both branches. Returns (St_upper, St_lower).
    For points outside bi-stable zone, upper == lower (averaged prediction).
    """
    from src.flow_physics import bistable_mask
    from src.models import smooth_predictions

    st_upper = model_upper.predict(X)
    st_lower = model_lower.predict(X)

    st_upper = smooth_predictions(st_upper, cfg)
    st_lower = smooth_predictions(st_lower, cfg)

    bs = bistable_mask(LD_arr, cfg)
    # Outside bi-stable zone: collapse to single value
    avg = (st_upper + st_lower) / 2
    st_upper[~bs] = avg[~bs]
    st_lower[~bs] = avg[~bs]

    return st_upper, st_lower


# ─── POSTPROCESSING: Polynomial Curve Fitting (for CD2) ──────────────────

def _poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def fit_cd2_bistable_branches(
    Re_arr: np.ndarray,
    CD2_ml: np.ndarray,
    CD2_original: np.ndarray,
    cfg: dict
) -> tuple:
    """
    For the bi-stable region of CD2 (around (L/D)cr):
    1. Fit a polynomial to ML prediction for both branches.
    2. Fit additional polynomial to original data for co-shedding (upper) branch.
    Returns arrays (CD2_upper_fitted, CD2_lower_fitted).
    """
    log_Re = np.log10(Re_arr)

    # Split ML predictions into rough upper / lower halves
    median_val = np.median(CD2_ml)
    upper_mask = CD2_ml >= median_val
    lower_mask = ~upper_mask

    try:
        p_upper, _ = curve_fit(_poly3, log_Re[upper_mask], CD2_ml[upper_mask],
                               maxfev=5000)
        p_lower, _ = curve_fit(_poly3, log_Re[lower_mask], CD2_ml[lower_mask],
                               maxfev=5000)
    except RuntimeError:
        logger.warning("Curve fit failed for CD2 bi-stable; using raw ML output")
        return CD2_ml, CD2_ml

    fitted_upper = _poly3(log_Re, *p_upper)
    fitted_lower = _poly3(log_Re, *p_lower)

    # Improve upper branch using original data (avoid ML underestimation)
    original_valid = (~np.isnan(CD2_original) & upper_mask)
    if original_valid.any():
        try:
            p_orig, _ = curve_fit(_poly3, log_Re[original_valid],
                                  CD2_original[original_valid], maxfev=5000)
            fitted_upper = _poly3(log_Re, *p_orig)
            logger.info("CD2 upper branch re-fitted on original data")
        except RuntimeError:
            pass

    return fitted_upper, fitted_lower
