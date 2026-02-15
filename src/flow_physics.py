"""
flow_physics.py
───────────────
Flow regime classification, drag crisis detection,
critical L/D identification for square cylinders.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("sqcyl")


# ─── Flow Regime Classifier ───────────────────────────────────────────────

REGIME_LABELS = {0: "Extended-body", 1: "Reattachment", 2: "Co-shedding"}

def classify_regime(Re: float, LD: float, cfg: dict) -> int:
    """
    Return integer regime code for a square cylinder pair:
      0 → Extended-body  (L/D = 1–2D)
      1 → Reattachment   (L/D = 2–3/4D depending on Re)
      2 → Co-shedding    (L/D > (L/D)cr)
    """
    ph = cfg["physics"]
    Re_cr = ph["Re_drag_crisis"]

    if Re < Re_cr:            # subcritical
        ld_cr_lo, ld_cr_hi = ph["LD_cr_subcrit"]
        ld_cr = (ld_cr_lo + ld_cr_hi) / 2
    else:                     # supercritical
        ld_cr_lo, ld_cr_hi = ph["LD_cr_supercrit"]
        ld_cr = (ld_cr_lo + ld_cr_hi) / 2

    if LD <= 2.0:
        return 0
    elif LD < ld_cr:
        return 1
    else:
        return 2


def add_regime_column(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add 'regime' and 'regime_label' columns to dataframe."""
    df = df.copy()
    df["regime"] = df.apply(
        lambda r: classify_regime(r["Re"], r["LD"], cfg), axis=1
    )
    df["regime_label"] = df["regime"].map(REGIME_LABELS)
    return df


# ─── Drag Crisis Detection ────────────────────────────────────────────────

def is_drag_crisis_region(Re: np.ndarray, cfg: dict) -> np.ndarray:
    """Boolean mask for Re values inside the drag-crisis transition zone."""
    Re_cr = cfg["physics"]["Re_drag_crisis"]
    return (Re > 2e5) & (Re < Re_cr * 1.5)


def is_supercritical(Re: np.ndarray, cfg: dict) -> np.ndarray:
    """Boolean mask for supercritical Reynolds numbers."""
    return Re >= cfg["physics"]["Re_drag_crisis"]


# ─── Critical L/D Identification ─────────────────────────────────────────

def get_ld_critical(Re: float, cfg: dict) -> tuple:
    """
    Return the (low, high) L/D range for regime transition,
    depending on whether Re is subcritical or supercritical.
    """
    ph = cfg["physics"]
    if Re < ph["Re_drag_crisis"]:
        return tuple(ph["LD_cr_subcrit"])
    else:
        return tuple(ph["LD_cr_supercrit"])


# ─── Bi-stable Zone Check ─────────────────────────────────────────────────

def in_bistable_zone(LD: float, cfg: dict) -> bool:
    """Return True if LD falls inside any declared bi-stable L/D range."""
    for lo, hi in cfg["physics"]["bistable_LD_ranges"]:
        if lo <= LD <= hi:
            return True
    return False


def bistable_mask(LD_arr: np.ndarray, cfg: dict) -> np.ndarray:
    return np.array([in_bistable_zone(ld, cfg) for ld in LD_arr])


# ─── Physics Validation Checks ────────────────────────────────────────────

def validate_cd1_trend(Re_arr: np.ndarray, CD1_arr: np.ndarray) -> dict:
    """
    Check that predicted CD1 passes basic physics sanity checks:
    1. Monotone decrease (drag crisis) between 2e5 and 5e5
    2. Stabilisation above 5e5 (supercritical plateau)
    Returns dict of {check_name: bool}.
    """
    results = {}

    # Check drag crisis region
    mask_lo = (Re_arr > 1e5) & (Re_arr < 3e5)
    mask_hi = (Re_arr > 3e5) & (Re_arr < 6e5)
    if mask_lo.any() and mask_hi.any():
        mean_lo = CD1_arr[mask_lo].mean()
        mean_hi = CD1_arr[mask_hi].mean()
        results["drag_crisis_decrease"] = mean_hi < mean_lo
    else:
        results["drag_crisis_decrease"] = None   # not enough data

    # Check supercritical plateau (std dev small compared to range)
    mask_sc = Re_arr > 5e5
    if mask_sc.sum() > 3:
        sc_std = CD1_arr[mask_sc].std()
        sc_range = CD1_arr[mask_sc].max() - CD1_arr[mask_sc].min()
        results["supercritical_plateau"] = sc_std < 0.05
    else:
        results["supercritical_plateau"] = None

    return results
