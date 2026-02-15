"""
preprocessing.py
─────────────────
Blockage correction, TI filtering, train/test split,
feature engineering for square cylinder hydrodynamic data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger("sqcyl")


# ─── Blockage Correction ──────────────────────────────────────────────────

def west_apelt_correction(CD: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """West & Apelt (1982) blockage correction for CD."""
    return CD * (1 - (np.pi**2 / 4) * beta**2 - 0.5 * beta * CD)


def ota_correction(CD: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Ota et al. (1994) blockage correction for CD."""
    return CD * (1 - 1.80 * beta)


def glauert_correction(CD: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Glauert (1933) blockage correction for CD."""
    return ((1 - 0.6 * beta) / (1 + 0.822 * beta**2))**2 * CD


def ota_st_correction(St: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Ota et al. (1994) blockage correction for Strouhal number."""
    return St * (1 - 0.44 * beta)


def apply_blockage_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all three CD blockage corrections to experimental rows,
    average the three results, and apply St correction.
    Numerical (DNS/LES) rows are not corrected (slip BC → negligible blockage).
    """
    df = df.copy()
    beta = df["blockage"].values / 100.0   # convert % to fraction

    exp_mask = df["method"] == "EXP"

    for col in ["CD1", "CD2", "CL"]:
        if col not in df.columns:
            continue
        cd = df[col].values.copy()
        valid = exp_mask & ~np.isnan(cd)
        if valid.any():
            c1 = west_apelt_correction(cd[valid], beta[valid])
            c2 = ota_correction(cd[valid], beta[valid])
            c3 = glauert_correction(cd[valid], beta[valid])
            df.loc[valid, col] = (c1 + c2 + c3) / 3.0
            logger.debug(f"Blockage-corrected {col}: {valid.sum()} rows")

    if "St" in df.columns:
        valid = exp_mask & ~np.isnan(df["St"].values)
        if valid.any():
            df.loc[valid, "St"] = ota_st_correction(
                df.loc[valid, "St"].values,
                beta[valid]
            )
            logger.debug(f"Blockage-corrected St: {valid.sum()} rows")

    return df


# ─── Data Selection ───────────────────────────────────────────────────────

def select_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply physics-guided data selection rules:
    - Re range filter
    - L/D range filter
    - Exclude RANS results
    - Exclude experimental St with TI > threshold
    - Keep DNS/LES only if mesh-dependence study noted (col 'mesh_study' == 1)
    """
    ph = cfg["physics"]
    df = df.copy()

    # Ensure numeric types
    df["Re"] = pd.to_numeric(df["Re"], errors="coerce")
    df["LD"] = pd.to_numeric(df["LD"], errors="coerce")
    df = df.dropna(subset=["Re", "LD"])

    # Re and L/D bounds
    df = df[(df["Re"] >= ph["Re_min"]) & (df["Re"] <= ph["Re_max"])]
    df = df[(df["LD"] >= ph["LD_min"]) & (df["LD"] <= ph["LD_max"])]

    # Exclude RANS
    df = df[df["method"] != "RANS"]

    # TI filter for St
    if "TI" in df.columns and "St" in df.columns:
        bad_ti = (df["method"] == "EXP") & (df["TI"] > ph["TI_max_St"])
        df.loc[bad_ti, "St"] = np.nan
        logger.info(f"Removed St for {bad_ti.sum()} high-TI rows (TI > {ph['TI_max_St']}%)")

    # Optional: require mesh study for DNS/LES
    if "mesh_study" in df.columns:
        num_mask = df["method"].isin(["DNS", "LES"])
        no_mesh = num_mask & (df["mesh_study"] == 0)
        df = df[~no_mesh]
        logger.info(f"Removed {no_mesh.sum()} DNS/LES rows without mesh study")

    logger.info(f"Dataset after selection: {len(df)} rows")
    return df.reset_index(drop=True)


# ─── Feature Engineering ─────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived input features to enrich the ML model:
      log10_Re        — log-scale Reynolds number
      LD_inv          — 1/LD (captures gap-flow physics)
      Re_LD           — interaction term
      alpha_rad       — angle in radians
      cos_alpha       — cosine of attack angle
      sin_alpha       — sine of attack angle
    """
    df = df.copy()
    df["log10_Re"]  = np.log10(df["Re"])
    df["LD_inv"]    = 1.0 / df["LD"]
    df["Re_LD"]     = df["Re"] * df["LD"]
    df["log_Re_LD"] = np.log10(df["Re_LD"])

    if "alpha" in df.columns:
        df["alpha_rad"]  = np.deg2rad(df["alpha"])
        df["cos_alpha"]  = np.cos(df["alpha_rad"])
        df["sin_alpha"]  = np.sin(df["alpha_rad"])
    else:
        df["alpha"]     = 0.0
        df["alpha_rad"] = 0.0
        df["cos_alpha"] = 1.0
        df["sin_alpha"] = 0.0

    return df


FEATURE_COLS = [
    "log10_Re", "LD", "LD_inv", "log_Re_LD",
    "cos_alpha", "sin_alpha",
]


# ─── Train/Test Split ─────────────────────────────────────────────────────

def split_dataset(df: pd.DataFrame, target: str, cfg: dict):
    """
    Return (X_train, X_test, y_train, y_test) for a given target column.
    Drops rows where target is NaN.
    """
    sub = df.dropna(subset=[target]).copy()
    X = sub[FEATURE_COLS].values
    y = sub[target].values
    seed = cfg["data"]["random_seed"]
    ratio = cfg["data"]["test_ratio"]
    return train_test_split(X, y, test_size=ratio, random_state=seed)


# ─── Sample Data Generator ────────────────────────────────────────────────

def generate_sample_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Synthesise a plausible literature-style dataset for square cylinders.
    Uses approximate empirical relationships so the XGBoost model has
    something physically meaningful to learn from.
    """
    rng = np.random.default_rng(seed)

    Re   = 10 ** rng.uniform(3, 6, n)
    LD   = rng.uniform(1.0, 6.0, n)
    alpha = rng.choice([0.0, 15.0, 30.0, 45.0], n)
    TI   = rng.choice([0.0, 0.1, 0.4, 0.6, 1.0, 2.0, 3.0], n)
    beta = rng.uniform(2.0, 12.0, n)
    method = rng.choice(["EXP", "DNS", "LES", "EXP", "EXP"], n)

    # Approximate physics-based target values
    log_Re = np.log10(Re)

    # CD1: upstream square cylinder drag (~2.0 at low Re, drag crisis ~4×10^5)
    CD1_base = 2.1 - 0.3 * np.clip((log_Re - 3) / 2, 0, 1)
    # drag crisis
    CD1_base[Re > 3e5] *= 0.65
    # L/D effect: slight increase near (L/D)cr
    CD1_base += 0.1 * np.exp(-((LD - 3.5)**2) / 0.5)
    # alpha effect
    CD1_base *= (1 + 0.15 * np.sin(np.deg2rad(alpha)))
    CD1 = CD1_base + rng.normal(0, 0.05, n)

    # CD2: downstream cylinder — negative in extended-body, positive in co-shedding
    transition = 1 / (1 + np.exp(-3 * (LD - 3.5)))
    CD2 = -0.4 * (1 - transition) + 0.5 * transition
    CD2 += 0.08 * rng.standard_normal(n)

    # CL: RMS lift — peaks near (L/D)cr
    CL = 0.5 + 0.3 * np.exp(-((LD - 3.5)**2) / 1.0)
    CL *= (1 + 0.1 * np.sin(np.deg2rad(2 * alpha)))
    CL = np.abs(CL + 0.05 * rng.standard_normal(n))

    # St: Strouhal number ~0.12–0.20, drops at drag crisis
    St = 0.155 + 0.02 * (LD - 3.5) / 2.5
    St[Re > 3e5] *= 0.9
    St = np.abs(St + 0.008 * rng.standard_normal(n))

    # bistable flag: 1 = upper branch, 2 = lower branch in bistable zone
    bistable = np.zeros(n, dtype=int)
    bs_mask = ((LD >= 1.0) & (LD <= 2.0)) | ((LD >= 3.0) & (LD <= 5.0))
    bistable[bs_mask] = rng.choice([1, 2], bs_mask.sum())

    df = pd.DataFrame({
        "Re": Re, "LD": LD, "alpha": alpha,
        "TI": TI, "blockage": beta, "method": method,
        "CD1": CD1, "CD2": CD2, "CL": CL, "St": St,
        "bistable": bistable, "mesh_study": 1,
        "source": "synthetic"
    })

    # Null out St for high-TI experimental rows (as in real data selection)
    df.loc[(df["method"] == "EXP") & (df["TI"] > 1.0), "St"] = np.nan

    return df
