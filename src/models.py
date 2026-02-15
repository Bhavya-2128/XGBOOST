"""
models.py
─────────
XGBoost model training, 5-fold CV hyperparameter search,
evaluation metrics, and model persistence.
"""

import numpy as np
import pandas as pd
import joblib
import logging
import itertools
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.ndimage import median_filter
import xgboost as xgb

logger = logging.getLogger("sqcyl")


# ─── Evaluation Metrics ───────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"R2": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4)}


# ─── 5-Fold CV Hyperparameter Search ─────────────────────────────────────

def cv_hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
    target_name: str = "target"
) -> dict:
    """
    Grid search over hyperparameter space using 5-fold CV.
    Returns the best hyperparameter dict.
    """
    hp_grid = cfg["hyperparams"]
    cv_folds = cfg["data"]["cv_folds"]
    seed = cfg["data"]["random_seed"]

    keys   = list(hp_grid.keys())
    values = list(hp_grid.values())
    all_combos = list(itertools.product(*values))

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    best_mse  = np.inf
    best_params = None

    logger.info(f"[{target_name}] CV search over {len(all_combos)} combos …")

    for combo in all_combos:
        params = dict(zip(keys, combo))
        params["random_state"] = seed
        params["objective"] = "reg:squarederror"
        params["verbosity"] = 0

        fold_mse = []
        for tr_idx, val_idx in kf.split(X_train):
            model = xgb.XGBRegressor(**params)
            model.fit(X_train[tr_idx], y_train[tr_idx], verbose=False)
            preds = model.predict(X_train[val_idx])
            fold_mse.append(mean_squared_error(y_train[val_idx], preds))

        mean_mse = np.mean(fold_mse)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_params = params.copy()

    logger.info(f"[{target_name}] Best CV MSE = {best_mse:.5f}")
    return best_params


# ─── Train Single Model ───────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    cfg: dict,
    target_name: str = "target"
) -> tuple:
    """
    Run CV hyperparameter search, fit final model, evaluate on test set.
    Returns (fitted_model, metrics_dict, best_params).
    """
    best_params = cv_hyperparameter_search(X_train, y_train, cfg, target_name)

    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    logger.info(f"[{target_name}] Test metrics: {metrics}")

    return model, metrics, best_params


# ─── Apply Median Filter to Predictions ──────────────────────────────────

def smooth_predictions(
    preds: np.ndarray,
    cfg: dict
) -> np.ndarray:
    """
    Apply median filter to 1-D prediction array to reduce overfitting artefacts.
    As in the paper: window size 5 for St.
    """
    if cfg["smoothing"]["apply_median_filter"]:
        w = cfg["smoothing"]["window_size"]
        return median_filter(preds, size=w)
    return preds


# ─── Save / Load Models ───────────────────────────────────────────────────

def save_model(model, target_name: str, cfg: dict):
    path = Path(cfg["paths"]["models_dir"]) / f"model_{target_name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(target_name: str, cfg: dict):
    path = Path(cfg["paths"]["models_dir"]) / f"model_{target_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model for {target_name} at {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model


# ─── Train All Targets ────────────────────────────────────────────────────

def train_all_models(df_features: pd.DataFrame, cfg: dict) -> dict:
    """
    Train XGBoost models for all targets in cfg['targets'].
    Returns dict {target_name: (model, metrics, best_params)}.
    """
    from src.preprocessing import split_dataset, FEATURE_COLS
    from src.bistable import separate_bistable_branches

    results = {}

    for target in cfg["targets"]:
        if target not in df_features.columns:
            logger.warning(f"Target '{target}' not in dataframe — skipping")
            continue

        available = df_features[target].notna().sum()
        if available < 20:
            logger.warning(f"Too few rows for {target} ({available}) — skipping")
            continue

        logger.info(f"\n{'='*50}\nTraining model for: {target}\n{'='*50}")

        if target == "St":
            # St uses two separate branch models
            df_upper, df_lower = separate_bistable_branches(df_features, target, cfg)
            for branch_name, branch_df in [("St_upper", df_upper), ("St_lower", df_lower)]:
                Xtr, Xte, ytr, yte = split_dataset(branch_df, target, cfg)
                model, metrics, params = train_model(Xtr, ytr, Xte, yte, cfg, branch_name)
                save_model(model, branch_name, cfg)
                results[branch_name] = {"model": model, "metrics": metrics, "params": params}
        else:
            Xtr, Xte, ytr, yte = split_dataset(df_features, target, cfg)
            model, metrics, params = train_model(Xtr, ytr, Xte, yte, cfg, target)
            save_model(model, target, cfg)
            results[target] = {"model": model, "metrics": metrics, "params": params}

    return results
