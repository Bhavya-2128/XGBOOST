"""
main.py
────────
CLI entry-point for the Square Cylinder XGBoost Hydrodynamic Database project.

Usage examples
──────────────
  python main.py generate-data           # create sample dataset
  python main.py train                   # train all XGBoost models
  python main.py database                # build dense prediction grid
  python main.py report                  # generate HTML report
  python main.py predict --Re 2e4 --LD 3.0 --alpha 0
  python main.py predict --Re 5e5 --LD 1.5
  python main.py batch   --input my_cases.csv --output results.csv
  python main.py shap    --target CD1
  python main.py full                    # run entire pipeline
"""

import argparse
import sys
import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# ─── Logging Setup ────────────────────────────────────────────────────────

def setup_logging(cfg: dict):
    log_dir = Path(cfg["paths"]["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{ts}.log"

    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    
    # Create console handler (no encoding parameter for compatibility)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    logger = logging.getLogger("sqcyl")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ─── Load Config ─────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Prediction Helper ────────────────────────────────────────────────────

def single_predict(Re: float, LD: float, alpha: float, cfg: dict) -> dict:
    """Make a prediction for a single (Re, L/D, alpha) condition."""
    from src.preprocessing import build_features, FEATURE_COLS
    from src.models import load_model

    row = pd.DataFrame({"Re": [Re], "LD": [LD], "alpha": [alpha]})
    row = build_features(row)
    X = row[FEATURE_COLS].values

    result = {"Re": Re, "LD": LD, "alpha": alpha}

    for target in cfg["targets"]:
        if target == "St":
            try:
                m_up = load_model("St_upper", cfg)
                m_lo = load_model("St_lower", cfg)
                result["St_upper"] = float(m_up.predict(X)[0])
                result["St_lower"] = float(m_lo.predict(X)[0])
            except FileNotFoundError:
                result["St_upper"] = result["St_lower"] = None
        else:
            try:
                model = load_model(target, cfg)
                result[target] = float(model.predict(X)[0])
            except FileNotFoundError:
                result[target] = None

    return result


# ─── Sub-commands ─────────────────────────────────────────────────────────

def cmd_generate_data(cfg, logger):
    from src.preprocessing import generate_sample_data
    df = generate_sample_data(n=300)
    out = Path(cfg["paths"]["sample_data"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Sample data saved to {out} ({len(df)} rows)")


def cmd_train(cfg, logger):
    from src.preprocessing import (
        apply_blockage_corrections, select_data, build_features, generate_sample_data
    )
    from src.models import train_all_models

    raw_path = Path(cfg["paths"]["raw_data"])
    if not raw_path.exists():
        logger.warning(f"raw_data.csv not found; using sample data")
        df = generate_sample_data(n=300)
    else:
        df = pd.read_csv(raw_path)

    df = apply_blockage_corrections(df)
    df = select_data(df, cfg)
    df = build_features(df)

    models_dict = train_all_models(df, cfg)

    # Print summary table
    print("\n" + "="*60)
    print(f"{'Target':<15} {'R²':>8} {'MAE':>10} {'RMSE':>10}")
    print("-"*60)
    for name, entry in models_dict.items():
        m = entry["metrics"]
        print(f"{name:<15} {m['R2']:>8.4f} {m['MAE']:>10.4f} {m['RMSE']:>10.4f}")
    print("="*60)

    return models_dict


def cmd_database(cfg, logger):
    from src.models import load_model
    from src.database import build_full_database, export_database

    # Reload all models
    models_dict = {}
    for target in cfg["targets"] + ["St_upper", "St_lower"]:
        try:
            models_dict[target] = {"model": load_model(target, cfg)}
        except FileNotFoundError:
            models_dict[target] = {"model": None}

    df_db = build_full_database(models_dict, cfg)
    export_database(df_db, cfg)
    logger.info("Database generation complete")
    return df_db


def cmd_plots(cfg, logger):
    from src.models import load_model
    from src.visualizer import make_all_plots
    from src.preprocessing import generate_sample_data, build_features, apply_blockage_corrections, select_data

    # Load DB
    db_path = Path(cfg["paths"]["database_dir"]) / "full_database.xlsx"
    if not db_path.exists():
        logger.error("Database not found. Run 'python main.py database' first.")
        sys.exit(1)
    
    try:
        df_db = pd.read_excel(db_path, sheet_name="Full_Grid")
    except Exception as e:
        logger.error(f"Failed to read database: {e}")
        sys.exit(1)

    # Raw data for coverage plot
    raw_path = Path(cfg["paths"]["raw_data"])
    if raw_path.exists():
        df_raw = pd.read_csv(raw_path)
    else:
        df_raw = generate_sample_data(n=300)
    df_raw = apply_blockage_corrections(df_raw)
    df_raw = select_data(df_raw, cfg)
    df_raw = build_features(df_raw)

    models_dict = {}
    for target in cfg["targets"] + ["St_upper", "St_lower"]:
        try:
            models_dict[target] = {"model": load_model(target, cfg)}
        except FileNotFoundError:
            models_dict[target] = {"model": None}

    make_all_plots(df_db, df_raw, models_dict, cfg)
    print("All plots saved to outputs/plots/")


def cmd_report(cfg, logger):
    from src.report import generate_report
    from src.flow_physics import validate_cd1_trend

    # Quick physics check
    db_path = Path(cfg["paths"]["database_dir"]) / "database_CD1.csv"
    physics_checks = {}
    if db_path.exists():
        try:
            df_db = pd.read_csv(db_path, nrows=500)
            if "CD1" in df_db.columns:
                physics_checks = validate_cd1_trend(df_db["Re"].values, df_db["CD1"].values)
        except Exception as e:
            logger.warning(f"Physics check failed: {e}")

    # Gather metrics from saved model metadata (if any)
    metrics_all = {}  # populate from train step; here just placeholder
    db_stats = {"note": "Run 'python main.py train' then 'python main.py database' first"}

    path = generate_report(metrics_all, db_stats, physics_checks, cfg)
    print(f"\nReport ready: {path}")


def cmd_predict(args, cfg, logger):
    result = single_predict(args.Re, args.LD, args.alpha, cfg)
    print("\n" + "="*55)
    print(f"  Prediction for Re={args.Re:.2e}, L/D={args.LD}, alpha={args.alpha} deg")
    print("-"*55)
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:<15} {v:.5f}")
    print("="*55)


def cmd_batch(args, cfg, logger):
    from src.preprocessing import build_features, FEATURE_COLS
    from src.models import load_model

    df_in = pd.read_csv(args.input)
    required = ["Re", "LD"]
    for col in required:
        if col not in df_in.columns:
            logger.error(f"Input CSV missing required column: {col}")
            sys.exit(1)

    if "alpha" not in df_in.columns:
        df_in["alpha"] = 0.0

    df_feat = build_features(df_in.copy())
    X = df_feat[FEATURE_COLS].values

    for target in cfg["targets"]:
        if target == "St":
            for branch in ["St_upper", "St_lower"]:
                try:
                    m = load_model(branch, cfg)
                    df_in[branch] = m.predict(X)
                except FileNotFoundError:
                    pass
        else:
            try:
                m = load_model(target, cfg)
                df_in[target] = m.predict(X)
            except FileNotFoundError:
                pass

    out_path = args.output if hasattr(args, "output") and args.output else "batch_results.csv"
    df_in.to_csv(out_path, index=False)
    print(f"\nBatch results saved to {out_path} ({len(df_in)} rows)")


def cmd_shap(args, cfg, logger):
    from src.models import load_model
    from src.preprocessing import generate_sample_data, build_features, FEATURE_COLS
    from src.shap_explainer import explain_model

    target = args.target
    try:
        model = load_model(target, cfg)
    except FileNotFoundError:
        logger.error(f"Model for {target} not found. Run 'python main.py train' first.")
        sys.exit(1)
    
    df_sample = generate_sample_data(n=100)
    df_sample = build_features(df_sample)
    X = df_sample[FEATURE_COLS].values
    explain_model(model, X, FEATURE_COLS, target, cfg)
    print(f"SHAP analysis done for {target}")


def cmd_full(cfg, logger):
    """Run the entire pipeline end-to-end."""
    logger.info("Running FULL pipeline...")
    cmd_generate_data(cfg, logger)
    cmd_train(cfg, logger)
    cmd_database(cfg, logger)
    cmd_plots(cfg, logger)
    cmd_report(cfg, logger)
    logger.info("Pipeline complete")


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Square Cylinder XGBoost Hydrodynamic Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("generate-data", help="Generate synthetic sample dataset")
    sub.add_parser("train",         help="Train all XGBoost models")
    sub.add_parser("database",      help="Generate dense prediction database")
    sub.add_parser("plots",         help="Generate all visualisation plots")
    sub.add_parser("report",        help="Generate HTML summary report")
    sub.add_parser("full",          help="Run entire pipeline end-to-end")

    p_pred = sub.add_parser("predict", help="Single-point prediction")
    p_pred.add_argument("--Re",    type=float, required=True)
    p_pred.add_argument("--LD",    type=float, required=True)
    p_pred.add_argument("--alpha", type=float, default=0.0, help="Angle of attack (deg)")

    p_batch = sub.add_parser("batch", help="Batch prediction from CSV")
    p_batch.add_argument("--input",  required=True, help="Input CSV path")
    p_batch.add_argument("--output", default="batch_results.csv")

    p_shap = sub.add_parser("shap", help="SHAP analysis for a target")
    p_shap.add_argument("--target", default="CD1",
                        choices=["CD1", "CD2", "CL", "St_upper", "St_lower"])

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    cfg = load_config(args.config)
    logger = setup_logging(cfg)

    dispatch = {
        "generate-data": lambda: cmd_generate_data(cfg, logger),
        "train":         lambda: cmd_train(cfg, logger),
        "database":      lambda: cmd_database(cfg, logger),
        "plots":         lambda: cmd_plots(cfg, logger),
        "report":        lambda: cmd_report(cfg, logger),
        "full":          lambda: cmd_full(cfg, logger),
        "predict":       lambda: cmd_predict(args, cfg, logger),
        "batch":         lambda: cmd_batch(args, cfg, logger),
        "shap":          lambda: cmd_shap(args, cfg, logger),
    }

    dispatch[args.command]()


if __name__ == "__main__":
    main()
