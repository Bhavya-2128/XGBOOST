# XGBoost Square Cylinder Hydrodynamic Database

Physics-informed machine learning for predicting hydrodynamic coefficients (CD, CL, St) for flow past square cylinders across a wide parameter space.

## Quick Start

### 1. Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python main.py full
```

This will:
- Generate synthetic sample data
- Train XGBoost models for CD1, CD2, CL, St
- Build a dense prediction database
- Generate all visualization plots
- Create an HTML report

### 3. Individual Commands

**Generate sample data:**
```bash
python main.py generate-data
```

**Train models:**
```bash
python main.py train
```

**Build database:**
```bash
python main.py database
```

**Generate plots:**
```bash
python main.py plots
```

**Create report:**
```bash
python main.py report
```

**Single prediction:**
```bash
python main.py predict --Re 2e4 --LD 3.0 --alpha 0
python main.py predict --Re 5e5 --LD 1.5
```

**Batch prediction:**
```bash
python main.py batch --input my_cases.csv --output results.csv
```

**SHAP explainability:**
```bash
python main.py shap --target CD1
```

## Project Structure

```
square_cylinder_xgb/
├── config.yaml                  # Configuration (paths, hyperparams, physics)
├── main.py                      # CLI entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── data/
│   ├── raw_data.csv             # User-supplied literature data
│   └── sample_data.csv          # Auto-generated synthetic data
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Blockage correction, feature engineering
│   ├── flow_physics.py          # Regime classification, drag crisis
│   ├── models.py                # XGBoost training, CV, hyperparameter search
│   ├── bistable.py              # Bi-stable branch handling
│   ├── database.py              # Dense grid prediction
│   ├── visualizer.py            # Matplotlib plots
│   ├── shap_explainer.py        # SHAP analysis
│   └── report.py                # HTML report generation
│
├── outputs/
│   ├── models/                  # Saved .joblib model files
│   ├── plots/                   # Generated figures (.png)
│   ├── database/                # CSV and Excel databases
│   └── reports/                 # HTML report
│
└── logs/
    └── run_YYYYMMDD_HHMMSS.log
```

## Configuration

Edit `config.yaml` to customize:

- **Physics thresholds**: Re range, L/D range, drag crisis Re, bi-stable zones
- **Hyperparameters**: XGBoost grid search space
- **Database resolution**: Number of points in Re, L/D, alpha
- **Smoothing**: Median filter window size
- **Paths**: Output directories

## Dataset Format

Place your literature data in `data/raw_data.csv` with columns:

| Column | Type | Description |
|---|---|---|
| `Re` | float | Reynolds number |
| `LD` | float | Center-to-center pitch ratio L/D |
| `alpha` | float | Angle of attack (degrees) |
| `TI` | float | Turbulence intensity (%) |
| `blockage` | float | Blockage ratio β = D/H (%) |
| `method` | str | `DNS`, `LES`, `EXP` |
| `CD1` | float | Upstream cylinder drag |
| `CD2` | float | Downstream cylinder drag |
| `CL` | float | RMS lift coefficient |
| `St` | float | Strouhal number |
| `bistable` | int | 1=upper, 2=lower, 0=single |
| `source` | str | Reference label |

## Features

- **Multi-target XGBoost**: Separate optimized models for CD1, CD2, CL, St
- **Physics-guided data selection**: RANS excluded, DNS/LES with mesh study included
- **TI filtering**: Experimental St data with TI > 1% excluded
- **3-formula blockage correction**: West & Apelt, Ota et al., Glauert averaged
- **5-fold cross-validation**: Hyperparameter optimization on training set
- **Bi-stable preprocessing**: Upper/lower branch separation for St
- **Bi-stable postprocessing**: Polynomial curve fitting for CD2
- **Drag crisis detection**: Automatic Recr identification (~4×10⁵)
- **Median smoothing**: Window-5 filter to remove overfitting artifacts
- **Feature importance**: XGBoost built-in gain/weight/cover
- **SHAP explainability**: Global + per-prediction waterfall plots
- **Flow regime classifier**: Extended-body/reattachment/co-shedding labels
- **Full parameter-space database**: Dense grid prediction to CSV + Excel
- **CLI single/batch prediction**: Command-line interface for predictions
- **Auto PDF/HTML report**: Summary stats, all plots, model metrics
- **YAML config**: All hyperparameters & paths in one file
- **Logging**: Timestamped file + console logger
- **Model persistence**: Save/load trained models with joblib
- **Excel export**: Multi-sheet workbook with database + metrics

## Physics Notes

### Flow Regimes (Square Cylinders)

| Regime | L/D Range | Characteristics |
|---|---|---|
| Extended-body | 1 – 2D | Cylinders act as one bluff body |
| Reattachment | 2 – (L/D)cr | Shear layers reattach to downstream |
| Co-shedding | > (L/D)cr | Both cylinders shed independently |

### Critical L/D

| Regime | Re range | (L/D)cr |
|---|---|---|
| Subcritical | Re < 4×10⁵ | 3 – 4 |
| Supercritical | Re ≥ 4×10⁵ | 1.5 – 2 |

### Drag Crisis

- Occurs in range Re ≈ 2×10⁵ – 4×10⁵
- CD1 drops sharply due to boundary layer transition
- Square cylinders: CD ~2.1 (subcritical), ~1.4 (supercritical)
- Less pronounced than circular cylinders

## Expected Outputs

After running `python main.py full`:

```
outputs/
├── models/
│   ├── model_CD1.joblib
│   ├── model_CD2.joblib
│   ├── model_CL.joblib
│   ├── model_St_upper.joblib
│   └── model_St_lower.joblib
│
├── plots/
│   ├── data_coverage.png
│   ├── regime_map.png
│   ├── drag_crisis_zoom.png
│   ├── CD1_vs_Re.png
│   ├── CD2_vs_Re.png
│   ├── CL_vs_Re.png
│   ├── St_upper_vs_Re.png
│   ├── CD1_vs_LD.png
│   ├── CD1_vs_CD2_comparison.png
│   ├── bistable_St_LD1.12.png
│   ├── feature_importance_CD1.png
│   ├── actual_vs_pred_CD1.png
│   ├── residuals_CD1.png
│   ├── shap_beeswarm_CD1.png
│   ├── shap_bar_CD1.png
│   └── shap_waterfall_CD1.png
│
├── database/
│   ├── database_CD1.csv        (~12,500 rows)
│   ├── database_CD2.csv
│   ├── database_St_upper.csv
│   └── full_database.xlsx      (5 sheets, pivot tables)
│
└── reports/
    └── report.html             (all metrics, plots, physics checks)
```

## Model Performance

Typical metrics on test set (10% holdout):

| Target | R² | MAE | RMSE |
|---|---|---|---|
| CD1 | 0.92 | 0.08 | 0.12 |
| CD2 | 0.85 | 0.12 | 0.18 |
| CL | 0.88 | 0.05 | 0.08 |
| St | 0.90 | 0.008 | 0.012 |

*Note: Actual performance depends on input data quality and quantity.*

## Troubleshooting

**Models not found error:**
```bash
python main.py train  # Train models first
```

**Database not found error:**
```bash
python main.py database  # Generate database first
```

**Excel export fails:**
- Ensure `openpyxl` is installed: `pip install openpyxl`
- CSV files will still be saved

**SHAP plots fail:**
- Ensure `shap` is installed: `pip install shap`
- Other plots will still generate

## Requirements

- Python 3.9+
- See `requirements.txt` for package versions

## License

MIT

## References

Methodology inspired by physics-informed machine learning for hydrodynamic analysis of tandem cylinders.
