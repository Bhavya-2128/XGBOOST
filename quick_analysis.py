"""
Quick XGBoost Analysis - Line Graphs & Data Tables
Tandem Cylinder Aerodynamic Coefficients
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from pathlib import Path

# Create outputs directory
Path('outputs').mkdir(exist_ok=True)

print("\n" + "="*70)
print("TANDEM CYLINDER AERODYNAMIC ANALYSIS")
print("="*70 + "\n")

# ===== STEP 1: Generate Database =====
print("Step 1: Generating synthetic physics database...")

Re = np.logspace(3, 6, 1000)
LDs = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
data = []

for ld in LDs:
    for re in Re:
        # Drag crisis physics
        if re < 2e5:
            cd1 = 1.0
        elif re > 5e5:
            cd1 = 0.4
        else:
            cd1 = 1.0 - 0.6 * (re - 2e5) / 3e5
        
        # Downstream drag
        if ld < 3.0:
            cd2 = -0.3 + 0.15 * np.log10(re / 1e3) if re < 2e5 else 0.45
        else:
            cd2 = 0.55 if re < 2e5 else 0.48
        
        # Strouhal
        st = 0.18 + 0.05 * np.log10(re / 1e3) - 0.02 * ld
        
        data.append([re, ld, cd1, cd2, st])

df = pd.DataFrame(data, columns=['Re', 'L_D', 'Cd1', 'Cd2', 'St'])
print(f"[OK] Generated {len(df)} samples")
print(f"  Re range: {df['Re'].min():.2e} to {df['Re'].max():.2e}")
print(f"  L/D range: {df['L_D'].min():.2f} to {df['L_D'].max():.2f}\n")

# ===== STEP 2: Train Models =====
print("Step 2: Training XGBoost models...")

X_train = df[['Re', 'L_D']].copy()
X_train['Re'] = np.log10(X_train['Re'])

model_cd1 = xgb.XGBRegressor(n_estimators=46, max_depth=4, learning_rate=0.1, random_state=42)
model_cd1.fit(X_train, df['Cd1'], verbose=False)

model_cd2 = xgb.XGBRegressor(n_estimators=34, max_depth=4, learning_rate=0.1, random_state=42)
model_cd2.fit(X_train, df['Cd2'], verbose=False)

model_st = xgb.XGBRegressor(n_estimators=31, max_depth=6, learning_rate=0.1, random_state=42)
model_st.fit(X_train, df['St'], verbose=False)

print("[OK] Models trained successfully\n")

# ===== STEP 3: Generate Predictions =====
print("Step 3: Generating predictions for analysis...")

# Test points
re_test = [1e4, 1e5, 5e5]
ld_test = [1.5, 3.0, 4.0]

results = []
for re in re_test:
    for ld in ld_test:
        inp = pd.DataFrame({'Re': [np.log10(re)], 'L_D': [ld]})
        cd1 = model_cd1.predict(inp)[0]
        cd2 = model_cd2.predict(inp)[0]
        st = model_st.predict(inp)[0]
        
        results.append({
            'Re': f'{re:.2e}',
            'L/D': f'{ld:.2f}',
            'Cd1': f'{cd1:.4f}',
            'Cd2': f'{cd2:.4f}',
            'St': f'{st:.4f}'
        })

pred_df = pd.DataFrame(results)

print("\n" + "="*70)
print("PREDICTION RESULTS TABLE")
print("="*70)
print(pred_df.to_string(index=False))
print("="*70 + "\n")

# Save table
pred_df.to_csv('outputs/predictions_table.csv', index=False)
print("[OK] Table saved to: outputs/predictions_table.csv\n")

# ===== STEP 4: Create Line Graphs =====
print("Step 4: Creating line graphs...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Tandem Cylinder Aerodynamic Analysis - XGBoost Predictions', 
             fontsize=16, fontweight='bold')

# Reynolds range
re_plot = np.logspace(3, 6, 500)
re_plot_log = np.log10(re_plot)
re_scaled = re_plot / 1e5

plot_lds = [1.5, 3.0, 4.0, 6.0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Graph 1: Cd1 vs Re
ax = axes[0, 0]
for ld, color in zip(plot_lds, colors):
    test_df = pd.DataFrame({'Re': re_plot_log, 'L_D': [ld] * 500})
    y_pred = model_cd1.predict(test_df)
    ax.plot(re_scaled, y_pred, label=f'L/D={ld}', color=color, linewidth=2.5)

ax.set_xlabel('Reynolds Number (Re × 10⁵)', fontsize=11, fontweight='bold')
ax.set_ylabel('Drag Coefficient (Cd1)', fontsize=11, fontweight='bold')
ax.set_title('Upstream Cylinder Drag (Cd1)', fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=9)

# Graph 2: Cd2 vs Re
ax = axes[0, 1]
for ld, color in zip(plot_lds, colors):
    test_df = pd.DataFrame({'Re': re_plot_log, 'L_D': [ld] * 500})
    y_pred = model_cd2.predict(test_df)
    ax.plot(re_scaled, y_pred, label=f'L/D={ld}', color=color, linewidth=2.5)

ax.set_xlabel('Reynolds Number (Re × 10⁵)', fontsize=11, fontweight='bold')
ax.set_ylabel('Drag Coefficient (Cd2)', fontsize=11, fontweight='bold')
ax.set_title('Downstream Cylinder Drag (Cd2)', fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=9)

# Graph 3: St vs Re
ax = axes[0, 2]
for ld, color in zip(plot_lds, colors):
    test_df = pd.DataFrame({'Re': re_plot_log, 'L_D': [ld] * 500})
    y_pred = model_st.predict(test_df)
    ax.plot(re_scaled, y_pred, label=f'L/D={ld}', color=color, linewidth=2.5)

ax.set_xlabel('Reynolds Number (Re × 10⁵)', fontsize=11, fontweight='bold')
ax.set_ylabel('Strouhal Number (St)', fontsize=11, fontweight='bold')
ax.set_title('Strouhal Number (St)', fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=9)

# Graph 4: Cd1 vs L/D
ax = axes[1, 0]
re_values_ld = [1e4, 1e5, 5e5]
colors_ld = ['#1f77b4', '#ff7f0e', '#2ca02c']
ld_range = np.linspace(1, 6, 200)

for re_v, color in zip(re_values_ld, colors_ld):
    test_df = pd.DataFrame({'Re': [np.log10(re_v)] * 200, 'L_D': ld_range})
    y_pred = model_cd1.predict(test_df)
    ax.plot(ld_range, y_pred, label=f'Re={re_v:.0e}', color=color, linewidth=2.5)

ax.set_xlabel('Pitch Ratio (L/D)', fontsize=11, fontweight='bold')
ax.set_ylabel('Drag Coefficient (Cd1)', fontsize=11, fontweight='bold')
ax.set_title('Pitch Ratio Effect on Cd1', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=9)

# Graph 5: Cd1 vs Cd2 Comparison
ax = axes[1, 1]
ld_comp = 3.0
test_df = pd.DataFrame({'Re': re_plot_log, 'L_D': [ld_comp] * 500})
cd1_vals = model_cd1.predict(test_df)
cd2_vals = model_cd2.predict(test_df)

ax.plot(re_scaled, cd1_vals, 'b-', linewidth=2.5, label='Cd1 (Upstream)')
ax.plot(re_scaled, cd2_vals, 'r--', linewidth=2.5, label='Cd2 (Downstream)')
ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)

ax.set_xlabel('Reynolds Number (Re × 10⁵)', fontsize=11, fontweight='bold')
ax.set_ylabel('Drag Coefficient', fontsize=11, fontweight='bold')
ax.set_title(f'Cd1 vs Cd2 Comparison (L/D={ld_comp})', fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=10)

# Graph 6: Prediction Stability
ax = axes[1, 2]
test_df = pd.DataFrame({'Re': re_plot_log, 'L_D': [1.0] * 500})
raw_st = model_st.predict(test_df)
smooth_st = medfilt(raw_st, kernel_size=5)

ax.plot(re_scaled, smooth_st, 'g-', linewidth=2.5, label='Smoothed (Median Filter)')
ax.plot(re_scaled, raw_st, 'k:', alpha=0.4, linewidth=1.5, label='Raw Prediction')

ax.set_xlabel('Reynolds Number (Re × 10⁵)', fontsize=11, fontweight='bold')
ax.set_ylabel('Strouhal Number (St)', fontsize=11, fontweight='bold')
ax.set_title('Prediction Stability (L/D=1.0)', fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('outputs/aerodynamic_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] Graphs saved to: outputs/aerodynamic_analysis.png\n")

print("="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("\nOutput Files:")
print("  1. outputs/predictions_table.csv - Data table with predictions")
print("  2. outputs/aerodynamic_analysis.png - 6-panel line graph analysis")
print("\n[OK] Ready for further analysis!\n")
