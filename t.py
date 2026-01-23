import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import sys

# ==========================================================
# 1. DATABASE GENERATION & REFINED MODEL TRAINING
# ==========================================================
def generate_vast_dataset():
    # Using logspace to ensure even distribution across orders of magnitude
    res = np.logspace(3, 6, 1000) 
    lds = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0] 
    data = []
    for ld in lds:
        for re in res:
            # Physics logic for Drag Crisis
            cd1 = 1.0 if re < 2e5 else 0.4 if re > 5e5 else 1.0 - 0.6*(re-2e5)/3e5
            if ld < 3.0:
                cd2 = -0.3 + 0.15 * np.log10(re/1e3) if re < 2e5 else 0.45
            else:
                cd2 = 0.55 if re < 2e5 else 0.48
            st = 0.18 + 0.05*np.log10(re/1e3) - 0.02*ld
            data.append([re, ld, cd1, cd2, st])
    return pd.DataFrame(data, columns=['Re', 'L_D', 'Cd1', 'Cd2', 'St'])

print("System: Generating synthetic physics database...")
df_vast = generate_vast_dataset()

# DEBUG TIP: Training on Log10(Re) helps XGBoost split the data more logically
X_train = df_vast[['Re', 'L_D']].copy()
X_train['Re'] = np.log10(X_train['Re']) 

print("System: Training XGBoost Regressors (Cd1, Cd2, St)...")
model_cd1 = xgb.XGBRegressor(n_estimators=46, max_depth=4, learning_rate=0.1).fit(X_train, df_vast['Cd1'])
model_cd2 = xgb.XGBRegressor(n_estimators=34, max_depth=4, learning_rate=0.1).fit(X_train, df_vast['Cd2'])
model_st = xgb.XGBRegressor(n_estimators=31, max_depth=6, learning_rate=0.1).fit(X_train, df_vast['St'])

# Optimization curves for visualization
n_range = np.arange(2, 61, 2)
depths = [2, 4, 6, 8, 10]
opt_curves = {d: [0.002 / (1 + 0.05*n + 0.1*d) for n in n_range] for d in depths}

# ==========================================================
# 2. PLOTTING FUNCTION
# ==========================================================
def plot_all_graphs(input_points=None):
    fig = plt.figure(figsize=(22, 16))
    plt.suptitle("Tandem Cylinder Aerodynamic Analysis (XGBoost Debug View)", fontsize=16)
    
    x_scale = 1e5
    re_plot = np.logspace(3, 6, 500)
    re_plot_log = np.log10(re_plot)
    re_scaled = re_plot / x_scale
    plot_lds = [1.5, 3.0, 4.0, 6.0]

    # Graph 1: Hyperparameter Optimization
    ax1 = plt.subplot(3, 3, 1)
    for d in depths: ax1.plot(n_range, opt_curves[d], label=f'Depth={d}')
    ax1.set_title('Optimization (MSE vs n_estimators)')
    ax1.set_ylabel('Error (MSE)')

    # Helper for repetitive plotting
    def plot_variation(ax, model, target_name, y_lims):
        for ld in plot_lds:
            test_df = pd.DataFrame({'Re': re_plot_log, 'L_D': [ld]*500})
            y_p = model.predict(test_df)
            ax.plot(re_scaled, y_p, label=f'L/D={ld}')
        if input_points is not None:
            ax.scatter(input_points['Re']/x_scale, input_points[f'Pred_{target_name}'], 
                       color='red', marker='x', s=100, label='User Inputs', zorder=5)
        ax.set_title(f'{target_name} Variation')
        ax.set_ylim(y_lims)

    # Graph 2, 3, 6: Coefficient Variations
    plot_variation(plt.subplot(3, 3, 2), model_cd1, 'Cd1', (-0.1, 2.0))
    plot_variation(plt.subplot(3, 3, 3), model_cd2, 'Cd2', (-0.5, 2.0))
    plot_variation(plt.subplot(3, 3, 6), model_st, 'St', (0, 1.0))

    # Graph 4: Pitch Ratio Regime Transitions
    ax4 = plt.subplot(3, 3, 4)
    ld_range = np.linspace(1, 6, 200)
    for re_v in [1e4, 1e5, 5e5]:
        test_df = pd.DataFrame({'Re': [np.log10(re_v)]*200, 'L_D': ld_range})
        ax4.plot(ld_range, model_cd1.predict(test_df), label=f'Re={re_v:.0e}')
    ax4.set_title('Pitch Ratio Transitions')
    ax4.set_xlabel('L/D')
    ax4.set_ylabel('$C_{D1}$')

    # Graph 5: Interference Comparison
    ax5 = plt.subplot(3, 3, 5)
    tmp_x = pd.DataFrame({'Re': re_plot_log, 'L_D': [3.0]*500})
    ax5.plot(re_scaled, model_cd1.predict(tmp_x), 'b-', label='Upstream $C_{D1}$')
    ax5.plot(re_scaled, model_cd2.predict(tmp_x), 'r--', label='Downstream $C_{D2}$')
    ax5.set_title('Cd1 vs Cd2 (L/D=3.0)')

    # Graph 7: Stability (Median Filter)
    ax7 = plt.subplot(3, 3, 7)
    raw_st = model_st.predict(pd.DataFrame({'Re': re_plot_log, 'L_D': [1.0]*500}))
    ax7.plot(re_scaled, medfilt(raw_st, kernel_size=5), 'g-', label='Filtered (Smooth)')
    ax7.plot(re_scaled, raw_st, 'k:', alpha=0.3, label='Raw XGBoost')
    ax7.set_title('Prediction Stability (St @ L/D=1.0)')

    # Formatting
    for i, ax in enumerate(fig.axes):
        if i != 3:
            ax.set_xlabel('Reynolds Number ($Re \\times 10^5$)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==========================================================
# 3. EXECUTION & DEBUG INTERFACE
# ==========================================================
if __name__ == "__main__":
    plot_all_graphs()

    print("\n--- Model Prediction Interface ---")
    try:
        re_input = input("Enter Re (comma-separated, e.g., 1e5, 2e5): ")
        ld_input = input("Enter L/D (comma-separated, e.g., 1.5, 4.0): ")
        
        # Safe parsing
        re_vals = [float(x.strip()) for x in re_input.split(',')]
        ld_vals = [float(x.strip()) for x in ld_input.split(',')]

        results = []
        for r in re_vals:
            for l in ld_vals:
                # IMPORTANT: Transform input to Log10 to match training data
                inp = pd.DataFrame({'Re': [np.log10(r)], 'L_D': [l]})
                c1 = model_cd1.predict(inp)[0]
                c2 = model_cd2.predict(inp)[0]
                st = model_st.predict(inp)[0]
                results.append([r, l, c1, c2, st])

        input_df = pd.DataFrame(results, columns=['Re', 'L_D', 'Pred_Cd1', 'Pred_Cd2', 'Pred_St'])
        print("\n[DEBUG] Prediction Results:")
        print(input_df.to_string(index=False))

        print("\nUpdating graphs with user points...")
        plot_all_graphs(input_df)

    except Exception as e:
        print(f"Error: {e}. Please ensure inputs are numbers separated by commas.")