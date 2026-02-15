"""
Interactive XGBoost Analysis with Line Graphs and Data Tables
Tandem Cylinder Aerodynamic Coefficient Prediction
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import sys

# ==========================================================
# 1. DATABASE GENERATION & MODEL TRAINING
# ==========================================================

def generate_vast_dataset():
    """Generate synthetic physics-based database"""
    print("Generating synthetic physics database...")
    
    Re = np.logspace(3, 6, 1000)
    LDs = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
    data = []
    
    for ld in LDs:
        for re in Re:
            # Physics-based drag crisis model
            if re < 2e5:
                cd1 = 1.0
            elif re > 5e5:
                cd1 = 0.4
            else:
                cd1 = 1.0 - 0.6 * (re - 2e5) / 3e5
            
            # Downstream cylinder drag
            if ld < 3.0:
                cd2 = -0.3 + 0.15 * np.log10(re / 1e3) if re < 2e5 else 0.45
            else:
                cd2 = 0.55 if re < 2e5 else 0.48
            
            # Strouhal number
            st = 0.18 + 0.05 * np.log10(re / 1e3) - 0.02 * ld
            
            data.append([re, ld, cd1, cd2, st])
    
    return pd.DataFrame(data, columns=['Re', 'L_D', 'Cd1', 'Cd2', 'St'])


def train_models(df):
    """Train XGBoost models"""
    print("Training XGBoost models...")
    
    X_train = df[['Re', 'L_D']].copy()
    X_train['Re'] = np.log10(X_train['Re'])
    
    model_cd1 = xgb.XGBRegressor(n_estimators=46, max_depth=4, learning_rate=0.1, random_state=42)
    model_cd1.fit(X_train, df['Cd1'])
    
    model_cd2 = xgb.XGBRegressor(n_estimators=34, max_depth=4, learning_rate=0.1, random_state=42)
    model_cd2.fit(X_train, df['Cd2'])
    
    model_st = xgb.XGBRegressor(n_estimators=31, max_depth=6, learning_rate=0.1, random_state=42)
    model_st.fit(X_train, df['St'])
    
    return model_cd1, model_cd2, model_st


# ==========================================================
# 2. PREDICTION & DATA TABLE GENERATION
# ==========================================================

def generate_prediction_table(model_cd1, model_cd2, model_st, re_values, ld_values):
    """Generate prediction table for given Re and L/D values"""
    results = []
    
    for re in re_values:
        for ld in ld_values:
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
    
    return pd.DataFrame(results)


# ==========================================================
# 3. PLOTTING FUNCTIONS
# ==========================================================

def plot_line_graphs(model_cd1, model_cd2, model_st, user_points=None):
    """Create comprehensive line graphs"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Tandem Cylinder Aerodynamic Analysis - XGBoost Predictions', fontsize=16, fontweight='bold')
    
    # Reynolds number range for plotting
    re_plot = np.logspace(3, 6, 500)
    re_plot_log = np.log10(re_plot)
    re_scaled = re_plot / 1e5
    
    plot_lds = [1.5, 3.0, 4.0, 6.0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # ===== Graph 1: Cd1 vs Re =====
    ax = axes[0, 0]
    for ld, color in zip(plot_lds, colors):
        test_df = pd.DataFrame({'Re': re_plot_log, 'L_D': [ld] * 500})
        y_pred = model_cd1.predict(test_df)
        ax.plot(re_scaled, y_pred, label=f'L/D={ld}', color=color, linewidth=2)
    
    if user_points is not None:
        ax.scatter(user_points['Re_scaled'], user_points['Cd1'], 
                  color='red', marker='o', s=100, label='User Input', zorder=5, edgecolors='darkred', linewidth=2)
    
    ax.set_xlabel('Reynolds Number (Re × 10⁵)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Drag Coefficient (Cd1)', fontsize=11, fontweight='bold')
    ax.set_title('Upstream Cylinder Drag (Cd1)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9, loc='best')
    
    # ===== Graph 2: Cd2 vs Re =====
    ax = axes[0, 1]
    for ld, color in zip(plot_lds, colors):
        test_df = pd.DataFrame({'Re': re_plot_log, 'L_D': [ld] * 500})
        y_pred = model_cd2.predict(test_df)
        ax.plot(re_scaled, y_pred, label=f'L/D={ld}', color=color, linewidth=2)
    
    if user_points is not None:
        ax.scatter(user_points['Re_scaled'], user_points['Cd2'], 
                  color='red', marker='o', s=100, label='User Input', zorder=5, edgecolors='darkred', linewidth=2)
    
    ax.set_xlabel('Reynolds Number (Re × 10⁵)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Drag Coefficient (Cd2)', fontsize=11, fontweight='bold')
    ax.set_title('Downstream Cylinder Drag (Cd2)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9, loc='best')
    
    # ===== Graph 3: St vs Re =====
    ax = axes[0, 2]
    for ld, color in zip(plot_lds, colors):
        test_df = pd.DataFrame({'Re': re_plot_log, 'L_D': [ld] * 500})
        y_pred = model_st.predict(test_df)
        ax.plot(re_scaled, y_pred, label=f'L/D={ld}', color=color, linewidth=2)
    
    if user_points is not None:
        ax.scatter(user_points['Re_scaled'], user_points['St'], 
                  color='red', marker='o', s=100, label='User Input', zorder=5, edgecolors='darkred', linewidth=2)
    
    ax.set_xlabel('Reynolds Number (Re × 10⁵)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Strouhal Number (St)', fontsize=11, fontweight='bold')
    ax.set_title('Strouhal Number (St)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9, loc='best')
    
    # ===== Graph 4: Cd1 vs L/D =====
    ax = axes[1, 0]
    re_values_ld = [1e4, 1e5, 5e5]
    colors_ld = ['#1f77b4', '#ff7f0e', '#2ca02c']
    ld_range = np.linspace(1, 6, 200)
    
    for re_v, color in zip(re_values_ld, colors_ld):
        test_df = pd.DataFrame({'Re': [np.log10(re_v)] * 200, 'L_D': ld_range})
        y_pred = model_cd1.predict(test_df)
        ax.plot(ld_range, y_pred, label=f'Re={re_v:.0e}', color=color, linewidth=2)
    
    ax.set_xlabel('Pitch Ratio (L/D)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Drag Coefficient (Cd1)', fontsize=11, fontweight='bold')
    ax.set_title('Pitch Ratio Effect on Cd1', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9, loc='best')
    
    # ===== Graph 5: Cd1 vs Cd2 Comparison =====
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
    ax.legend(fontsize=10, loc='best')
    
    # ===== Graph 6: Prediction Stability (Smoothing) =====
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
    ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    plt.savefig('outputs/aerodynamic_analysis.png', dpi=300, bbox_inches='tight')
    print("Graph saved to: outputs/aerodynamic_analysis.png")
    plt.show()


# ==========================================================
# 4. MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TANDEM CYLINDER AERODYNAMIC ANALYSIS - XGBoost Interactive Tool")
    print("="*70 + "\n")
    
    # Generate and train
    df_vast = generate_vast_dataset()
    model_cd1, model_cd2, model_st = train_models(df_vast)
    
    print("\nModels trained successfully!")
    print(f"Training dataset size: {len(df_vast)} samples")
    print(f"Re range: {df_vast['Re'].min():.2e} to {df_vast['Re'].max():.2e}")
    print(f"L/D range: {df_vast['L_D'].min():.2f} to {df_vast['L_D'].max():.2f}\n")
    
    # User input
    print("-" * 70)
    print("PREDICTION INTERFACE")
    print("-" * 70)
    
    try:
        re_input = input("\nEnter Reynolds numbers (comma-separated, e.g., 1e4, 1e5, 5e5): ").strip()
        ld_input = input("Enter L/D values (comma-separated, e.g., 1.5, 3.0, 4.0): ").strip()
        
        # Parse inputs
        re_vals = [float(x.strip()) for x in re_input.split(',')]
        ld_vals = [float(x.strip()) for x in ld_input.split(',')]
        
        # Generate prediction table
        print("\n" + "="*70)
        print("PREDICTION RESULTS TABLE")
        print("="*70)
        
        pred_table = generate_prediction_table(model_cd1, model_cd2, model_st, re_vals, ld_vals)
        print("\n" + pred_table.to_string(index=False))
        
        # Save table to CSV
        pred_table.to_csv('outputs/predictions_table.csv', index=False)
        print("\n✓ Table saved to: outputs/predictions_table.csv")
        
        # Prepare user points for plotting
        user_points_data = []
        for re in re_vals:
            for ld in ld_vals:
                inp = pd.DataFrame({'Re': [np.log10(re)], 'L_D': [ld]})
                cd1 = model_cd1.predict(inp)[0]
                cd2 = model_cd2.predict(inp)[0]
                st = model_st.predict(inp)[0]
                
                user_points_data.append({
                    'Re': re,
                    'Re_scaled': re / 1e5,
                    'L_D': ld,
                    'Cd1': cd1,
                    'Cd2': cd2,
                    'St': st
                })
        
        user_points_df = pd.DataFrame(user_points_data)
        
        # Generate plots
        print("\nGenerating line graphs...")
        plot_line_graphs(model_cd1, model_cd2, model_st, user_points_df)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print("✓ Graphs saved to: outputs/aerodynamic_analysis.png")
        print("✓ Data table saved to: outputs/predictions_table.csv")
        
    except ValueError as e:
        print(f"\n❌ Error: Invalid input format. Please enter numbers separated by commas.")
        print(f"   Example: 1e4, 2e5, 5e5")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
