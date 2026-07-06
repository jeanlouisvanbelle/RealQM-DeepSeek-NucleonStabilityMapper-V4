"""
analyse_v4_1_results.py - Comprehensive Analysis of V4.1 Scan Results
======================================================================
Loads V4.1 results CSV, compares against V3 and V4 baselines.
Generates confusion matrix, heatmap, and comparison plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# ============================================================
# EMPIRICAL STABILITY DATA
# ============================================================
EMPIRICAL_STABLE = {
    (1, 0): True, (1, 1): True, (1, 2): True,
    (2, 1): True, (2, 2): True,
    (3, 3): True, (3, 4): True,
    (4, 5): True,
    (5, 5): True, (5, 6): True,
    (6, 6): True, (6, 7): True, (6, 8): True,
    (7, 7): True, (7, 8): True,
    (8, 8): True, (8, 9): True, (8, 10): True,
    (9, 10): True,
    (10, 10): True, (10, 11): True, (10, 12): True,
    (11, 12): True,
    (12, 12): True, (12, 13): True, (12, 14): True,
    (13, 14): True,
    (14, 14): True, (14, 15): True, (14, 16): True,
    (15, 16): True,
    (16, 16): True, (16, 17): True, (16, 18): True,
    (17, 18): True, (17, 20): True,
    (18, 18): True, (18, 20): True, (18, 22): True,
    (19, 20): True, (19, 21): True, (19, 22): True,
    (20, 20): True, (20, 21): True, (20, 22): True, (20, 24): True, (20, 28): True,
}

def is_empirically_stable(Z, N):
    return EMPIRICAL_STABLE.get((Z, N), False)

def load_data(filename='stability_scan_v4_1_full.csv'):
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found!")
        return None
    
    df = pd.read_csv(filename, encoding='utf-8-sig')
    print(f"Columns: {df.columns.tolist()}")
    
    # Find delta_E column
    if 'delta_E' not in df.columns:
        for col in df.columns:
            if 'delta' in col.lower():
                df.rename(columns={col: 'delta_E'}, inplace=True)
                break
    
    df['delta_E'] = pd.to_numeric(df['delta_E'], errors='coerce')
    df['empirical_stable'] = df.apply(lambda row: is_empirically_stable(int(row['Z']), int(row['N'])), axis=1)
    
    return df

def generate_plots(df):
    """Generate all analysis plots."""
    
    # 1. Heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    pivot = df.pivot(index='Z', columns='N', values='delta_E')
    
    vmin = max(0, pivot.min().min() - 10) if not pivot.isna().all().all() else -10
    vmax = min(200, pivot.max().max() + 10) if not pivot.isna().all().all() else 50
    
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto',
                   extent=[pivot.columns.min()-0.5, pivot.columns.max()+0.5,
                           pivot.index.max()+0.5, pivot.index.min()-0.5],
                   vmin=vmin, vmax=vmax)
    
    ax.plot([0, pivot.index.max()], [0, pivot.index.max()], 'k--', linewidth=2, alpha=0.7, label='N = Z')
    
    for (Z, N), stable in EMPIRICAL_STABLE.items():
        if stable and Z in pivot.index and N in pivot.columns:
            if not np.isnan(pivot.loc[Z, N]):
                ax.plot(N, Z, 'wo', markersize=8, markerfacecolor='none', markeredgecolor='white', markeredgewidth=1.5)
    
    ax.set_xlabel('Neutron Number (N)', fontsize=14)
    ax.set_ylabel('Proton Number (Z)', fontsize=14)
    ax.set_title('V4.1 Stability Heatmap (A-Dependent Repulsion)', fontsize=16)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Binding Energy (MeV)', fontsize=12)
    plt.tight_layout()
    plt.savefig('v4_1_heatmap.png', dpi=150, bbox_inches='tight')
    print("✅ Heatmap saved to: v4_1_heatmap.png")
    plt.close()
    
    # 2. Confusion Matrix
    df['predicted_stable'] = (df['delta_E'] > 0) & (~df['delta_E'].isna())
    TP = ((df['empirical_stable'] == True) & (df['predicted_stable'] == True)).sum()
    FN = ((df['empirical_stable'] == True) & (df['predicted_stable'] == False)).sum()
    TN = ((df['empirical_stable'] == False) & (df['predicted_stable'] == False)).sum()
    FP = ((df['empirical_stable'] == False) & (df['predicted_stable'] == True)).sum()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = np.array([[TP, FP], [FN, TN]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Stable', 'Predicted Unstable'],
                yticklabels=['Empirical Stable', 'Empirical Unstable'],
                ax=ax)
    ax.set_title('Confusion Matrix - V4.1 vs Empirical', fontsize=14)
    plt.tight_layout()
    plt.savefig('v4_1_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✅ Confusion matrix saved to: v4_1_confusion_matrix.png")
    plt.close()
    
    # 3. Binding Energy vs A
    fig, ax = plt.subplots(figsize=(12, 6))
    stable = df[df['predicted_stable'] == True]
    unstable = df[df['predicted_stable'] == False]
    emp_stable = df[df['empirical_stable'] == True]
    
    ax.scatter(stable['A'], stable['delta_E'], c='green', alpha=0.6, s=30, label='Predicted Stable')
    ax.scatter(unstable['A'], unstable['delta_E'], c='red', alpha=0.6, s=30, label='Predicted Unstable')
    ax.scatter(emp_stable['A'], emp_stable['delta_E'],
               marker='o', facecolors='none', edgecolors='black', s=80,
               label='Empirical Stable')
    ax.set_xlabel('Mass Number A', fontsize=12)
    ax.set_ylabel('Binding Energy (MeV)', fontsize=12)
    ax.set_title('V4.1 Binding Energy vs Mass Number', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('v4_1_binding_by_A.png', dpi=150, bbox_inches='tight')
    print("✅ Binding by A plot saved to: v4_1_binding_by_A.png")
    plt.close()
    
    # 4. Summary
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    
    print("\n" + "="*80)
    print("V4.1 PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Total isotopes scanned:   {TP + TN + FP + FN}")
    print(f"True Positives (TP):      {TP}")
    print(f"False Positives (FP):     {FP}")
    print(f"True Negatives (TN):      {TN}")
    print(f"False Negatives (FN):     {FN}")
    print("-"*80)
    print(f"Recall (Sensitivity):     {recall*100:.1f}%")
    print(f"Precision:                {precision*100:.1f}%")
    print(f"Specificity:              {specificity*100:.1f}%")
    print(f"Accuracy:                 {accuracy*100:.1f}%")
    print("="*80)
    
    fn_cases = df[(df['empirical_stable'] == True) & (df['predicted_stable'] == False)]
    if not fn_cases.empty:
        print("\n❌ False Negatives (Missed Stable Isotopes):")
        for _, row in fn_cases.iterrows():
            print(f"  Z={int(row['Z'])}, N={int(row['N'])}, A={int(row['A'])}: ΔE={row['delta_E']:.2f} MeV")
    
    fp_cases = df[(df['empirical_stable'] == False) & (df['predicted_stable'] == True)]
    if not fp_cases.empty:
        print(f"\n⚠️ False Positives (Overbound Unstable Isotopes) - Top 10:")
        fp_sorted = fp_cases.sort_values('delta_E', ascending=False)
        for _, row in fp_sorted.head(10).iterrows():
            print(f"  Z={int(row['Z'])}, N={int(row['N'])}, A={int(row['A'])}: ΔE={row['delta_E']:.2f} MeV")
    
    metrics = {'TP': int(TP), 'FP': int(FP), 'TN': int(TN), 'FN': int(FN),
               'recall': float(recall), 'precision': float(precision),
               'specificity': float(specificity), 'accuracy': float(accuracy)}
    with open('v4_1_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\n✅ Metrics saved to: v4_1_metrics.json")

if __name__ == "__main__":
    print("="*80)
    print("RealQM V4.1 Results Analysis")
    print("="*80)
    
    df = load_data()
    if df is not None:
        generate_plots(df)
        print("\n" + "="*80)
        print("Analysis complete!")
        print("="*80)
    
    input("\nPress Enter to close...")