"""
calibrate_gamma_v4_1.py - Calibrate gamma for V4.1 to Carbon-12
==================================================================
Sweeps gamma values to find the one that matches Carbon-12
binding energy (92.16 MeV). Saves the result to a config file.

For V4.1, gamma will be much smaller (~10-100 MeV) because
the A-dependent scaling amplifies the repulsion for larger nuclei.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
import pickle

# Import the V4.1 solver
from NucleonSolverV4_1 import generate_nucleus_geometry, NucleonSolverV4_1, A0_REF, POWER

# ============================================================
# CONFIGURATION
# ============================================================
TARGET_BINDING = 92.16  # MeV (experimental Carbon-12)
GAMMA_MIN = 5.0         # MeV (much smaller than V4)
GAMMA_MAX = 200.0       # MeV
GAMMA_STEPS = 25
N_REPEATS = 3

CHECKPOINT_FILE = 'calibration_checkpoint_v4_1.pkl'

# ============================================================
# CHECKPOINT FUNCTIONS
# ============================================================
def save_checkpoint(gamma_values, binding_values, errors, current_idx):
    checkpoint = {
        'gamma_values': gamma_values,
        'binding_values': binding_values,
        'errors': errors,
        'current_idx': current_idx
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return None

# ============================================================
# CALIBRATION FUNCTION
# ============================================================
def calibrate_gamma():
    print("="*80)
    print("Gamma Calibration for V4.1 Repulsion (with Checkpointing)")
    print("="*80)
    print(f"Target: Carbon-12 binding = {TARGET_BINDING} MeV")
    print(f"Gamma range: {GAMMA_MIN} to {GAMMA_MAX} in {GAMMA_STEPS} steps")
    print(f"Repeats per gamma: {N_REPEATS}")
    print(f"Reference A0: {A0_REF:.0f}, Power: {POWER:.1f}")
    print(f"Checkpoint file: {CHECKPOINT_FILE}")
    print("="*80)
    
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"Found checkpoint — resuming from step {checkpoint['current_idx']+1}/{GAMMA_STEPS}")
        gamma_values = checkpoint['gamma_values']
        binding_values = checkpoint['binding_values']
        errors = checkpoint['errors']
        start_idx = checkpoint['current_idx'] + 1
    else:
        gamma_values = np.linspace(GAMMA_MIN, GAMMA_MAX, GAMMA_STEPS)
        binding_values = []
        errors = []
        start_idx = 0
    
    if len(binding_values) >= GAMMA_STEPS:
        print("All gamma values already computed. Finalizing...")
    else:
        for idx in tqdm(range(start_idx, GAMMA_STEPS), desc="Calibrating gamma", initial=start_idx, total=GAMMA_STEPS):
            gamma = gamma_values[idx]
            bindings = []
            
            for rep in range(N_REPEATS):
                seed = 42 + rep * 100 + idx * 7
                positions, identities = generate_nucleus_geometry(6, 6, seed=seed)
                solver = NucleonSolverV4_1(positions, identities, gamma=gamma)
                solver.solve(verbose=False)
                if solver.success and not np.isnan(solver.delta_E):
                    bindings.append(solver.delta_E)
            
            if bindings:
                avg_binding = np.mean(bindings)
                std_binding = np.std(bindings)
                binding_values.append(avg_binding)
                errors.append(std_binding)
            else:
                binding_values.append(np.nan)
                errors.append(np.nan)
            
            save_checkpoint(gamma_values, binding_values, errors, idx)
            print(f"  Checkpoint saved — step {idx+1}/{GAMMA_STEPS} complete")
    
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    valid_idx = [i for i, b in enumerate(binding_values) if not np.isnan(b)]
    if valid_idx:
        best_idx = min(valid_idx, key=lambda i: abs(binding_values[i] - TARGET_BINDING))
        best_gamma = gamma_values[best_idx]
        best_binding = binding_values[best_idx]
        
        print("\n" + "="*80)
        print("Calibration Complete!")
        print("="*80)
        print(f"Best gamma: {best_gamma:.3f} MeV")
        print(f"Binding energy: {best_binding:.3f} MeV")
        print(f"Error: {abs(best_binding - TARGET_BINDING):.3f} MeV ({100*abs(best_binding - TARGET_BINDING)/TARGET_BINDING:.1f}%)")
        
        config = {
            'gamma': float(best_gamma),
            'target_binding': TARGET_BINDING,
            'achieved_binding': float(best_binding),
            'error_mev': float(abs(best_binding - TARGET_BINDING)),
            'error_percent': float(100*abs(best_binding - TARGET_BINDING)/TARGET_BINDING),
            'A0': A0_REF,
            'power': POWER
        }
        
        with open('gamma_config_v4_1.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nConfig saved to: gamma_config_v4_1.json")
        
        data = {
            'gamma_values': gamma_values.tolist(),
            'binding_values': [float(x) if not np.isnan(x) else None for x in binding_values],
            'errors': [float(x) if not np.isnan(x) else None for x in errors],
            'best_gamma': float(best_gamma),
            'best_binding': float(best_binding)
        }
        with open('calibration_data_v4_1.json', 'w') as f:
            json.dump(data, f, indent=2)
        print("Full data saved to: calibration_data_v4_1.json")
        
        plt.figure(figsize=(10, 6))
        valid_mask = ~np.isnan(binding_values)
        plt.plot(gamma_values[valid_mask], np.array(binding_values)[valid_mask], 'b-', linewidth=2, label='Binding Energy')
        
        valid_errors = [e for e in errors if not np.isnan(e)]
        if valid_errors:
            plt.fill_between(gamma_values[valid_mask], 
                             np.array(binding_values)[valid_mask] - np.array([e if not np.isnan(e) else 0 for e in errors])[valid_mask],
                             np.array(binding_values)[valid_mask] + np.array([e if not np.isnan(e) else 0 for e in errors])[valid_mask],
                             alpha=0.2, color='blue')
        
        plt.axhline(y=TARGET_BINDING, color='r', linestyle='--', label=f'Target: {TARGET_BINDING} MeV')
        plt.axvline(x=best_gamma, color='g', linestyle='--', label=f'Best gamma: {best_gamma:.3f}')
        plt.xlabel('Gamma (MeV)')
        plt.ylabel('Binding Energy (MeV)')
        plt.title('Gamma Calibration for V4.1 (A-Dependent Repulsion)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('gamma_calibration_v4_1.png', dpi=150, bbox_inches='tight')
        print("Plot saved to: gamma_calibration_v4_1.png")
        plt.show(block=False)
        input("\nPress Enter to close...")
        plt.close()
        
        return best_gamma
    else:
        print("ERROR: No valid binding energies found. Check the solver.")
        return None

if __name__ == "__main__":
    calibrate_gamma()
    input("\nPress Enter to close...")