"""
run_stability_scan_v4_1_parallel.py - Full Parallel Stability Scan for V4.1
============================================================================
Uses multiprocessing.Pool with imap_unordered for streaming results.
Results are written to disk in real-time.
"""

import numpy as np
from tqdm import tqdm
import csv
import time
import os
import json
import sys
import multiprocessing as mp

# Import the V4.1 solver
from NucleonSolverV4_1 import generate_nucleus_geometry, NucleonSolverV4_1

# ============================================================
# CONFIGURATION
# ============================================================
Z_MIN = 1
Z_MAX = 20
N_RATIO_MIN = 1.0
N_RATIO_MAX = 3.0
OUTPUT_FILE = 'stability_scan_v4_1_full.csv'
N_JOBS = 16

# Load gamma from calibration
try:
    with open('gamma_config_v4_1.json', 'r') as f:
        config = json.load(f)
        GAMMA = config['gamma']
    print(f"Loaded gamma from config: {GAMMA:.3f} MeV")
except FileNotFoundError:
    print("ERROR: gamma_config_v4_1.json not found. Run calibrate_gamma_v4_1.py first!")
    sys.exit(1)

# ============================================================
# SOLVER FUNCTION
# ============================================================
def solve_single_isotope(Z, N, gamma=GAMMA):
    try:
        positions, identities = generate_nucleus_geometry(Z, N, seed=42 + Z*100 + N)
        solver = NucleonSolverV4_1(positions, identities, gamma=gamma)
        solver.solve(verbose=False)
        
        delta_E = solver.delta_E
        movement = solver.movement_distance
        stable = delta_E > 0 if not np.isnan(delta_E) else False
        boundary_hit = solver.boundary_hit
        success = solver.success
        
        return {
            'Z': Z,
            'N': N,
            'A': Z + N,
            'delta_E': delta_E if not np.isnan(delta_E) else None,
            'movement': movement if not np.isnan(movement) else None,
            'stable': stable,
            'boundary_hit': boundary_hit,
            'success': success,
        }
    except Exception as e:
        return {
            'Z': Z,
            'N': N,
            'A': Z + N,
            'delta_E': None,
            'movement': None,
            'stable': False,
            'boundary_hit': False,
            'success': False,
            'error': str(e),
        }

def solver_wrapper(args):
    Z, N, gamma = args
    return solve_single_isotope(Z, N, gamma)

# ============================================================
# MAIN SCAN FUNCTION
# ============================================================
def run_full_scan_parallel():
    print("="*80)
    print("RealQM V4.1 Full Stability Scan (Parallel - multiprocessing)")
    print("="*80)
    print(f"Gamma: {GAMMA:.3f} MeV")
    print(f"A0: {config.get('A0', 12.0):.0f}, Power: {config.get('power', 1.5):.1f}")
    print(f"Scanning: Z = {Z_MIN} to {Z_MAX}, N = Z to {N_RATIO_MAX:.1f}Z")
    print(f"Parallel jobs: {N_JOBS} cores")
    print(f"Output file: {OUTPUT_FILE}")
    print("="*80)
    
    scan_list = []
    for Z in range(Z_MIN, Z_MAX + 1):
        N_min = int(np.ceil(Z * N_RATIO_MIN))
        N_max = int(np.floor(Z * N_RATIO_MAX))
        for N in range(N_min, N_max + 1):
            scan_list.append((Z, N, GAMMA))
    
    total_isotopes = len(scan_list)
    print(f"Total isotopes to compute: {total_isotopes}")
    print("="*80)
    
    fieldnames = ['Z', 'N', 'A', 'delta_E', 'movement', 'stable', 'boundary_hit', 'success']
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    start_time = time.time()
    
    print("\nStarting parallel scan... (results will stream to disk in real-time)")
    print("-"*80)
    
    with mp.Pool(processes=N_JOBS) as pool:
        results = pool.imap_unordered(solver_wrapper, scan_list)
        
        with tqdm(total=total_isotopes, desc="Processing & Saving Isotopes", 
                  mininterval=0.5, maxinterval=1.0) as pbar:
            for result in results:
                row = {
                    'Z': result['Z'],
                    'N': result['N'],
                    'A': result['A'],
                    'delta_E': result['delta_E'] if result['delta_E'] is not None else np.nan,
                    'movement': result['movement'] if result['movement'] is not None else np.nan,
                    'stable': result['stable'],
                    'boundary_hit': result['boundary_hit'],
                    'success': result['success'],
                }
                
                with open(OUTPUT_FILE, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(row)
                
                pbar.update(1)
                
                if pbar.n % 10 == 0:
                    sys.stdout.flush()
    
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print(f"Scan Complete in {elapsed:.2f} seconds!")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("="*80)

if __name__ == "__main__":
    run_full_scan_parallel()
    input("\nPress Enter to close...")