"""
NucleonSolverV4_1.py - RealQM Nuclear Engine with A-Dependent Repulsion
========================================================================
Version 4.1 introduces A-dependent repulsion:
- U_repulsion_MeV = gamma * (A/A0)^power * (V0/(V_local + V0))^2
- gamma is calibrated to Carbon-12 (re-calibrated for V4.1)
- A0 = 12 (reference nucleus)
- power = 1.5 (controls how rapidly repulsion grows with A)

All V4 physics preserved:
- Field-based coherence saturation
- Mutual inductance via Neumann integrals
- Convex Hull bounding volume
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import csv
import sys
import traceback
from tqdm import tqdm
import os
import time

# ============================================================
# 1. PHYSICAL CONSTANTS
# ============================================================
e = 1.602176634e-19
m_p = 1.67262192369e-27
m_n = 1.67492749804e-27
c = 299792458.0
h = 6.62607015e-34
hbar = h / (2 * np.pi)
mu0 = 4e-7 * np.pi
epsilon0 = 1 / (mu0 * c**2)
MeV = 1.602176634e-13
fm = 1e-15

# ============================================================
# 2. LOOP GEOMETRY
# ============================================================
R_LOOP = 0.8415 * fm                # Loop radius
R_WIRE = 0.1 * fm                   # Wire thickness
E_SATURATION = 1.0 / R_LOOP**3

# ============================================================
# 3. CALIBRATED PARAMETERS (from V3 multi-nucleus calibration)
# ============================================================
ALPHA_SCALE = 0.850
REPULSION_STRENGTH = 1.30  # MeV (will be replaced by gamma in V4)
EMAX = 5.0
NEUTRON_E0 = 0.5
COMPACTION = 0.9

# ============================================================
# 4. V4.1 REPULSION PARAMETERS
# ============================================================
A0_REF = 12.0          # Reference nucleon number (Carbon-12)
POWER = 1.5            # A-scaling exponent

# ============================================================
# 5. FIELD STRENGTH & COHERENCE (V3 physics, unchanged)
# ============================================================
def field_strength(r, centers, coupling_order=3):
    """Compute field strength at position r."""
    E_total = 0.0
    for center in centers:
        if np.linalg.norm(r - center) < 1e-30:
            continue
        dist = np.linalg.norm(r - center)
        r_scaled = dist / fm
        field = 1.0 / (r_scaled ** coupling_order + 1e-30)
        E_total += field
    return E_total

def coherence_fraction_v2_2(E, eta_0, E0, Emax=EMAX, E_saturation=None):
    """Coherence with field-based saturation (Emax=5.0)."""
    if E_saturation is None:
        E_saturation = E_SATURATION
    base = 1 - (1 - eta_0) * np.exp(-E / E0)
    roll_off = np.exp(-(E / Emax) ** 2)
    base = base * roll_off
    saturation = np.exp(-(E / E_saturation) ** 2)
    return base * saturation

# ============================================================
# 6. INDUCTANCE & ENERGY FUNCTIONS (V3, unchanged)
# ============================================================
def generate_loop_points(center, tilt, yaw, radius=R_LOOP, steps=12):
    t = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    base_points = np.zeros((steps, 3))
    base_points[:, 0] = radius * np.cos(t)
    base_points[:, 1] = radius * np.sin(t)
    cos_t, sin_t = np.cos(tilt), np.sin(tilt)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R_tilt = np.array([[1.0, 0.0, 0.0], [0.0, cos_t, -sin_t], [0.0, sin_t, cos_t]])
    R_yaw = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
    return np.dot(base_points, np.dot(R_yaw, R_tilt).T) + center

def calculate_mutual_inductance(loop1, loop2):
    n = len(loop1)
    dl1 = np.zeros((n, 3))
    dl1[:-1] = loop1[1:] - loop1[:-1]
    dl1[-1] = loop1[0] - loop1[-1]
    dl2 = np.zeros((n, 3))
    dl2[:-1] = loop2[1:] - loop2[:-1]
    dl2[-1] = loop2[0] - loop2[-1]
    diff = loop1[:, np.newaxis, :] - loop2[np.newaxis, :, :]
    r12 = np.linalg.norm(diff, axis=2)
    core_buffer = 0.1e-15
    r12 = np.where(r12 < core_buffer, core_buffer, r12)
    M = -(mu0 / (4 * np.pi)) * np.sum(np.sum(dl1[:, np.newaxis, :] * dl2[np.newaxis, :, :], axis=2) / r12)
    return M

def calculate_self_inductance(loop, radius, q, I):
    a = 0.1e-15
    L = mu0 * radius * (np.log(8 * radius / a) - 2)
    return L

def calculate_coulomb_energy(q1, q2, r):
    if r == 0:
        return 1e6
    return (1 / (4 * np.pi * epsilon0)) * q1 * q2 / r

# ============================================================
# 7. V4.1 REPULSION MODULE (A-DEPENDENT)
# ============================================================
def compute_bounding_volume(centers, R_LOOP, use_convex_hull=True):
    """
    Compute the bounding volume using a Convex Hull.
    Falls back to a sphere for A < 4.
    """
    A = len(centers)
    
    if A < 4 or not use_convex_hull:
        com = np.mean(centers, axis=0)
        r_max = np.max([np.linalg.norm(c - com) for c in centers]) + R_LOOP
        return (4/3) * np.pi * r_max**3
    
    try:
        centers_fm = centers / fm
        hull = ConvexHull(centers_fm)
        return hull.volume * (fm**3)
    except Exception as e:
        com = np.mean(centers, axis=0)
        r_max = np.max([np.linalg.norm(c - com) for c in centers]) + R_LOOP
        return (4/3) * np.pi * r_max**3

def calculate_repulsion_v4_1(centers, identities, loops, currents, gamma=1.0, A0=A0_REF, power=POWER):
    """
    V4.1 density-dependent repulsion with A-dependent scaling.
    
    Physics:
    - U_repulsion_MeV = gamma * (A/A0)^power * (V0/(V_local + V0))^2
    - gamma is calibrated to Carbon-12 (re-calibrated for V4.1)
    - A0 = 12 (reference nucleus)
    - power = 1.5 (controls how rapidly repulsion grows with A)
    
    Why this works:
    - The A-dependent scaling ensures repulsion grows with nucleon number
    - Counteracts the cumulative magnetic attraction of many loops
    - Preserves the dimensionless ratio for numerical stability
    """
    A = len(centers)
    
    # 1. Compute local bounding volume (convert to fm³)
    V_local_m3 = compute_bounding_volume(centers, R_LOOP, use_convex_hull=True)
    V_local_fm3 = V_local_m3 / (fm**3)
    
    # 2. Characteristic volume (nucleon volume scale) in fm³
    R_LOOP_fm = R_LOOP / fm
    V0_fm3 = (4/3) * np.pi * R_LOOP_fm**3  # ≈ 2.495 fm³
    
    # 3. Dimensionless ratio (0 to 1)
    ratio = V0_fm3 / (V_local_fm3 + V0_fm3)
    
    # 4. A-dependent scaling: repulsion grows with nucleon number
    A_scaling = (A / A0) ** power
    
    # 5. Repulsion in MeV
    U_repulsion_MeV = gamma * A_scaling * ratio**2
    
    # 6. Safety cap to prevent unphysical values
    max_repulsion_mev = 5000.0  # 5 GeV cap
    if U_repulsion_MeV > max_repulsion_mev:
        U_repulsion_MeV = max_repulsion_mev
    
    # 7. Convert to Joules
    return U_repulsion_MeV * MeV

# ============================================================
# 8. NUCLEON SOLVER V4.1
# ============================================================
class NucleonSolverV4_1:
    def __init__(self, positions, identities, gamma=1.0, A0=A0_REF, power=POWER):
        self.centers_initial = positions
        self.identities = identities
        self.n_loops = len(positions)
        self.n_params = 5 * self.n_loops
        self.Z = np.sum(identities == 0)
        self.N = np.sum(identities == 1)
        self.gamma = gamma
        self.A0 = A0
        self.power = power
        self.I_p = e * (m_p * c**2 / h)
        self.I_n_base = e * (m_n * c**2 / h)
        
        # Calibrated parameters (from V3)
        self.alpha_scale = ALPHA_SCALE
        self.neutron_eta_0 = 0.676
        self.neutron_E0 = NEUTRON_E0
        self.proton_eta_0 = 1.0
        self.proton_E0 = 1.5
        self.Emax = EMAX
        self.E_saturation = E_SATURATION
        
        self.U0 = None
        self.U_min = None
        self.delta_E = None
        self.free_energy = None
        self.final_centers = None
        self.movement_distance = None
        self.mag_attractive_energy = 0.0
        self.repulsive_energy = 0.0
        self.boundary_hit = False
        self.success = False

    def compute_eta_for_nucleon(self, idx, centers):
        center = centers[idx]
        E = field_strength(center, centers)
        if self.identities[idx] == 0:
            return coherence_fraction_v2_2(E, self.proton_eta_0, self.proton_E0, self.Emax, self.E_saturation)
        else:
            return coherence_fraction_v2_2(E, self.neutron_eta_0, self.neutron_E0, self.Emax, self.E_saturation)

    def compute_currents(self, centers):
        currents = []
        for idx in range(self.n_loops):
            eta = self.compute_eta_for_nucleon(idx, centers)
            if self.identities[idx] == 0:
                currents.append(self.I_p * eta)
            else:
                currents.append(self.I_n_base * eta)
        return currents

    def pack_params(self, centers, angles):
        params = np.zeros(5 * self.n_loops)
        for i in range(self.n_loops):
            params[5*i : 5*i + 3] = centers[i] / fm
            params[5*i + 3 : 5*i + 5] = angles[i]
        return params

    def unpack_params(self, params):
        centers = np.zeros((self.n_loops, 3))
        angles = np.zeros((self.n_loops, 2))
        for i in range(self.n_loops):
            centers[i] = params[5*i : 5*i + 3] * fm
            angles[i] = params[5*i + 3 : 5*i + 5]
        return centers, angles

    def calculate_energy(self, params):
        centers, angles = self.unpack_params(params)
        loops = []
        for i in range(self.n_loops):
            loops.append(generate_loop_points(centers[i], angles[i, 0], angles[i, 1], steps=12))
        currents = self.compute_currents(centers)
        charges = [e if self.identities[i] == 0 else 0.0 for i in range(self.n_loops)]
        
        # Magnetic energy (V3 physics)
        mag_energy = 0.0
        for i in range(self.n_loops):
            L_ii = calculate_self_inductance(loops[i], R_LOOP, charges[i], currents[i])
            mag_energy -= 0.5 * L_ii * currents[i]**2
        for i in range(self.n_loops):
            for j in range(i + 1, self.n_loops):
                M_ij = calculate_mutual_inductance(loops[i], loops[j])
                mag_energy += M_ij * currents[i] * currents[j]
        mag_energy_scaled = self.alpha_scale * mag_energy
        
        # Coulomb energy (V3 physics)
        coulomb_energy = 0.0
        for i in range(self.n_loops):
            for j in range(i + 1, self.n_loops):
                if self.identities[i] == 0 and self.identities[j] == 0:
                    r = np.linalg.norm(centers[i] - centers[j])
                    coulomb_energy += calculate_coulomb_energy(e, e, r)
        
        # Kinetic energy (V3 physics)
        kinetic_energy = 0.0
        for i in range(self.n_loops):
            if self.identities[i] == 0:
                kinetic_energy += 0.5 * m_p * c**2
            else:
                kinetic_energy += 0.5 * m_n * c**2
        
        # V4.1 REPULSION (A-dependent)
        repulsion_energy = calculate_repulsion_v4_1(centers, self.identities, loops, currents, self.gamma, self.A0, self.power)
        
        total_energy = mag_energy_scaled + coulomb_energy + kinetic_energy + repulsion_energy
        return total_energy / MeV

    def solve(self, verbose=False):
        initial_angles = np.zeros((self.n_loops, 2))
        initial_params = self.pack_params(self.centers_initial, initial_angles)
        
        half_range = max(5.0, 3.0 * 1.2 * (self.n_loops ** (1.0/3.0)))
        angle_range = np.pi / 2
        bounds = []
        for i in range(self.n_loops):
            bounds.extend([
                (self.centers_initial[i, 0]/fm - half_range, self.centers_initial[i, 0]/fm + half_range),
                (self.centers_initial[i, 1]/fm - half_range, self.centers_initial[i, 1]/fm + half_range),
                (self.centers_initial[i, 2]/fm - half_range, self.centers_initial[i, 2]/fm + half_range)
            ])
            bounds.extend([
                (-angle_range, angle_range),
                (-angle_range, angle_range)
            ])
        
        np.random.seed(42)
        initial_params += np.random.uniform(-0.01, 0.01, len(initial_params))
        
        try:
            result = minimize(
                self.calculate_energy,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-8}
            )
            
            if not result.success:
                if verbose:
                    print(f"  Warning: Optimiser did not converge (message: {result.message})")
                self.success = False
            else:
                self.success = True
            
            free_energy = 0.0
            for i in range(self.n_loops):
                if self.identities[i] == 0:
                    free_energy += 0.5 * m_p * c**2
                else:
                    free_energy += 0.5 * m_n * c**2
            free_energy = free_energy / MeV
            
            self.U_min = result.fun
            self.free_energy = free_energy
            self.delta_E = free_energy - result.fun
            
            self.final_centers, self.final_angles = self.unpack_params(result.x)
            distances = np.linalg.norm(self.final_centers - self.centers_initial, axis=1)
            self.movement_distance = np.mean(distances) / fm
            
            # Boundary hit detection
            tolerance = 0.01 * half_range
            hit = False
            for i in range(self.n_loops):
                for axis in range(3):
                    lower = self.centers_initial[i, axis]/fm - half_range
                    upper = self.centers_initial[i, axis]/fm + half_range
                    val = result.x[5*i + axis]
                    if (val - lower < tolerance) or (upper - val < tolerance):
                        hit = True
                        break
                if hit:
                    break
            self.boundary_hit = hit
            
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            self.success = False
            self.delta_E = np.nan
            self.movement_distance = np.nan
            self.boundary_hit = False
        
        return self

# ============================================================
# 9. GEOMETRY GENERATOR (V3, unchanged)
# ============================================================
def generate_nucleus_geometry(Z, N, seed=42, compaction=COMPACTION):
    A = Z + N
    R0 = compaction * 1.2 * (A ** (1.0/3.0)) * fm
    np.random.seed(seed)
    positions = []
    identities = []
    for i in range(A):
        theta = np.arccos(2 * np.random.rand() - 1)
        phi = 2 * np.pi * np.random.rand()
        r = R0 * np.random.rand() ** (1.0/3.0)
        pos = np.array([r * np.sin(theta) * np.cos(phi),
                        r * np.sin(theta) * np.sin(phi),
                        r * np.cos(theta)])
        positions.append(pos)
        identities.append(0 if i < Z else 1)
    positions = np.array(positions)
    identities = np.array(identities)
    com = np.mean(positions, axis=0)
    positions -= com
    return positions, identities

# ============================================================
# 10. TEST FUNCTION
# ============================================================
def test_v4_1_on_carbon(gamma=1.0, verbose=True):
    """Test V4.1 on Carbon-12 as a quick validation."""
    print(f"\nTesting V4.1 on Carbon-12 (Z=6, N=6) with gamma={gamma:.3f}...")
    positions, identities = generate_nucleus_geometry(6, 6, seed=42)
    solver = NucleonSolverV4_1(positions, identities, gamma=gamma)
    solver.solve(verbose=verbose)
    
    if solver.success:
        print(f"  Binding Energy: {solver.delta_E:.3f} MeV")
        print(f"  Target (experimental): 92.16 MeV")
        print(f"  Error: {abs(solver.delta_E - 92.16):.3f} MeV ({100*abs(solver.delta_E - 92.16)/92.16:.1f}%)")
    else:
        print("  Optimisation failed!")
    
    return solver

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("="*80)
    print("RealQM NucleonSolver V4.1 - A-Dependent Repulsion")
    print("="*80)
    print(f"Loop radius (R_LOOP): {R_LOOP/fm:.4f} fm")
    print(f"Nucleon volume scale (V0): {(4/3)*np.pi*(R_LOOP/fm)**3:.4f} fm^3")
    print(f"Reference A0: {A0_REF:.0f}")
    print(f"Power: {POWER:.1f}")
    print("Repulsion law: U_MeV = gamma * (A/A0)^power * (V0/(V_local + V0))^2")
    print("="*80)
    
    # Quick test with gamma=1.0
    test_v4_1_on_carbon(gamma=1.0, verbose=True)
    
    input("\nPress Enter to close...")