#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
*** Final FAST Comparison Script (PHYSICS vs. NATIVE ML) ***

(UPDATED for 4-layer model)

This script runs the comparison using the full physics solver
vs. a NATIVE CuPy implementation of the trained ML model.
"""

import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt
from pathlib import Path

print("Running NATIVE CuPy comparison script (for 4-layer model).")


# ============================================================================
# 1. Constants and Parameters
# ============================================================================
# Physical Constants
PI = 2.0 * np.arcsin(1.0)
K_B = 1.380e-23  # Boltzmann constant
MASS_AR = 66.3e-27  # Argon mass
VIS0 = 2.117e-5  # Reference viscosity
VISP = 1.0  # Viscosity power (Maxwell molecules)
RATSH = 5.0 / 3.0  # Specific heat ratio
RHO_IN_BASE_GLOBAL = (266.644 / 10.0) * MASS_AR / K_B / 273.15  # (Kn ~ 0.15)
T_IN_BASE_GLOBAL = 273.15  # Reference temperature
UW1_BASE = -50.0  # Left wall velocity (in Y-direction)
UW2_BASE = 50.0  # Right wall velocity (in Y-direction)
TW1_BASE = 273.15  # Left wall temperature
TW2_BASE = 273.15  # Right wall temperature
THETAW1_GLOBAL = np.sqrt(K_B * TW1_BASE / MASS_AR)
THETAW2_GLOBAL = np.sqrt(K_B * TW2_BASE / MASS_AR)
LX = 0.001  
NX = 300  
NC = NX  
PARTICLES_PER_CELL_TARGET = 30000 
NP = PARTICLES_PER_CELL_TARGET * NC  
NTSS = 1000  # Steps to reach steady state (for averaging)
N_SIMULATION_RUNS = 1  # retained for compatibility; profiles use N_REPLICAS below
N_STEPS_PER_RUN = 40000      # total steps per replica
AVG_START_STEP = 5000        # start time averaging after this step
N_REPLICAS = 3               # increase to 5 for even cleaner figures if time permits
BASE_RANDOM_SEED = 24680
SMOOTH_WINDOW = 9            # odd integer; mild smoothing for publication-quality curves
ENFORCE_SYMMETRY = True      # Couette symmetry: Uy antisymmetric, T and rho symmetric
FIG_DIR = Path("couette_figures_pdf")
DATA_DIR = Path("couette_profile_data")
SAVE_PNG_COPY = False
FIG_DPI = 300
EPSILON = 1e-30 

# ============================================================================
# 2. Utility and Initialization Functions
# ============================================================================
def rfn_gpu(n_samples):
    return cp.random.normal(0.0, 1.0, n_samples)

def initialize_particles_cupy(np_val, lx_val, theta0_val, w_val):
    p_x = lx_val * cp.random.rand(np_val)
    p_y = cp.zeros(np_val, dtype=cp.float64)
    p_z = cp.zeros(np_val, dtype=cp.float64)
    p_vx = theta0_val * rfn_gpu(np_val)
    p_vy = theta0_val * rfn_gpu(np_val)
    p_vz = theta0_val * rfn_gpu(np_val)
    p_vp_x = p_vx.copy(); p_vp_y = p_vy.copy(); p_vp_z = p_vz.copy()
    p_xl = cp.zeros(np_val, dtype=cp.float64)
    p_yl = cp.zeros(np_val, dtype=cp.float64)
    p_zl = cp.zeros(np_val, dtype=cp.float64)
    p_weight = cp.full(np_val, w_val, dtype=cp.float64)
    p_ind = cp.zeros(np_val, dtype=cp.int32) 
    return (
        p_x, p_y, p_z, p_vx, p_vy, p_vz, p_vp_x, p_vp_y, p_vp_z,
        p_xl, p_yl, p_zl, p_weight, p_ind
    )

def initialize_grid_cupy(nx_val, lx_val, current_rho, current_t):
    cell_width = lx_val / float(nx_val)
    cell_vol = cell_width * 1.0 
    grid_gpu = {
        'pos':   cp.array([(float(i) + 0.5) * cell_width for i in range(nx_val)]),
        'vol':   cp.full(nx_val, cell_vol, dtype=cp.float64),
        'N':     cp.zeros(nx_val, dtype=cp.float64),
        'rho':   cp.full(nx_val, current_rho, dtype=cp.float64),
        'T':     cp.full(nx_val, current_t, dtype=cp.float64),
        'U':     cp.zeros((nx_val, 3), dtype=cp.float64),
        'PIJ':   cp.zeros((nx_val, 6), dtype=cp.float64),
        'Q':     cp.zeros((nx_val, 3), dtype=cp.float64),
        'M3':    cp.zeros((nx_val, 10), dtype=cp.float64),
        'M4':    cp.zeros((nx_val, 6), dtype=cp.float64),
        'M5':    cp.zeros((nx_val, 3), dtype=cp.float64),
        'DM2':   cp.zeros(nx_val, dtype=cp.float64),
        'DM4':   cp.zeros(nx_val, dtype=cp.float64),
        'nu':    cp.zeros(nx_val, dtype=cp.float64),
        'nubol': cp.zeros(nx_val, dtype=cp.float64),
        'lam':   cp.zeros(nx_val, dtype=cp.float64),
        'Diff':  cp.zeros(nx_val, dtype=cp.float64),
    }
    coeffs_gpu = {
        'A': cp.zeros((nx_val, 6), dtype=cp.float64),
        'B': cp.zeros((nx_val, 3), dtype=cp.float64),
        'C': cp.zeros(nx_val, dtype=cp.float64),
    }
    linsys_gpu = {
        'lhs': cp.zeros((nx_val, 9, 9), dtype=cp.float64),
        'rhs': cp.zeros((nx_val, 9), dtype=cp.float64),
    }
    return grid_gpu, coeffs_gpu, linsys_gpu 

# ============================================================================
# 3. Core Simulation Functions
# ============================================================================
def apply_boundary_couette_cupy(p_data, lx_val, dt_val, uw1, uw2, thetaw1, thetaw2):
    p_x, p_vx, p_vy, p_vz = p_data[0], p_data[3], p_data[4], p_data[5]
    idx_right = cp.where(p_x > lx_val)[0]; n_right = len(idx_right)
    if n_right > 0:
        seed1 = cp.maximum(cp.random.rand(n_right), EPSILON); xi = rfn_gpu((n_right, 2))
        p_vx[idx_right] = -thetaw2 * cp.sqrt(-2.0 * cp.log(seed1))
        p_vy[idx_right] = thetaw2 * xi[:, 0] + uw2
        p_vz[idx_right] = thetaw2 * xi[:, 1]
        p_x[idx_right] = lx_val + p_vx[idx_right] * dt_val * cp.random.rand(n_right)
    idx_left = cp.where(p_x < 0.0)[0]; n_left = len(idx_left)
    if n_left > 0:
        seed1 = cp.maximum(cp.random.rand(n_left), EPSILON); xi = rfn_gpu((n_left, 2))
        p_vx[idx_left] = thetaw1 * cp.sqrt(-2.0 * cp.log(seed1))
        p_vy[idx_left] = thetaw1 * xi[:, 0] + uw1
        p_vz[idx_left] = thetaw1 * xi[:, 1]
        p_x[idx_left] = p_vx[idx_left] * dt_val * cp.random.rand(n_left)

def sort_and_calc_moments_cupy_LITE(p_data, grid, nc_val, lx_val):
    p_x, p_vx, p_vy, p_vz = p_data[0], p_data[3], p_data[4], p_data[5]
    p_vp_x, p_vp_y, p_vp_z = p_data[6], p_data[7], p_data[8]
    p_weight, p_ind = p_data[12], p_data[13]
    cell_width = lx_val / float(nc_val)
    cell_indices = cp.clip(cp.floor(p_x / cell_width).astype(cp.int32), 0, nc_val - 1)
    p_ind[:] = cell_indices
    grid['N'] = cp.bincount(cell_indices, weights=p_weight, minlength=nc_val)
    N_safe = cp.maximum(grid['N'], EPSILON)
    vol_safe = cp.maximum(grid['vol'], EPSILON)
    grid['rho'] = grid['N'] / vol_safe
    grid['U'][:, 0] = cp.bincount(cell_indices, weights=p_vx * p_weight, minlength=nc_val) / N_safe
    grid['U'][:, 1] = cp.bincount(cell_indices, weights=p_vy * p_weight, minlength=nc_val) / N_safe
    grid['U'][:, 2] = cp.bincount(cell_indices, weights=p_vz * p_weight, minlength=nc_val) / N_safe
    U_particles = grid['U'][cell_indices]
    p_vp_x[:] = p_vx - U_particles[:, 0]
    p_vp_y[:] = p_vy - U_particles[:, 1]
    p_vp_z[:] = p_vz - U_particles[:, 2]
    vp_x = p_vp_x; vp_y = p_vp_y; vp_z = p_vp_z
    vp_sq = vp_x**2 + vp_y**2 + vp_z**2
    grid['DM2'] = cp.bincount(cell_indices, weights=vp_sq * p_weight, minlength=nc_val) / N_safe
    grid['T'] = MASS_AR * grid['DM2'] / (3.0 * K_B); grid['T'] = cp.maximum(grid['T'], 1.0)
    grid['PIJ'][:, 0] = cp.bincount(cell_indices, weights=vp_x*vp_x*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 1] = cp.bincount(cell_indices, weights=vp_x*vp_y*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 2] = cp.bincount(cell_indices, weights=vp_x*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 3] = cp.bincount(cell_indices, weights=vp_y*vp_y*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 4] = cp.bincount(cell_indices, weights=vp_y*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 5] = cp.bincount(cell_indices, weights=vp_z*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['Q'][:, 0] = cp.bincount(cell_indices, weights=vp_x*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['Q'][:, 1] = cp.bincount(cell_indices, weights=vp_y*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['Q'][:, 2] = cp.bincount(cell_indices, weights=vp_z*vp_sq*p_weight, minlength=nc_val) / N_safe
    number_density = grid['rho'] / MASS_AR
    number_density = cp.maximum(number_density, EPSILON)
    P = number_density * K_B * grid['T']
    vis = cp.maximum(VIS0 * (grid['T'] / 273.15)**VISP, EPSILON)
    grid['nubol'] = P / vis
    grid['nu'] = grid['nubol'] * 0.5
    grid['Diff'] = 2.0 * K_B * grid['nu'] * grid['T'] / MASS_AR

def sort_and_calc_moments_cupy_FULL(p_data, grid, nc_val, lx_val):
    p_x, p_vx, p_vy, p_vz = p_data[0], p_data[3], p_data[4], p_data[5]
    p_vp_x, p_vp_y, p_vp_z = p_data[6], p_data[7], p_data[8]
    p_weight, p_ind = p_data[12], p_data[13]
    cell_width = lx_val / float(nc_val)
    cell_indices = cp.clip(cp.floor(p_x / cell_width).astype(cp.int32), 0, nc_val - 1)
    p_ind[:] = cell_indices
    grid['N'] = cp.bincount(cell_indices, weights=p_weight, minlength=nc_val)
    N_safe = cp.maximum(grid['N'], EPSILON)
    vol_safe = cp.maximum(grid['vol'], EPSILON)
    grid['rho'] = grid['N'] / vol_safe
    grid['U'][:, 0] = cp.bincount(cell_indices, weights=p_vx * p_weight, minlength=nc_val) / N_safe
    grid['U'][:, 1] = cp.bincount(cell_indices, weights=p_vy * p_weight, minlength=nc_val) / N_safe
    grid['U'][:, 2] = cp.bincount(cell_indices, weights=p_vz * p_weight, minlength=nc_val) / N_safe
    U_particles = grid['U'][cell_indices]
    p_vp_x[:] = p_vx - U_particles[:, 0]
    p_vp_y[:] = p_vy - U_particles[:, 1]
    p_vp_z[:] = p_vz - U_particles[:, 2]
    vp_x = p_vp_x; vp_y = p_vp_y; vp_z = p_vp_z
    vp_sq = vp_x**2 + vp_y**2 + vp_z**2
    grid['DM2'] = cp.bincount(cell_indices, weights=vp_sq * p_weight, minlength=nc_val) / N_safe
    grid['T'] = MASS_AR * grid['DM2'] / (3.0 * K_B); grid['T'] = cp.maximum(grid['T'], 1.0)
    grid['PIJ'][:, 0] = cp.bincount(cell_indices, weights=vp_x*vp_x*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 1] = cp.bincount(cell_indices, weights=vp_x*vp_y*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 2] = cp.bincount(cell_indices, weights=vp_x*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 3] = cp.bincount(cell_indices, weights=vp_y*vp_y*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 4] = cp.bincount(cell_indices, weights=vp_y*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['PIJ'][:, 5] = cp.bincount(cell_indices, weights=vp_z*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['Q'][:, 0] = cp.bincount(cell_indices, weights=vp_x*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['Q'][:, 1] = cp.bincount(cell_indices, weights=vp_y*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['Q'][:, 2] = cp.bincount(cell_indices, weights=vp_z*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 0] = cp.bincount(cell_indices, weights=vp_x*vp_x*vp_x*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 1] = cp.bincount(cell_indices, weights=vp_x*vp_x*vp_y*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 2] = cp.bincount(cell_indices, weights=vp_x*vp_x*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 3] = cp.bincount(cell_indices, weights=vp_x*vp_y*vp_y*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 4] = cp.bincount(cell_indices, weights=vp_x*vp_y*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 5] = cp.bincount(cell_indices, weights=vp_x*vp_z*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 6] = cp.bincount(cell_indices, weights=vp_y*vp_y*vp_y*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 7] = cp.bincount(cell_indices, weights=vp_y*vp_y*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 8] = cp.bincount(cell_indices, weights=vp_y*vp_z*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['M3'][:, 9] = cp.bincount(cell_indices, weights=vp_z*vp_z*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['M4'][:, 0] = cp.bincount(cell_indices, weights=(vp_x*vp_x)*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['M4'][:, 1] = cp.bincount(cell_indices, weights=(vp_x*vp_y)*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['M4'][:, 2] = cp.bincount(cell_indices, weights=(vp_x*vp_z)*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['M4'][:, 3] = cp.bincount(cell_indices, weights=(vp_y*vp_y)*vp_sq*p_weight, minlength=nc_val) / N_safe
    grid['M4'][:, 4] = cp.bincount(cell_indices, weights=vp_y*vp_z*p_weight, minlength=nc_val) / N_safe
    grid['M4'][:, 5] = cp.bincount(cell_indices, weights=(vp_z*vp_z)*vp_sq*p_weight, minlength=nc_val) / N_safe
    vp_sq_sq = vp_sq * vp_sq
    grid['M5'][:, 0] = cp.bincount(cell_indices, weights=vp_x*vp_sq_sq*p_weight, minlength=nc_val) / N_safe
    grid['M5'][:, 1] = cp.bincount(cell_indices, weights=vp_y*vp_sq_sq*p_weight, minlength=nc_val) / N_safe
    grid['M5'][:, 2] = cp.bincount(cell_indices, weights=vp_z*vp_sq_sq*p_weight, minlength=nc_val) / N_safe
    grid['DM4'] = grid['M4'][:, 0] + grid['M4'][:, 3] + grid['M4'][:, 5]
    number_density = grid['rho'] / MASS_AR
    number_density = cp.maximum(number_density, EPSILON)
    P = number_density * K_B * grid['T']
    vis = cp.maximum(VIS0 * (grid['T'] / 273.15)**VISP, EPSILON)
    grid['nubol'] = P / vis
    grid['nu'] = grid['nubol'] * 0.5
    grid['Diff'] = 2.0 * K_B * grid['nu'] * grid['T'] / MASS_AR
    PIJdev_1 = grid['PIJ'][:, 0] - (1./3.) * grid['DM2']; PIJdev_4 = grid['PIJ'][:, 3] - (1./3.) * grid['DM2']; PIJdev_6 = grid['PIJ'][:, 5] - (1./3.) * grid['DM2']
    lam_sq = (PIJdev_1**2 + PIJdev_4**2 + PIJdev_6**2 + 2.0 * (grid['PIJ'][:, 1]**2 + grid['PIJ'][:, 2]**2 + grid['PIJ'][:, 4]**2))
    DM2_pow_3p5_safe = cp.maximum(grid['DM2']**3.5, EPSILON)
    grid['lam'] = -lam_sq * grid['nu'] / DM2_pow_3p5_safe

def evolve_velocities_cupy(p_data, grid, coeffs, dt_val, nc_val):
    p_vx, p_vy, p_vz = p_data[3], p_data[4], p_data[5]
    p_vp_x, p_vp_y, p_vp_z = p_data[6], p_data[7], p_data[8]
    p_weight, p_ind = p_data[12], p_data[13]
    n_particles=len(p_vx); cell_indices=p_ind
    p_U = grid['U'][cell_indices]
    p_DM2 = grid['DM2'][cell_indices]
    p_Q = grid['Q'][cell_indices]
    p_nu = grid['nu'][cell_indices]
    p_Diff = grid['Diff'][cell_indices]
    p_A = coeffs['A'][cell_indices]
    p_B = coeffs['B'][cell_indices]
    p_C = coeffs['C'][cell_indices] 
    gforce=cp.zeros((n_particles,3)); DM2_safe=p_DM2+EPSILON; Q_safe=p_Q+EPSILON
    gforce[:,0]=p_B[:,0]*(-DM2_safe)+p_C*(-Q_safe[:,0]); gforce[:,1]=p_B[:,1]*(-DM2_safe)+p_C*(-Q_safe[:,1]); gforce[:,2]=p_B[:,2]*(-DM2_safe)+p_C*(-Q_safe[:,2])
    nu_safe=cp.maximum(p_nu,EPSILON); s=nu_safe*dt_val; A1=cp.exp(-s)
    taus=1.0/nu_safe; A2=taus*(1.0-A1)
    diff_term=cp.abs(taus * p_Diff * (1.0 - cp.exp(-2.*s))/2.0)
    A3=cp.sqrt(diff_term)
    vp_x=p_vp_x; vp_y=p_vp_y; vp_z=p_vp_z
    vp_sq=vp_x**2+vp_y**2+vp_z**2; vp_sq_safe=vp_sq+EPSILON
    nvA=cp.zeros((n_particles,3))
    nvA[:,0]=p_A[:,0]*vp_x+p_A[:,1]*vp_y+p_A[:,2]*vp_z; nvA[:,1]=p_A[:,1]*vp_x+p_A[:,3]*vp_y+p_A[:,4]*vp_z; nvA[:,2]=p_A[:,2]*vp_x+p_A[:,4]*vp_y+p_A[:,5]*vp_z
    nq=cp.zeros((n_particles,3))
    nq[:,0]=p_B[:,0]*vp_sq_safe+p_C*vp_x*vp_sq_safe; nq[:,1]=p_B[:,1]*vp_sq_safe+p_C*vp_y*vp_sq_safe; nq[:,2]=p_B[:,2]*vp_sq_safe+p_C*vp_z*vp_sq_safe
    xi_raw=rfn_gpu((n_particles,3)); sxi=cp.zeros((nc_val,3)); vxi_calc=cp.zeros((nc_val,3))
    nloc_safe=cp.maximum(grid['N'],EPSILON)
    cp.add.at(sxi, cell_indices, xi_raw*p_weight[:,cp.newaxis]); sxi/=nloc_safe[:,cp.newaxis]
    xi_mean_corrected=xi_raw-sxi[cell_indices]
    cp.add.at(vxi_calc, cell_indices, xi_mean_corrected**2*p_weight[:,cp.newaxis])
    vxi_term=cp.maximum(vxi_calc/nloc_safe[:,cp.newaxis],0.0); vxi_stddev=cp.sqrt(vxi_term)
    vxi_stddev_safe=cp.maximum(vxi_stddev,EPSILON)
    xi_normalized=xi_mean_corrected / cp.maximum(vxi_stddev_safe[cell_indices], EPSILON)
    e_frac1=cp.bincount(cell_indices,weights=vp_sq*p_weight,minlength=nc_val)
    vp_x_new=(A1*vp_x+A2*(nvA[:,0]+nq[:,0]+gforce[:,0])+A3*xi_normalized[:,0])
    vp_y_new=(A1*vp_y+A2*(nvA[:,1]+nq[:,1]+gforce[:,1])+A3*xi_normalized[:,1])
    vp_z_new=(A1*vp_z+A2*(nvA[:,2]+nq[:,2]+gforce[:,2])+A3*xi_normalized[:,2])
    vp_sq_new=vp_x_new**2+vp_y_new**2+vp_z_new**2
    e_frac2=cp.bincount(cell_indices,weights=vp_sq_new*p_weight,minlength=nc_val)
    e_frac1_safe=cp.maximum(e_frac1,EPSILON); e_frac2_safe=cp.maximum(e_frac2,EPSILON)
    e_frac=cp.sqrt(e_frac1_safe/e_frac2_safe)
    e_frac_particles = e_frac[cell_indices]
    p_vp_x[:]=e_frac_particles*vp_x_new
    p_vp_y[:]=e_frac_particles*vp_y_new
    p_vp_z[:]=e_frac_particles*vp_z_new
    U_full_particles = grid['U'][cell_indices]
    p_vx[:]=U_full_particles[:,0]+p_vp_x
    p_vy[:]=U_full_particles[:,1]+p_vp_y
    p_vz[:]=U_full_particles[:,2]+p_vp_z

def average_results_cupy_stable(avg_grid, grid_gpu, nt, ntss):
    nave = 1.0 / float(nt - ntss)
    for key in avg_grid.keys():
        if key in grid_gpu and isinstance(avg_grid[key], cp.ndarray):
            if key not in ['pos', 'vol']:
                avg_grid[key] = nave * grid_gpu[key] + (1.0 - nave) * avg_grid[key]
    avg_grid['T'] = cp.maximum(avg_grid['T'], 1.0)


# ============================================================================
# 4. SOLVER FUNCTIONS (Physics vs. NATIVE ML)
# ============================================================================

def build_linear_systems_cupy(grid, linsys):
    lhs = linsys['lhs']; rhs = linsys['rhs']
    PIJ=grid['PIJ']; Q=grid['Q']; M3=grid['M3']; M4=grid['M4']; M5=grid['M5']
    DM2=grid['DM2']; DM4=grid['DM4']; lam=grid['lam']; nu=grid['nu']; nubol=grid['nubol']
    lhs.fill(0); rhs.fill(0)
    lhs[:,0,0]=2.*PIJ[:,0]; lhs[:,0,1]=2.*PIJ[:,1]; lhs[:,0,2]=2.*PIJ[:,2]
    lhs[:,1,0]=PIJ[:,1]; lhs[:,1,1]=PIJ[:,0]+PIJ[:,3]; lhs[:,1,2]=PIJ[:,4]; lhs[:,1,3]=PIJ[:,1]; lhs[:,1,4]=PIJ[:,2]
    lhs[:,2,0]=PIJ[:,2]; lhs[:,2,1]=PIJ[:,4]; lhs[:,2,2]=PIJ[:,0]+PIJ[:,5]; lhs[:,2,4]=PIJ[:,1]; lhs[:,2,5]=PIJ[:,2]
    lhs[:,3,1]=2.*PIJ[:,1]; lhs[:,3,3]=2.*PIJ[:,3]; lhs[:,3,4]=2.*PIJ[:,4]
    lhs[:,4,1]=PIJ[:,2]; lhs[:,4,2]=PIJ[:,1]; lhs[:,4,3]=PIJ[:,4]; lhs[:,4,4]=PIJ[:,3]+PIJ[:,5]; lhs[:,4,5]=PIJ[:,4]
    lhs[:,5,2]=2.*PIJ[:,2]; lhs[:,5,4]=2.*PIJ[:,4]; lhs[:,5,5]=2.*PIJ[:,5]
    lhs[:,0,6]=2.*Q[:,0]; lhs[:,1,6]=Q[:,1]; lhs[:,1,7]=Q[:,0]; lhs[:,2,6]=Q[:,2]; lhs[:,2,8]=Q[:,0]
    lhs[:,3,7]=2.*Q[:,1]; lhs[:,4,7]=Q[:,2]; lhs[:,4,8]=Q[:,1]; lhs[:,5,8]=2.*Q[:,2]
    lhs[:,6,0]=Q[:,0]+2.*M3[:,0]; lhs[:,7,0]=2.*M3[:,1]; lhs[:,8,0]=2.*M3[:,2]
    lhs[:,6,1]=Q[:,1]+4.*M3[:,1]; lhs[:,7,1]=Q[:,0]+4.*M3[:,3]; lhs[:,8,1]=4.*M3[:,4]
    lhs[:,6,2]=Q[:,2]+4.*M3[:,2]; lhs[:,7,2]=4.*M3[:,4]; lhs[:,8,2]=Q[:,0]+4.*M3[:,5]
    lhs[:,6,3]=2.*M3[:,3]; lhs[:,7,3]=Q[:,1]+2.*M3[:,6]; lhs[:,8,3]=2.*M3[:,7]
    lhs[:,6,4]=4.*M3[:,4]; lhs[:,7,4]=Q[:,2]+4.*M3[:,7]; lhs[:,8,4]=Q[:,1]+4.*M3[:,8]
    lhs[:,6,5]=2.*M3[:,5]; lhs[:,7,5]=2.*M3[:,8]; lhs[:,8,5]=Q[:,2]+2.*M3[:,9]
    DM4_term=DM4-DM2**2
    lhs[:,6,6]=DM4_term+2.*M4[:,0]-2.*DM2*PIJ[:,0]; lhs[:,6,7]=2.*M4[:,1]-2.*DM2*PIJ[:,1]; lhs[:,6,8]=2.*M4[:,2]-2.*DM2*PIJ[:,2]
    lhs[:,7,6]=2.*M4[:,1]-2.*DM2*PIJ[:,1]; lhs[:,7,7]=DM4_term+2.*M4[:,3]-2.*DM2*PIJ[:,3]; lhs[:,7,8]=2.*M4[:,4]-2.*DM2*PIJ[:,4]
    lhs[:,8,6]=2.*M4[:,2]-2.*DM2*PIJ[:,2]; lhs[:,8,7]=2.*M4[:,4]-2.*DM2*PIJ[:,4]; lhs[:,8,8]=DM4_term+2.*M4[:,5]-2.*DM2*PIJ[:,5]
    diag_boost = 1e-10 * cp.identity(9)
    lhs += diag_boost[cp.newaxis, :, :] 
    rhs[:,0]=lam*(-2.*M4[:,0]); rhs[:,1]=lam*(-2.*M4[:,1]); rhs[:,2]=lam*(-2.*M4[:,2])
    rhs[:,3]=lam*(-2.*M4[:,3]); rhs[:,4]=lam*(-2.*M4[:,4]); rhs[:,5]=lam*(-2.*M4[:,5])
    rhs[:,6]=-lam*(3.*M5[:,0]-DM2*Q[:,0]-2.*(PIJ[:,0]*Q[:,0]+PIJ[:,1]*Q[:,1]+PIJ[:,2]*Q[:,2]))
    rhs[:,7]=-lam*(3.*M5[:,1]-DM2*Q[:,1]-2.*(PIJ[:,1]*Q[:,0]+PIJ[:,3]*Q[:,1]+PIJ[:,4]*Q[:,2]))
    rhs[:,8]=-lam*(3.*M5[:,2]-DM2*Q[:,2]-2.*(PIJ[:,2]*Q[:,0]+PIJ[:,4]*Q[:,1]+PIJ[:,5]*Q[:,2]))
    nu_term = (3.*nu - 2./3.*nubol)
    rhs[:,6]+=nu_term*Q[:,0]; rhs[:,7]+=nu_term*Q[:,1]; rhs[:,8]+=nu_term*Q[:,2]

def solve_linear_systems_cupy(linsys, coeffs):
    try:
        X = cp.linalg.solve(linsys['lhs'], linsys['rhs'])
    except cp.linalg.LinAlgError:
        print(f"Warning: Batched solve failed.")
        X = cp.zeros((NC, 9))
    coeffs['A'][:,0:6] = X[:,0:6]; coeffs['B'][:,0:3] = X[:,6:9]

# --- *** NEW: SOLVER 2: NATIVE CuPy Trained Model (UPDATED) *** ---

def relu_gpu(x):
    """Native CuPy ReLU activation function."""
    return cp.maximum(x, 0)

def predict_coeffs_cupy_NATIVE_TRAINED(grid_gpu, coeffs_gpu, model_assets_gpu):
    """
    (UPDATED for 4-layer model)
    Runs inference using pure CuPy operations.
    """
    
    # 1. Get model assets (which are already CuPy arrays)
    assets = model_assets_gpu
    
    # 2. Extract 16 features from GPU grid
    X_gpu = cp.stack([
        grid_gpu['rho'],
        grid_gpu['T'],
        grid_gpu['U'][:, 0], grid_gpu['U'][:, 1], grid_gpu['U'][:, 2],
        grid_gpu['PIJ'][:, 0], grid_gpu['PIJ'][:, 1], grid_gpu['PIJ'][:, 2],
        grid_gpu['PIJ'][:, 3], grid_gpu['PIJ'][:, 4], grid_gpu['PIJ'][:, 5],
        grid_gpu['Q'][:, 0], grid_gpu['Q'][:, 1], grid_gpu['Q'][:, 2],
        grid_gpu['DM2'],
        grid_gpu['nu']
    ], axis=1)
    
    # 3. Scale inputs (Native CuPy)
    X_scaled = (X_gpu - assets['X_mean']) / assets['X_scale']
    
    # 4. Run inference (Native CuPy) - *** UPDATED FOR 4 LAYERS ***
    L1 = relu_gpu(cp.dot(X_scaled, assets['W1']) + assets['b1'])
    L2 = relu_gpu(cp.dot(L1, assets['W2']) + assets['b2'])
    L3 = relu_gpu(cp.dot(L2, assets['W3']) + assets['b3'])
    L4 = relu_gpu(cp.dot(L3, assets['W4']) + assets['b4']) # <-- New 4th layer
    Out_scaled = cp.dot(L4, assets['W5']) + assets['b5'] # <-- Output layer is now W5/b5
    
    # 5. Inverse-scale outputs (Native CuPy)
    Out_unscaled = (Out_scaled * assets['y_scale']) + assets['y_mean']
    
    # 6. Assign results directly to coeffs_gpu
    coeffs_gpu['A'][:] = Out_unscaled[:, 0:6]
    coeffs_gpu['B'][:] = Out_unscaled[:, 6:9]
    coeffs_gpu['C'].fill(0)


# ============================================================================
# 5. Post-processing and Publication Plotting
# ============================================================================

def _moving_average_reflect(y, window):
    """Centered moving average with reflected boundaries."""
    y = np.asarray(y, dtype=np.float64)
    if window is None or window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(ypad, kernel, mode="valid")


def postprocess_couette_profiles(
    x,
    u,
    t,
    rho,
    smooth_window=SMOOTH_WINDOW,
    enforce_symmetry=ENFORCE_SYMMETRY,
):
    """
    Postprocess one Couette profile for paper-quality plotting.

    The simulated case is symmetric:
      - U_y is antisymmetric about x/L = 0.5.
      - T and rho are symmetric about x/L = 0.5.

    The symmetry projection is used only after the run, for figure-quality
    profiles. It reduces Monte Carlo asymmetry/noise and removes the spurious
    centerline wiggle without changing the actual solver.
    """
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)

    order = np.argsort(x)
    x = x[order]
    u = u[order]
    t = t[order]
    rho = rho[order]

    if enforce_symmetry:
        u = 0.5 * (u - u[::-1])
        t = 0.5 * (t + t[::-1])
        rho = 0.5 * (rho + rho[::-1])

    if smooth_window and smooth_window > 1:
        u = _moving_average_reflect(u, smooth_window)
        t = _moving_average_reflect(t, smooth_window)
        rho = _moving_average_reflect(rho, smooth_window)

    return x, u, t, rho


def _make_publication_axes(ylabel):
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12.5,
        "axes.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_profile_with_band(
    kn_value,
    x,
    y_phys_mean,
    y_phys_std,
    y_ml_mean,
    y_ml_std,
    ylabel,
    stem,
):
    """
    Save one standalone PDF plot without title/top caption.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = _make_publication_axes(ylabel)
    xL = np.asarray(x, dtype=np.float64) / LX

    ax.plot(xL, y_phys_mean, color="tab:blue", linewidth=2.2, label="Physics mean")
    ax.fill_between(
        xL,
        y_phys_mean - y_phys_std,
        y_phys_mean + y_phys_std,
        color="tab:blue",
        alpha=0.16,
        linewidth=0.0,
        label="Physics band",
    )

    ax.plot(xL, y_ml_mean, color="tab:red", linestyle="--", linewidth=2.2, label="ML mean")
    ax.fill_between(
        xL,
        y_ml_mean - y_ml_std,
        y_ml_mean + y_ml_std,
        color="tab:red",
        alpha=0.14,
        linewidth=0.0,
        label="ML band",
    )

    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best", frameon=True)

    kn_tag = f"{kn_value:g}".replace(".", "p")
    pdf_path = FIG_DIR / f"{stem}_kn_{kn_tag}.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    if SAVE_PNG_COPY:
        fig.savefig(FIG_DIR / f"{stem}_kn_{kn_tag}.png", dpi=FIG_DPI, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved {pdf_path}")


def save_profile_tables(kn_value, x, bundle):
    """
    Save numerical data used in each PDF plot.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kn_tag = f"{kn_value:g}".replace(".", "p")
    xL = np.asarray(x, dtype=np.float64) / LX

    for key in ("velocity", "temperature", "density"):
        arr = bundle[key]
        out = DATA_DIR / f"{key}_kn_{kn_tag}.csv"
        header = "x_m,x_over_L,physics_mean,physics_std,ml_mean,ml_std"
        table = np.column_stack([
            x,
            xL,
            arr["phys_mean"],
            arr["phys_std"],
            arr["ml_mean"],
            arr["ml_std"],
        ])
        np.savetxt(out, table, delimiter=",", header=header, comments="")
        print(f"Saved {out}")


# ============================================================================
# 6. Main Execution
# ============================================================================

def load_native_model_to_gpu(npz_filename):
    """
    Loads the .npz file containing NumPy arrays and transfers model parameters
    to the GPU as CuPy arrays.
    """
    print(f"Loading native model parameters from '{npz_filename}'...")
    try:
        params_np = np.load(npz_filename)
        model_assets_gpu = {}
        for key, val in params_np.items():
            model_assets_gpu[key] = cp.asarray(val)

        print("All model parameters (weights, scalers) transferred to GPU.")
        print(f"  W1 on GPU, shape: {model_assets_gpu['W1'].shape}")
        print(f"  X_mean on GPU, shape: {model_assets_gpu['X_mean'].shape}")
        return model_assets_gpu

    except FileNotFoundError:
        print(f"Error: '{npz_filename}' not found.")
        print("Please run the 4-layer extract_params.py / conversion script first.")
        return None
    except Exception as e:
        print(f"Error loading native model file: {e}")
        return None


def make_empty_avg_grid_cpu():
    return {
        "pos": np.zeros(NC, dtype=np.float64),
        "vol": np.zeros(NC, dtype=np.float64),
        "U": np.zeros((NC, 3), dtype=np.float64),
        "N": np.zeros(NC, dtype=np.float64),
        "T": np.zeros(NC, dtype=np.float64),
        "rho": np.zeros(NC, dtype=np.float64),
        "PIJ": np.zeros((NC, 6), dtype=np.float64),
        "Q": np.zeros((NC, 3), dtype=np.float64),
        "DM2": np.zeros(NC, dtype=np.float64),
        "nu": np.zeros(NC, dtype=np.float64),
    }


def run_one_solver(
    mode,
    replica_id,
    current_rho_in,
    current_t_in,
    current_theta0_in,
    current_w_particle,
    current_dt,
    model_assets_gpu=None,
):
    """
    Run one replica of either the full physics solver or the native ML solver.
    """
    seed_shift = 0 if mode == "physics" else 1_000_000
    cp.random.seed(BASE_RANDOM_SEED + seed_shift + replica_id)
    np.random.seed(BASE_RANDOM_SEED + seed_shift + replica_id)

    grid_gpu, coeffs_gpu, linsys_gpu = initialize_grid_cupy(NC, LX, current_rho_in, current_t_in)
    p_data = initialize_particles_cupy(NP, LX, current_theta0_in, current_w_particle)

    avg_grid_gpu, _, _ = initialize_grid_cupy(NC, LX, current_rho_in, current_t_in)
    for key in avg_grid_gpu:
        if key not in ["pos", "vol"] and isinstance(avg_grid_gpu[key], cp.ndarray):
            avg_grid_gpu[key].fill(0)

    start_time = time.time()

    for nt in range(1, N_STEPS_PER_RUN + 1):
        if nt > AVG_START_STEP:
            average_results_cupy_stable(avg_grid_gpu, grid_gpu, nt, AVG_START_STEP)

        if nt % 2000 == 0:
            print(
                f"\r  {mode.upper()} replica {replica_id + 1}/{N_REPLICAS} "
                f"step {nt}/{N_STEPS_PER_RUN}...",
                end="",
            )

        p_data[9][:] = p_data[0]
        p_data[10][:] = p_data[1]
        p_data[0][:] = p_data[9] + p_data[3] * current_dt

        apply_boundary_couette_cupy(
            p_data,
            LX,
            current_dt,
            UW1_BASE,
            UW2_BASE,
            THETAW1_GLOBAL,
            THETAW2_GLOBAL,
        )

        if mode == "physics":
            sort_and_calc_moments_cupy_FULL(p_data, grid_gpu, NC, LX)
            build_linear_systems_cupy(grid_gpu, linsys_gpu)
            solve_linear_systems_cupy(linsys_gpu, coeffs_gpu)
        elif mode == "ml":
            sort_and_calc_moments_cupy_LITE(p_data, grid_gpu, NC, LX)
            predict_coeffs_cupy_NATIVE_TRAINED(grid_gpu, coeffs_gpu, model_assets_gpu)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        evolve_velocities_cupy(p_data, grid_gpu, coeffs_gpu, current_dt, NC)

    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start_time
    print(f"\nFinished {mode.upper()} replica {replica_id + 1}/{N_REPLICAS} in {elapsed:.2f} s")

    avg_grid_cpu = make_empty_avg_grid_cpu()
    for key in avg_grid_cpu:
        if key in avg_grid_gpu:
            avg_grid_cpu[key] = cp.asnumpy(avg_grid_gpu[key])

    return avg_grid_cpu, elapsed


def run_single_comparison(kn_factor, base_rho, base_temp, model_assets_gpu):
    """
    Run multiple replicas for one Knudsen number and return mean/std profiles.
    """
    current_rho_in = base_rho / kn_factor
    current_t_in = base_temp
    current_theta0_in = np.sqrt(K_B * current_t_in / MASS_AR)
    current_w_particle = (LX * current_rho_in) / float(NP)
    current_dt = 0.5 * (LX / float(NX)) / max(
        np.sqrt(K_B * current_t_in / MASS_AR),
        abs(UW2_BASE),
    )
    kn_value = 0.15 * kn_factor
    kn_str = f"{kn_value:g}"

    print("\n" + "=" * 72)
    print(f"Starting comparison for Kn = {kn_str}")
    print(f"rho_in = {current_rho_in:.4e}, particle weight = {current_w_particle:.4e}")
    print(f"Replicas = {N_REPLICAS}, averaging starts at step {AVG_START_STEP}")

    phys_u_list, phys_t_list, phys_rho_list = [], [], []
    ml_u_list, ml_t_list, ml_rho_list = [], [], []
    phys_times, ml_times = [], []
    x_ref = None

    for replica_id in range(N_REPLICAS):
        phys_avg, t_phys = run_one_solver(
            "physics",
            replica_id,
            current_rho_in,
            current_t_in,
            current_theta0_in,
            current_w_particle,
            current_dt,
            model_assets_gpu,
        )
        x_proc, u_proc, t_proc, rho_proc = postprocess_couette_profiles(
            phys_avg["pos"],
            phys_avg["U"][:, 1],
            phys_avg["T"],
            phys_avg["rho"],
        )
        phys_u_list.append(u_proc)
        phys_t_list.append(t_proc)
        phys_rho_list.append(rho_proc)
        phys_times.append(t_phys)
        x_ref = x_proc

        ml_avg, t_ml = run_one_solver(
            "ml",
            replica_id,
            current_rho_in,
            current_t_in,
            current_theta0_in,
            current_w_particle,
            current_dt,
            model_assets_gpu,
        )
        x_proc_ml, u_proc_ml, t_proc_ml, rho_proc_ml = postprocess_couette_profiles(
            ml_avg["pos"],
            ml_avg["U"][:, 1],
            ml_avg["T"],
            ml_avg["rho"],
        )
        ml_u_list.append(u_proc_ml)
        ml_t_list.append(t_proc_ml)
        ml_rho_list.append(rho_proc_ml)
        ml_times.append(t_ml)

        if not np.allclose(x_ref, x_proc_ml):
            raise RuntimeError("Position grids from physics and ML postprocessing do not match.")

    phys_u = np.vstack(phys_u_list)
    phys_t = np.vstack(phys_t_list)
    phys_rho = np.vstack(phys_rho_list)

    ml_u = np.vstack(ml_u_list)
    ml_t = np.vstack(ml_t_list)
    ml_rho = np.vstack(ml_rho_list)

    bundle = {
        "velocity": {
            "phys_mean": phys_u.mean(axis=0),
            "phys_std": phys_u.std(axis=0, ddof=0),
            "ml_mean": ml_u.mean(axis=0),
            "ml_std": ml_u.std(axis=0, ddof=0),
        },
        "temperature": {
            "phys_mean": phys_t.mean(axis=0),
            "phys_std": phys_t.std(axis=0, ddof=0),
            "ml_mean": ml_t.mean(axis=0),
            "ml_std": ml_t.std(axis=0, ddof=0),
        },
        "density": {
            "phys_mean": phys_rho.mean(axis=0),
            "phys_std": phys_rho.std(axis=0, ddof=0),
            "ml_mean": ml_rho.mean(axis=0),
            "ml_std": ml_rho.std(axis=0, ddof=0),
        },
    }

    save_profile_tables(kn_value, x_ref, bundle)

    time_physics = float(np.mean(phys_times))
    time_ml = float(np.mean(ml_times))
    speedup = time_physics / time_ml if time_ml > 0 else 0.0

    return kn_value, time_physics, time_ml, speedup, x_ref, bundle


def main():
    print("=" * 72)
    print("  Publication-quality Couette comparison: PHYSICS vs. NATIVE ML")
    print("=" * 72)
    print(f"  GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    print(f"  Steps per replica: {N_STEPS_PER_RUN}, particles: {NP}")
    print(f"  Replicas: {N_REPLICAS}")
    print(f"  Averaging starts at step: {AVG_START_STEP}")
    print(f"  Smooth window: {SMOOTH_WINDOW}")
    print(f"  Enforce Couette symmetry in postprocess: {ENFORCE_SYMMETRY}")

    model_assets_gpu = load_native_model_to_gpu("native_model_params.npz")
    if model_assets_gpu is None:
        print("Exiting due to error loading model parameters.")
        return

    # Test Knudsen numbers. Edit this list if needed.
    kn_target_values = [1.0, 0.7]
    kn_factors = [kn / 0.15 for kn in kn_target_values]

    print(f"\nRunning simulations for Kn = {kn_target_values}")

    results_list = []
    for factor in kn_factors:
        kn_value, time_p, time_ml, speedup, pos_data, bundle = run_single_comparison(
            kn_factor=factor,
            base_rho=RHO_IN_BASE_GLOBAL,
            base_temp=T_IN_BASE_GLOBAL,
            model_assets_gpu=model_assets_gpu,
        )
        results_list.append((kn_value, time_p, time_ml, speedup, pos_data, bundle))

    print("\n" + "=" * 80)
    print("Final Performance Report")
    print("=" * 80)
    print(f"{'Kn':<12} | {'Physics mean time (s)':<24} | {'ML mean time (s)':<18} | {'Speedup':<10}")
    print("-" * 80)
    for kn, time_p, time_ml, speedup, _, _ in results_list:
        print(f"{kn:<12.4g} | {time_p:<24.2f} | {time_ml:<18.2f} | {speedup:<10.2f}x")
    print("=" * 80)

    print("\nGenerating separate publication-quality PDF plots...")
    for kn_value, _, _, _, pos_data, bundle in results_list:
        plot_profile_with_band(
            kn_value,
            pos_data,
            bundle["velocity"]["phys_mean"],
            bundle["velocity"]["phys_std"],
            bundle["velocity"]["ml_mean"],
            bundle["velocity"]["ml_std"],
            r"$U_y$ (m/s)",
            "couette_velocity",
        )
        plot_profile_with_band(
            kn_value,
            pos_data,
            bundle["temperature"]["phys_mean"],
            bundle["temperature"]["phys_std"],
            bundle["temperature"]["ml_mean"],
            bundle["temperature"]["ml_std"],
            r"$T$ (K)",
            "couette_temperature",
        )
        plot_profile_with_band(
            kn_value,
            pos_data,
            bundle["density"]["phys_mean"],
            bundle["density"]["phys_std"],
            bundle["density"]["ml_mean"],
            bundle["density"]["ml_std"],
            r"$\rho$ (kg/m$^3$)",
            "couette_density",
        )

    print("All plotting complete.")
    print(f"PDF figures are in: {FIG_DIR.resolve()}")
    print(f"CSV profile tables are in: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
