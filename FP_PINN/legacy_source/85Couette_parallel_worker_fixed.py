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
import argparse

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
N_SIMULATION_RUNS = 1  # Only 1 run
N_STEPS_PER_RUN = 4500      # overridden by --steps
EPSILON = 1e-30

BASE_RANDOM_SEED = 24680
 

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
# 5. Parallel worker: one Physics or ML replica
# ============================================================================

def load_native_model_to_gpu(npz_filename):
    print(f"Loading native model parameters from '{npz_filename}'...", flush=True)
    try:
        params_np = np.load(npz_filename)
        model_assets_gpu = {key: cp.asarray(val) for key, val in params_np.items()}
        print("All model parameters transferred to GPU.", flush=True)
        print(f"  W1 shape: {model_assets_gpu['W1'].shape}", flush=True)
        return model_assets_gpu
    except FileNotFoundError:
        print(f"ERROR: '{npz_filename}' not found.", flush=True)
        return None
    except Exception as e:
        print(f"ERROR loading native model file: {e}", flush=True)
        return None


def save_profile(outdir, tag, kn_value, mode, replica, step, elapsed, grid_cpu):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pos = np.asarray(grid_cpu["pos"], dtype=np.float64)
    uy = np.asarray(grid_cpu["U"][:, 1], dtype=np.float64)
    temp = np.asarray(grid_cpu["T"], dtype=np.float64)
    rho = np.asarray(grid_cpu["rho"], dtype=np.float64)

    npz_path = outdir / f"{tag}.npz"
    csv_path = outdir / f"{tag}.csv"

    np.savez(
        npz_path,
        kn=kn_value,
        mode=mode,
        replica=replica,
        step=step,
        elapsed_sec=elapsed,
        x_m=pos,
        x_over_L=pos / LX,
        uy=uy,
        T=temp,
        rho=rho,
    )

    table = np.column_stack([pos, pos / LX, uy, temp, rho])
    np.savetxt(csv_path, table, delimiter=",", header="x_m,x_over_L,uy,T,rho", comments="")
    print(f"Saved {npz_path}", flush=True)


def run_worker(args):
    global NX, NC, PARTICLES_PER_CELL_TARGET, NP, N_STEPS_PER_RUN, NTSS
    global BASE_RANDOM_SEED

    NX = int(args.nx)
    NC = NX
    PARTICLES_PER_CELL_TARGET = int(args.ppc)
    NP = PARTICLES_PER_CELL_TARGET * NC
    N_STEPS_PER_RUN = int(args.steps)
    NTSS = int(args.ntss)
    BASE_RANDOM_SEED = int(args.seed)

    if NTSS >= N_STEPS_PER_RUN:
        raise ValueError("--ntss must be smaller than --steps")

    mode = args.mode.lower()
    if mode not in ("physics", "ml"):
        raise ValueError("--mode must be physics or ml")

    # Same seed for corresponding physics/ML replica: paired comparison.
    seed = BASE_RANDOM_SEED + int(args.replica)
    cp.random.seed(seed)
    np.random.seed(seed)

    model_assets_gpu = None
    if mode == "ml":
        model_assets_gpu = load_native_model_to_gpu(args.model)
        if model_assets_gpu is None:
            raise RuntimeError("Could not load ML model.")

    kn_factor = float(args.kn) / 0.15
    current_rho_in = RHO_IN_BASE_GLOBAL / kn_factor
    current_t_in = T_IN_BASE_GLOBAL
    current_theta0_in = np.sqrt(K_B * current_t_in / MASS_AR)
    current_w_particle = (LX * current_rho_in) / float(NP)
    current_dt = 0.5 * (LX / float(NX)) / max(np.sqrt(K_B * current_t_in / MASS_AR), abs(UW2_BASE))

    outdir = Path(args.outdir) / f"{mode}_r{int(args.replica):02d}"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print(f"Couette parallel worker | mode={mode} | replica={args.replica}", flush=True)
    print(f"Kn={args.kn}, Nx={NX}, ppc={PARTICLES_PER_CELL_TARGET}, NP={NP}", flush=True)
    print(f"steps={N_STEPS_PER_RUN}, NTSS={NTSS}, averaging samples={N_STEPS_PER_RUN-NTSS}", flush=True)
    print(f"rho_in={current_rho_in:.6e}, particle_weight={current_w_particle:.6e}", flush=True)
    print(f"Output dir: {outdir}", flush=True)
    print("=" * 78, flush=True)

    grid_gpu, coeffs_gpu, linsys_gpu = initialize_grid_cupy(NC, LX, current_rho_in, current_t_in)
    p_data = initialize_particles_cupy(NP, LX, current_theta0_in, current_w_particle)
    avg_grid_gpu, _, _ = initialize_grid_cupy(NC, LX, current_rho_in, current_t_in)

    for key in avg_grid_gpu:
        if key not in ["pos", "vol"] and isinstance(avg_grid_gpu[key], cp.ndarray):
            avg_grid_gpu[key].fill(0)

    start_time = time.time()
    last_save = 0

    for nt in range(1, N_STEPS_PER_RUN + 1):
        if nt > NTSS:
            average_results_cupy_stable(avg_grid_gpu, grid_gpu, nt, NTSS)

        if nt == 1 or nt % args.report_every == 0:
            elapsed_now = time.time() - start_time
            sec_per_step = elapsed_now / max(nt, 1)
            eta_min = sec_per_step * max(N_STEPS_PER_RUN - nt, 0) / 60.0
            print(
                f"\n  {mode.upper()} r{args.replica} step {nt}/{N_STEPS_PER_RUN} "
                f"| elapsed={elapsed_now/60:.1f} min | {sec_per_step:.3f} s/step | ETA={eta_min:.1f} min",
                flush=True,
            )

        p_data[9][:] = p_data[0]
        p_data[10][:] = p_data[1]
        p_data[0][:] = p_data[9] + p_data[3] * current_dt

        apply_boundary_couette_cupy(p_data, LX, current_dt, UW1_BASE, UW2_BASE, THETAW1_GLOBAL, THETAW2_GLOBAL)

        if mode == "physics":
            sort_and_calc_moments_cupy_FULL(p_data, grid_gpu, NC, LX)
            build_linear_systems_cupy(grid_gpu, linsys_gpu)
            solve_linear_systems_cupy(linsys_gpu, coeffs_gpu)
        else:
            sort_and_calc_moments_cupy_LITE(p_data, grid_gpu, NC, LX)
            predict_coeffs_cupy_NATIVE_TRAINED(grid_gpu, coeffs_gpu, model_assets_gpu)

        evolve_velocities_cupy(p_data, grid_gpu, coeffs_gpu, current_dt, NC)

        if args.checkpoint_every > 0 and nt > NTSS and (nt - last_save) >= args.checkpoint_every:
            cp.cuda.Stream.null.synchronize()
            grid_cpu = {
                "pos": cp.asnumpy(avg_grid_gpu["pos"]),
                "U": cp.asnumpy(avg_grid_gpu["U"]),
                "T": cp.asnumpy(avg_grid_gpu["T"]),
                "rho": cp.asnumpy(avg_grid_gpu["rho"]),
            }
            save_profile(outdir, "profile_latest", float(args.kn), mode, int(args.replica), nt, time.time() - start_time, grid_cpu)
            last_save = nt

    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start_time

    grid_cpu = {
        "pos": cp.asnumpy(avg_grid_gpu["pos"]),
        "U": cp.asnumpy(avg_grid_gpu["U"]),
        "T": cp.asnumpy(avg_grid_gpu["T"]),
        "rho": cp.asnumpy(avg_grid_gpu["rho"]),
    }
    save_profile(outdir, "profile_final", float(args.kn), mode, int(args.replica), N_STEPS_PER_RUN, elapsed, grid_cpu)

    summary = "\n".join([
        "Couette parallel worker summary",
        "=" * 78,
        f"mode = {mode}",
        f"replica = {args.replica}",
        f"kn = {args.kn}",
        f"nx = {NX}",
        f"ppc = {PARTICLES_PER_CELL_TARGET}",
        f"NP = {NP}",
        f"steps = {N_STEPS_PER_RUN}",
        f"NTSS = {NTSS}",
        f"elapsed_sec = {elapsed:.3f}",
        f"profile = {outdir/'profile_final.npz'}",
    ])
    (outdir / "summary.txt").write_text(summary)
    print("\n" + summary, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["physics", "ml"])
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--kn", type=float, default=0.015)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ppc", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=4500)
    parser.add_argument("--ntss", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--outdir", default="couette82_parallel_kn0015_runs")
    parser.add_argument("--model", default="native_model_params.npz")
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    return parser.parse_args()


def main():
    run_worker(parse_args())


if __name__ == "__main__":
    main()
