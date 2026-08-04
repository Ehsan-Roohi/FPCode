import numpy as np
import math
import time
import warnings
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import pandas as pd
from numba import cuda, float32, float64
import cupy as cp  # REQUIRED FOR FAST ML
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32
from numba.core.errors import NumbaPerformanceWarning
import gc

# =============================================================================
# 1) CONFIGURATION
# =============================================================================
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# Simulation Controls
TOTAL_STEPS = 5000       
WARMUP_STEPS = 2000      
REPORT_INTERVAL = 500   
MODEL_PARAMS_FILE = "model_params.npz"

# Gas & Geometry
MASS = 6.63e-26
KB = 1.38e-23
T_INF = 200.0
T_WALL = 500.0
U_INF = 2634.1
NUM_DENS_INF = 4.247e20
RHO_INF = NUM_DENS_INF * MASS 
LAMBDA_INF = 3.048e-3
OMEGA = 0.74
PR = 0.667
T_REF = 1000.0
DIAMETER = 0.3048          
R_CYL = DIAMETER / 2.0
R_DOM = 0.65
Q_DYN_REF = 0.5 * RHO_INF * (U_INF**2)
Q_HEAT_REF = 0.5 * RHO_INF * (U_INF**3)
P_INF = (RHO_INF / MASS) * KB * T_INF 

# Solver Settings
MAX_NODES = 1000000        
MAX_DEPTH = 9              
MIN_DEPTH = 5              
INIT_PARTICLES = 4000000   
GRAD_LIMIT = 0.10          
COARSEN_GRAD_LIMIT = 0.05  
REFINE_MAX_PARTICLES = 5000 
FREEZE_STEPS = 2000        
SURF_BINS = 90             
CFL_NUM = 0.5

# =============================================================================
# 2) DEVICE FUNCTIONS
# =============================================================================
@cuda.jit(device=True)
def get_leaf_index(x, y, tree_children, tree_center, R_cyl, R_dom):
    r_sq = x*x + y*y
    if r_sq < R_cyl*R_cyl or r_sq > R_dom*R_dom: return -1
    r = math.sqrt(r_sq)
    xi = (r - R_cyl) / (R_dom - R_cyl)
    theta = math.atan2(y, x)
    if theta < 0: theta += 2*math.pi 
    eta = theta / math.pi 
    curr = 0
    for _ in range(MAX_DEPTH + 1):
        if tree_children[curr, 0] == -1: return curr
        c_xi = tree_center[curr, 0]; c_eta = tree_center[curr, 1]
        quad = 0
        if xi >= c_xi: quad += 1
        if eta >= c_eta: quad += 2
        curr = tree_children[curr, quad]
        if curr == -1: return -1
    return curr

@cuda.jit(device=True)
def calc_cell_volume(tree_size, tree_center, R_cyl, R_dom):
    d_eta = tree_size[1] * 2.0
    r1 = R_cyl + (tree_center[0] - tree_size[0]) * (R_dom - R_cyl)
    r2 = R_cyl + (tree_center[0] + tree_size[0]) * (R_dom - R_cyl)
    dtheta = d_eta * math.pi
    vol = 0.5 * (r2*r2 - r1*r1) * dtheta
    if vol < 1e-20: vol = 1e-20
    return vol

@cuda.jit(device=True)
def relu(x): return max(0.0, x)

# Optimized Forward Pass with Ping-Pong Buffers
@cuda.jit(device=True)
def dnn_forward_optimized(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15,
                          mean_in, scale_in, mean_out, scale_out, 
                          W1T, b1, W2T, b2, W3T, b3, W4T, b4, W5T, b5, out_coeffs):
    
    buff_A = cuda.local.array(256, dtype=float32)
    buff_B = cuda.local.array(256, dtype=float32)

    buff_A[0]=(in_0-mean_in[0])/scale_in[0]; buff_A[1]=(in_1-mean_in[1])/scale_in[1]; buff_A[2]=(in_2-mean_in[2])/scale_in[2]; buff_A[3]=(in_3-mean_in[3])/scale_in[3]
    buff_A[4]=(in_4-mean_in[4])/scale_in[4]; buff_A[5]=(in_5-mean_in[5])/scale_in[5]; buff_A[6]=(in_6-mean_in[6])/scale_in[6]; buff_A[7]=(in_7-mean_in[7])/scale_in[7]
    buff_A[8]=(in_8-mean_in[8])/scale_in[8]; buff_A[9]=(in_9-mean_in[9])/scale_in[9]; buff_A[10]=(in_10-mean_in[10])/scale_in[10]; buff_A[11]=(in_11-mean_in[11])/scale_in[11]
    buff_A[12]=(in_12-mean_in[12])/scale_in[12]; buff_A[13]=(in_13-mean_in[13])/scale_in[13]; buff_A[14]=(in_14-mean_in[14])/scale_in[14]; buff_A[15]=(in_15-mean_in[15])/scale_in[15]

    for i in range(256):
        val = b1[i]
        val += buff_A[0]*W1T[i,0] + buff_A[1]*W1T[i,1] + buff_A[2]*W1T[i,2] + buff_A[3]*W1T[i,3]
        val += buff_A[4]*W1T[i,4] + buff_A[5]*W1T[i,5] + buff_A[6]*W1T[i,6] + buff_A[7]*W1T[i,7]
        val += buff_A[8]*W1T[i,8] + buff_A[9]*W1T[i,9] + buff_A[10]*W1T[i,10] + buff_A[11]*W1T[i,11]
        val += buff_A[12]*W1T[i,12] + buff_A[13]*W1T[i,13] + buff_A[14]*W1T[i,14] + buff_A[15]*W1T[i,15]
        buff_B[i] = relu(val)

    for i in range(256):
        val = b2[i]
        for j in range(256): val += buff_B[j] * W2T[i, j]
        buff_A[i] = relu(val)

    for i in range(256):
        val = b3[i]
        for j in range(256): val += buff_A[j] * W3T[i, j]
        buff_B[i] = relu(val)

    for i in range(256):
        val = b4[i]
        for j in range(256): val += buff_B[j] * W4T[i, j]
        buff_A[i] = relu(val)
    
    for i in range(9):
        val = b5[i]
        for j in range(256): val += buff_A[j] * W5T[i, j]
        out_coeffs[i] = (val * scale_out[i]) + mean_out[i]

# =============================================================================
# 3) KERNELS
# =============================================================================
@cuda.jit
def reset_accumulators_kernel(sum_N, sum_U, sum_E, sum_Pij, sum_Q, n_limit):
    idx = cuda.grid(1)
    if idx < n_limit:
        sum_N[idx] = 0.0; sum_E[idx] = 0.0
        sum_U[idx, 0]=0.0; sum_U[idx, 1]=0.0; sum_U[idx, 2]=0.0
        for k in range(6): sum_Pij[idx, k] = 0.0
        for k in range(3): sum_Q[idx, k] = 0.0

@cuda.jit
def reset_stats_kernel(sum_N2, sum_MN, n_limit):
    idx = cuda.grid(1)
    if idx < n_limit: sum_N2[idx]=0.0; sum_MN[idx]=0.0

@cuda.jit
def reset_averages_kernel(avg_rho, avg_U, avg_T, avg_N, n_limit):
    idx = cuda.grid(1)
    if idx < n_limit:
        avg_rho[idx] = 0.0; avg_T[idx] = 0.0; avg_N[idx] = 0.0
        for k in range(3): avg_U[idx, k] = 0.0

@cuda.jit
def accumulate_averages_kernel(rho, U, T, avg_rho, avg_U, avg_T, avg_N, n_limit):
    idx = cuda.grid(1)
    if idx < n_limit:
        if rho[idx] > 0:
            avg_rho[idx] += rho[idx]
            avg_T[idx] += T[idx]
            for k in range(3): avg_U[idx, k] += U[idx, k]
            avg_N[idx] += 1.0

@cuda.jit
def finalize_averages_kernel(avg_rho, avg_U, avg_T, avg_N, n_limit):
    idx = cuda.grid(1)
    if idx < n_limit:
        if avg_N[idx] > 0:
            inv_n = 1.0 / avg_N[idx]
            avg_rho[idx] *= inv_n
            avg_T[idx] *= inv_n
            for k in range(3): avg_U[idx, k] *= inv_n

@cuda.jit
def locate_particles_kernel(x, y, tree_children, tree_center, cell_ids, R_cyl, R_dom):
    i = cuda.grid(1)
    if i < x.shape[0]:
        if math.isnan(x[i]): cell_ids[i] = -1
        else: cell_ids[i] = get_leaf_index(x[i], y[i], tree_children, tree_center, R_cyl, R_dom)

@cuda.jit
def moments_pass1_kernel(vx, vy, vz, cell_id, sum_N, sum_U, sum_E):
    i = cuda.grid(1)
    if i < vx.shape[0]:
        idx = cell_id[i]; 
        if idx < 0: return
        cuda.atomic.add(sum_N, idx, 1.0)
        cuda.atomic.add(sum_U, (idx, 0), float64(vx[i]))
        cuda.atomic.add(sum_U, (idx, 1), float64(vy[i]))
        cuda.atomic.add(sum_U, (idx, 2), float64(vz[i]))
        cuda.atomic.add(sum_E, idx, float64(vx[i]**2 + vy[i]**2 + vz[i]**2))

@cuda.jit
def propagate_kernel(tree_children, tree_depth, sum_N, sum_U, sum_E, sum_Pij, sum_Q, target_d, n_limit):
    idx = cuda.grid(1)
    if idx >= n_limit: return
    if tree_depth[idx] != target_d: return
    if tree_children[idx, 0] == -1: return 
    for k in range(4):
        child = tree_children[idx, k]
        if child == -1: continue
        sum_N[idx] += sum_N[child]; sum_E[idx] += sum_E[child]
        for j in range(3): sum_U[idx, j] += sum_U[child, j]; sum_Q[idx, j] += sum_Q[child, j]
        for j in range(6): sum_Pij[idx, j] += sum_Pij[child, j]

@cuda.jit
def finalize_mean_kernel(rho, U, T, sum_N, sum_U, sum_E, tree_children, tree_size, tree_center, mass, kb, f_num, R_cyl, R_dom):
    idx = cuda.grid(1)
    if idx >= rho.shape[0]: return
    n = sum_N[idx]
    if n < 1.0: rho[idx]=0.0; T[idx]=0.0; return
    inv_n = 1.0/n
    ux = sum_U[idx,0]*inv_n; uy = sum_U[idx,1]*inv_n; uz = sum_U[idx,2]*inv_n
    U[idx,0]=ux; U[idx,1]=uy; U[idx,2]=uz
    v2_avg = sum_E[idx]*inv_n
    var_v = v2_avg - (ux**2+uy**2+uz**2)
    if var_v < 1e-10: var_v = 1e-10
    T[idx] = max((mass/(3.0*kb))*var_v, 10.0)
    vol = calc_cell_volume(tree_size[idx], tree_center[idx], R_cyl, R_dom)
    rho[idx] = (n * f_num * mass) / vol

@cuda.jit
def moments_pass2_kernel(vx, vy, vz, cell_id, U, sum_Pij, sum_Q):
    i = cuda.grid(1)
    if i < vx.shape[0]:
        idx = cell_id[i]
        if idx < 0: return
        ux=float64(U[idx,0]); uy=float64(U[idx,1]); uz=float64(U[idx,2])
        v_x=float64(vx[i]); v_y=float64(vy[i]); v_z=float64(vz[i])
        cx=v_x-ux; cy=v_y-uy; cz=v_z-uz
        c2 = cx**2 + cy**2 + cz**2
        cuda.atomic.add(sum_Pij, (idx,0), cx*cx); cuda.atomic.add(sum_Pij, (idx,1), cx*cy)
        cuda.atomic.add(sum_Pij, (idx,2), cx*cz); cuda.atomic.add(sum_Pij, (idx,3), cy*cy)
        cuda.atomic.add(sum_Pij, (idx,4), cy*cz); cuda.atomic.add(sum_Pij, (idx,5), cz*cz)
        cuda.atomic.add(sum_Q, (idx,0), c2*cx); cuda.atomic.add(sum_Q, (idx,1), c2*cy); cuda.atomic.add(sum_Q, (idx,2), c2*cz)

@cuda.jit
def finalize_higher_kernel(stress, heat, sum_Pij, sum_Q, sum_N, rho, T, mass, kb, tree_children):
    idx = cuda.grid(1)
    if idx >= stress.shape[0]: return
    n = sum_N[idx]
    if n < 2.0 or rho[idx] <= 0: return
    inv_n = 1.0/n
    p = (rho[idx] / mass) * kb * T[idx]
    P_xx = rho[idx] * sum_Pij[idx, 0] * inv_n
    P_xy = rho[idx] * sum_Pij[idx, 1] * inv_n
    P_xz = rho[idx] * sum_Pij[idx, 2] * inv_n
    P_yy = rho[idx] * sum_Pij[idx, 3] * inv_n
    P_yz = rho[idx] * sum_Pij[idx, 4] * inv_n
    P_zz = rho[idx] * sum_Pij[idx, 5] * inv_n
    stress[idx, 0] = -(P_xx - p)
    stress[idx, 1] = -P_xy
    stress[idx, 2] = -P_xz
    stress[idx, 3] = -(P_yy - p)
    stress[idx, 4] = -P_yz
    stress[idx, 5] = -(P_zz - p)
    for j in range(3): heat[idx, j] = 0.5 * rho[idx] * sum_Q[idx, j] * inv_n

@cuda.jit
def update_age_kernel(tree_children, cell_age, n_limit):
    idx = cuda.grid(1)
    if idx >= n_limit: return
    cell_age[idx] += 1

@cuda.jit
def refine_flag_kernel(rho, U, T, tree_children, tree_parent, tree_depth, sum_N, flags, 
                       grad_lim, max_d, max_p, mass, kb, p_inf, min_d, tree_size, tree_center):
    idx = cuda.grid(1)
    if idx >= rho.shape[0]: return
    if tree_children[idx, 0] != -1: return 
    p_idx = tree_parent[idx]
    if p_idx == -1:
        if sum_N[idx] > max_p: flags[idx] = 1
        return
    if tree_depth[idx] < min_d:
        flags[idx] = 1; return
    xi = tree_center[idx, 0]; xi_min = xi - tree_size[idx, 0]
    if xi_min <= 0.05 and tree_depth[idx] < max_d:
        flags[idx] = 1; return
    if rho[idx] <= 0: return
    p_cell = (rho[idx] / mass) * kb * T[idx]
    if p_cell > 1.2 * p_inf and tree_depth[idx] < max_d:
        flags[idx] = 1; return
    if rho[p_idx] > 0:
        p_parent = (rho[p_idx]/ mass) * kb * T[p_idx]
        dT   = abs(T[idx]   - T[p_idx])   / (abs(T[idx])   + 1e-12)
        dRho = abs(rho[idx] - rho[p_idx]) / (abs(rho[idx]) + 1e-30)
        dP   = abs(p_cell   - p_parent)   / (abs(p_cell)   + 1e-30)
        val = dT
        if dP > val: val = dP
        if dRho > val: val = dRho
        if val > grad_lim and tree_depth[idx] < max_d:
            flags[idx] = 1; return
    if sum_N[idx] > max_p and tree_depth[idx] < max_d:
        flags[idx] = 1

@cuda.jit
def refine_apply_kernel(tree_children, tree_parent, tree_center, tree_size, tree_depth, cell_age,
                        flags, next_free, did_refine, max_nodes):
    idx = cuda.grid(1)
    if idx >= max_nodes: return
    if flags[idx] == 1:
        if next_free[0] >= max_nodes - 4: flags[idx] = 0; return
        start = cuda.atomic.add(next_free, 0, 4)
        if start + 4 > max_nodes: flags[idx] = 0; return
        did_refine[0] = 1
        cx = tree_center[idx,0]; cy = tree_center[idx,1]
        hw = tree_size[idx,0]*0.5; hh = tree_size[idx,1]*0.5
        d = tree_depth[idx]
        for k in range(4):
            child = start + k
            tree_children[idx, k] = child
            tree_parent[child] = idx
            tree_children[child, 0] = -1 
            tree_depth[child] = d + 1
            tree_size[child, 0] = hw; tree_size[child, 1] = hh
            dx = -1.0 if (k==0 or k==2) else 1.0
            dy = -1.0 if (k==0 or k==1) else 1.0
            tree_center[child, 0] = cx + dx*hw
            tree_center[child, 1] = cy + dy*hh
            cell_age[child] = 0
        flags[idx] = 0

@cuda.jit
def mark_coarsen_parents(tree_children, sum_N, rho, T, cell_age, coarsen_parent, n_limit, grad_lim, mass, kb, freeze_steps, tree_depth, min_d, tree_size, tree_center):
    i = cuda.grid(1)
    if i >= n_limit: return
    if tree_children[i, 0] == -1: return
    if tree_depth[i] < min_d: return 
    ok = 1
    rho_min = 1.0e20; rho_max = -1.0
    p_min = 1.0e20;   p_max = -1.0
    for k in range(4):
        c = tree_children[i, k]
        if c < 0: ok = 0; break
        if tree_children[c, 0] != -1: ok = 0; break 
        xi_min = tree_center[c, 0] - tree_size[c, 0]
        if xi_min <= 0.05: ok = 0; break
        if cell_age[c] < freeze_steps: ok = 0; break
        if sum_N[c] >= 10.0: ok = 0; break 
        if rho[c] > 0:
            pres = (rho[c] / mass) * kb * T[c]
            if rho[c] < rho_min: rho_min = rho[c]
            if rho[c] > rho_max: rho_max = rho[c]
            if pres < p_min: p_min = pres
            if pres > p_max: p_max = pres
    if ok == 1 and rho_max > 0:
        dRho = (rho_max - rho_min) / (rho_max + 1e-30)
        dP   = (p_max - p_min) / (p_max + 1e-30)
        if dRho > grad_lim or dP > grad_lim: ok = 0
    if ok == 1: coarsen_parent[i] = 1

@cuda.jit
def apply_coarsen(tree_children, coarsen_parent, n_limit):
    i = cuda.grid(1)
    if i >= n_limit: return
    if coarsen_parent[i] == 1:
        for k in range(4): tree_children[i, k] = -1
        coarsen_parent[i] = 0

@cuda.jit
def calc_stats_kernel(vx, vy, vz, cell_id, U, T, rho, heat, c_C, c_Gamma, c_Beta, tau_arr, sum_N2, sum_MN, mass, kb):
    i = cuda.grid(1)
    if i < vx.shape[0]:
        idx = cell_id[i]
        if idx < 0: return
        if tau_arr[idx] > 1e15: return
        rt = kb * T[idx] / mass
        ux=U[idx,0]; uy=U[idx,1]; uz=U[idx,2]
        mx=vx[i]-ux; my=vy[i]-uy; mz=vz[i]-uz
        m2 = mx**2 + my**2 + mz**2
        m2_lim = min(m2, 25.0 * rt)
        C = c_C[idx]; Beta = c_Beta[idx]; Gamma = c_Gamma[idx]
        qx = heat[idx, 0]; qy = heat[idx, 1]; qz = heat[idx, 2]
        term_C_x = C[0]*mx + C[3]*my + C[4]*mz
        term_C_y = C[3]*mx + C[1]*my + C[5]*mz
        term_C_z = C[4]*mx + C[5]*my + C[2]*mz
        poly_gamma = m2_lim - 5.0 * rt
        term_G_x = Gamma[0] * poly_gamma
        term_G_y = Gamma[1] * poly_gamma
        term_G_z = Gamma[2] * poly_gamma
        inv_rho = 1.0 / (rho[idx] + 1e-20)
        term_B_x = Beta * (m2_lim * mx - 2.0 * qx * inv_rho)
        term_B_y = Beta * (m2_lim * my - 2.0 * qy * inv_rho)
        term_B_z = Beta * (m2_lim * mz - 2.0 * qz * inv_rho)
        Nx = term_C_x + term_G_x + term_B_x
        Ny = term_C_y + term_G_y + term_B_y
        Nz = term_C_z + term_G_z + term_B_z
        N2 = Nx**2 + Ny**2 + Nz**2
        MN = mx*Nx + my*Ny + mz*Nz
        cuda.atomic.add(sum_N2, idx, N2)
        cuda.atomic.add(sum_MN, idx, MN)

@cuda.jit
def calc_alpha_kernel(sum_N, sum_N2, sum_MN, T, tau_arr, alpha_arr, dt, mass, kb, tree_children):
    idx = cuda.grid(1)
    if idx >= sum_N.shape[0]: return
    if tree_children[idx, 0] != -1: return
    n = sum_N[idx]
    if n < 10.0: alpha_arr[idx] = 1.0; return
    tau = tau_arr[idx]
    if tau > 1e15: alpha_arr[idx] = 1.0; return
    tloc = T[idx]
    if tloc <= 0.0: alpha_arr[idx] = 1.0; return
    arg = -dt / tau
    E = math.exp(arg)
    E2 = math.exp(2*arg)
    term1 = tau * (1.0 - E)**2 * (sum_N2[idx] / n)
    term2 = 2.0 * (E - E2) * (sum_MN[idx] / n)
    fac = (mass * tau) / (3.0 * kb * tloc)
    val = 1.0 + fac * (term1 + term2)
    if val <= 1e-6 or math.isnan(val): alpha_arr[idx] = 1.0; return
    alpha_arr[idx] = math.sqrt(val)

@cuda.jit
def update_particles_kernel(vx, vy, vz, cell_id, rng, dt, U, T, rho, heat, tau_arr, alpha_arr, c_C, c_Gamma, c_Beta, mass, kb):
    i = cuda.grid(1)
    if i < vx.shape[0]:
        idx = cell_id[i]
        if idx < 0: return
        tau = tau_arr[idx]
        if tau > 1e15: return
        alpha = alpha_arr[idx]
        rt = kb * T[idx] / mass
        ux=U[idx,0]; uy=U[idx,1]; uz=U[idx,2]
        mx=vx[i]-ux; my=vy[i]-uy; mz=vz[i]-uz
        m2 = mx**2 + my**2 + mz**2
        m2_lim = min(m2, 25.0 * rt)
        C = c_C[idx]; Beta = c_Beta[idx]; Gamma = c_Gamma[idx]
        qx = heat[idx, 0]; qy = heat[idx, 1]; qz = heat[idx, 2]
        term_C_x = C[0]*mx + C[3]*my + C[4]*mz
        term_C_y = C[3]*mx + C[1]*my + C[5]*mz
        term_C_z = C[4]*mx + C[5]*my + C[2]*mz
        poly_gamma = m2_lim - 5.0 * rt
        term_G_x = Gamma[0] * poly_gamma
        term_G_y = Gamma[1] * poly_gamma
        term_G_z = Gamma[2] * poly_gamma
        inv_rho = 1.0 / (rho[idx] + 1e-20)
        term_B_x = Beta * (m2_lim * mx - 2.0 * qx * inv_rho)
        term_B_y = Beta * (m2_lim * my - 2.0 * qy * inv_rho)
        term_B_z = Beta * (m2_lim * mz - 2.0 * qz * inv_rho)
        Nx = term_C_x + term_G_x + term_B_x
        Ny = term_C_y + term_G_y + term_B_y
        Nz = term_C_z + term_G_z + term_B_z
        arg = -dt/tau
        E = math.exp(arg)
        sig = math.sqrt(rt * (1.0 - math.exp(2.0*arg)))
        r1 = xoroshiro128p_normal_float32(rng, i)
        r2 = xoroshiro128p_normal_float32(rng, i)
        r3 = xoroshiro128p_normal_float32(rng, i)
        inv_alpha = 1.0 / alpha
        vx[i] = ux + inv_alpha * (mx*E + (1.0-E)*tau*Nx + sig*r1)
        vy[i] = uy + inv_alpha * (my*E + (1.0-E)*tau*Ny + sig*r2)
        vz[i] = uz + inv_alpha * (mz*E + (1.0-E)*tau*Nz + sig*r3)

@cuda.jit
def move_boundary_kernel(x, y, vx, vy, vz, dt, R_cyl, R_dom, T_wall, mass, kb, rng, surf_acc, d_theta, step_count, warmup_steps):
    i = cuda.grid(1)
    if i < x.shape[0]:
        if math.isnan(x[i]): return
        xn = x[i] + vx[i]*dt; yn = y[i] + vy[i]*dt
        if yn < 0.0: yn = -yn; vy[i] = -vy[i]
        r2 = xn**2 + yn**2
        if r2 < R_cyl**2: 
            r = math.sqrt(r2); nx = xn/r; ny = yn/r
            theta = math.atan2(ny*R_cyl, nx*R_cyl)
            bin_idx = int(theta / d_theta)
            if bin_idx >= surf_acc.shape[0]: bin_idx = surf_acc.shape[0] - 1
            v_n_in = vx[i]*nx + vy[i]*ny
            if v_n_in < 0.0:
                v_t_in = vx[i]*(-ny) + vy[i]*(nx)
                E_in = 0.5*mass*(vx[i]**2 + vy[i]**2 + vz[i]**2)
                vth = math.sqrt(kb * T_wall / mass)
                u1 = max(xoroshiro128p_uniform_float32(rng, i), 1e-12)
                vn = math.sqrt(-2.0 * math.log(u1)) * vth 
                vt1 = xoroshiro128p_normal_float32(rng, i) * vth
                vt2 = xoroshiro128p_normal_float32(rng, i) * vth
                vx[i] = vn*nx + vt1*(-ny); vy[i] = vn*ny + vt1*(nx); vz[i] = vt2
                x[i] = nx*R_cyl*1.00001; y[i] = ny*R_cyl*1.00001
                if step_count >= warmup_steps:
                    cuda.atomic.add(surf_acc, (bin_idx, 0), mass*(vn - v_n_in))
                    cuda.atomic.add(surf_acc, (bin_idx, 1), mass*(v_t_in - vt1))
                    cuda.atomic.add(surf_acc, (bin_idx, 2), E_in - 0.5*mass*(vx[i]**2 + vy[i]**2 + vz[i]**2))
                    cuda.atomic.add(surf_acc, (bin_idx, 3), 1.0)
            else: x[i]=xn; y[i]=yn
        elif r2 > R_dom**2: 
            theta = xoroshiro128p_uniform_float32(rng, i) * 3.14159
            x[i] = R_dom * math.cos(theta) * 0.999; y[i] = R_dom * math.sin(theta) * 0.999
            vt = math.sqrt(kb * T_INF / mass)
            vx[i] = xoroshiro128p_normal_float32(rng, i)*vt + U_INF; vy[i] = xoroshiro128p_normal_float32(rng, i)*vt; vz[i] = xoroshiro128p_normal_float32(rng, i)*vt
        else: x[i]=xn; y[i]=yn

# --- PHYSICS ONLY KERNEL ---
@cuda.jit
def calc_coeffs_physics_kernel(U, T, rho, stress, heat, c_C, c_Gamma, c_Beta, tau_arr, 
                               mass, kb, vis0, t0, omega, pr, tree_children):
    idx = cuda.grid(1)
    if idx >= rho.shape[0]: return
    if tree_children[idx, 0] != -1: return 
    if rho[idx] <= 0: tau_arr[idx] = 1e20; return
    t_val = T[idx]; p_val = (rho[idx]/mass)*kb*t_val
    if p_val < 1e-15: tau_arr[idx] = 1e20; return
    mu = vis0 * (t_val/t0)**omega
    tau = 2.0 * mu / p_val
    tau_arr[idx] = tau
    RT = kb * t_val / mass
    inv_tau = 1.0/tau; inv_p = 1.0/p_val
    c_Beta[idx] = inv_tau * ((1.0-pr)/pr) * (1.0/(10.0*RT*RT))
    c_C[idx,0] = inv_tau*(stress[idx,0]*inv_p); c_C[idx,1] = inv_tau*(stress[idx,3]*inv_p); c_C[idx,2] = inv_tau*(stress[idx,5]*inv_p)
    c_C[idx,3] = inv_tau*(stress[idx,1]*inv_p); c_C[idx,4] = inv_tau*(stress[idx,2]*inv_p); c_C[idx,5] = inv_tau*(stress[idx,4]*inv_p)
    vth = math.sqrt(RT); q_scale = rho[idx]*RT*vth+1e-30; g0 = 0.05*inv_tau*((1.0-pr)/pr)
    c_Gamma[idx,0] = g0*(heat[idx,0]/q_scale)/(RT+1e-30)
    c_Gamma[idx,1] = g0*(heat[idx,1]/q_scale)/(RT+1e-30)
    c_Gamma[idx,2] = g0*(heat[idx,2]/q_scale)/(RT+1e-30)

# --- PREPARE INPUTS FOR ML KERNEL ---
@cuda.jit
def prepare_ml_inputs_kernel(rho, T, U, stress, heat, ml_input_array, tree_children, mass, kb, vis0, t0, omega):
    idx = cuda.grid(1)
    if idx >= rho.shape[0]: return
    if tree_children[idx, 0] != -1: return 
    if rho[idx] <= 0: return
    
    # Calculate Tau for input feature
    t_val = T[idx]
    p_val = (rho[idx]/mass)*kb*t_val
    mu = vis0 * (t_val/t0)**omega
    tau = 2.0 * mu / (p_val + 1e-30)
    nu = 1.0 / tau

    # Fill Input Array (16 Features)
    ml_input_array[idx, 0] = rho[idx]
    ml_input_array[idx, 1] = T[idx]
    ml_input_array[idx, 2] = U[idx, 0]; ml_input_array[idx, 3] = U[idx, 1]; ml_input_array[idx, 4] = U[idx, 2]
    ml_input_array[idx, 5] = stress[idx, 0]; ml_input_array[idx, 6] = stress[idx, 1]; ml_input_array[idx, 7] = stress[idx, 2]
    ml_input_array[idx, 8] = stress[idx, 3]; ml_input_array[idx, 9] = stress[idx, 4]; ml_input_array[idx, 10] = stress[idx, 5]
    ml_input_array[idx, 11] = heat[idx, 0]; ml_input_array[idx, 12] = heat[idx, 1]; ml_input_array[idx, 13] = heat[idx, 2]
    ml_input_array[idx, 14] = T[idx] # DM2 proxy
    ml_input_array[idx, 15] = nu

# --- APPLY ML OUTPUTS KERNEL ---
@cuda.jit
def apply_ml_outputs_kernel(c_C, c_Gamma, c_Beta, tau_arr, ml_output_array, rho, T, mass, kb, vis0, t0, omega, pr, tree_children):
    idx = cuda.grid(1)
    if idx >= rho.shape[0]: return
    if tree_children[idx, 0] != -1: return 
    if rho[idx] <= 0: tau_arr[idx] = 1e20; return

    # Physics Calc for Tau and Beta (Analytical)
    t_val = T[idx]
    p_val = (rho[idx]/mass)*kb*t_val
    if p_val < 1e-15: tau_arr[idx] = 1e20; return
    mu = vis0 * (t_val/t0)**omega
    tau = 2.0 * mu / p_val
    tau_arr[idx] = tau
    RT = kb * t_val / mass
    inv_tau = 1.0/tau
    c_Beta[idx] = inv_tau * ((1.0-pr)/pr) * (1.0/(10.0*RT*RT))

    # Load ML Predictions for C and Gamma
    c_C[idx, 0] = ml_output_array[idx, 0]
    c_C[idx, 1] = ml_output_array[idx, 1]
    c_C[idx, 2] = ml_output_array[idx, 2]
    c_C[idx, 3] = ml_output_array[idx, 3]
    c_C[idx, 4] = ml_output_array[idx, 4]
    c_C[idx, 5] = ml_output_array[idx, 5]
    c_Gamma[idx, 0] = ml_output_array[idx, 6]
    c_Gamma[idx, 1] = ml_output_array[idx, 7]
    c_Gamma[idx, 2] = ml_output_array[idx, 8]

class CubicFPSolver:
    def __init__(self, mode='PHYSICS'):
        self.mode = mode
        print(f"Initializing {self.mode} Solver...")
        self.n_part = INIT_PARTICLES
        self.dt = 1e-8
        self.sampling_time = 0.0
        self._init_physics_constants()
        
        # GPU Arrays
        self.x = cuda.device_array(self.n_part, dtype=np.float32)
        self.y = cuda.device_array(self.n_part, dtype=np.float32)
        self.vx = cuda.device_array(self.n_part, dtype=np.float32)
        self.vy = cuda.device_array(self.n_part, dtype=np.float32)
        self.vz = cuda.device_array(self.n_part, dtype=np.float32)
        self.cell_id = cuda.device_array(self.n_part, dtype=np.int32)
        
        self.tree_children = cuda.device_array((MAX_NODES, 4), dtype=np.int32)
        self.tree_parent = cuda.device_array(MAX_NODES, dtype=np.int32)
        self.tree_center = cuda.device_array((MAX_NODES, 2), dtype=np.float32)
        self.tree_size = cuda.device_array((MAX_NODES, 2), dtype=np.float32)
        self.tree_depth = cuda.device_array(MAX_NODES, dtype=np.int8)
        self.cell_age = cuda.device_array(MAX_NODES, dtype=np.int32)
        self.next_free = cuda.to_device(np.array([1], dtype=np.int32))
        
        self.refine_flags = cuda.device_array(MAX_NODES, dtype=np.int32)
        self.did_refine = cuda.to_device(np.array([0], dtype=np.int32))
        self.coarsen_parent = cuda.device_array(MAX_NODES, dtype=np.int32)
        
        self.rho = cuda.device_array(MAX_NODES, dtype=np.float64)
        self.U = cuda.device_array((MAX_NODES, 3), dtype=np.float64)
        self.T = cuda.device_array(MAX_NODES, dtype=np.float64)
        self.stress = cuda.device_array((MAX_NODES, 6), dtype=np.float32)
        self.heat = cuda.device_array((MAX_NODES, 3), dtype=np.float32)
        
        self.c_C = cuda.device_array((MAX_NODES, 6), dtype=np.float32)
        self.c_Gamma = cuda.device_array((MAX_NODES, 3), dtype=np.float32)
        self.c_Beta = cuda.device_array(MAX_NODES, dtype=np.float32)
        self.tau_arr = cuda.device_array(MAX_NODES, dtype=np.float32)
        self.alpha_arr = cuda.device_array(MAX_NODES, dtype=np.float32)
        
        # Averaging arrays for Tecplot
        self.avg_rho = cuda.device_array(MAX_NODES, dtype=np.float32)
        self.avg_U = cuda.device_array((MAX_NODES, 3), dtype=np.float32)
        self.avg_T = cuda.device_array(MAX_NODES, dtype=np.float32)
        self.avg_N = cuda.device_array(MAX_NODES, dtype=np.float32)

        self.sum_N = cuda.device_array(MAX_NODES, dtype=np.float64)
        self.sum_U = cuda.device_array((MAX_NODES, 3), dtype=np.float64)
        self.sum_E = cuda.device_array(MAX_NODES, dtype=np.float64)
        self.sum_Pij = cuda.device_array((MAX_NODES, 6), dtype=np.float64)
        self.sum_Q = cuda.device_array((MAX_NODES, 3), dtype=np.float64)
        self.sum_N2 = cuda.device_array(MAX_NODES, dtype=np.float64)
        self.sum_MN = cuda.device_array(MAX_NODES, dtype=np.float64)
        
        self.surf_acc = cuda.device_array((SURF_BINS, 4), dtype=np.float64)
        self.rng = create_xoroshiro128p_states(self.n_part, seed=42)
        
        self._init_tree()
        self._init_particles()
        
        # --- ML SPECIFIC BUFFERS FOR CUPY ---
        self.ml_input = None
        self.ml_output = None
        if self.mode == 'ML':
            # Create Numba Device Arrays for IO
            self.ml_input = cuda.device_array((MAX_NODES, 16), dtype=np.float32)
            self.ml_output = cuda.device_array((MAX_NODES, 9), dtype=np.float32)
            self._load_dnn_weights_cupy()
        
        self.total_compute_time = 0.0

    def _init_physics_constants(self):
        v_mean = math.sqrt(8.0 * KB * T_INF / (math.pi * MASS))
        mu_inf = 0.5 * RHO_INF * v_mean * LAMBDA_INF
        self.vis_ref = mu_inf * (T_REF / T_INF)**OMEGA

    def _init_tree(self):
        c = np.zeros((1,2), dtype=np.float32); c[0] = [0.5, 0.5]
        s = np.zeros((1,2), dtype=np.float32); s[0] = [0.5, 0.5]
        self.tree_center[:1].copy_to_device(c)
        self.tree_size[:1].copy_to_device(s)
        init_children = -1 * np.ones((MAX_NODES, 4), dtype=np.int32); self.tree_children.copy_to_device(init_children)
        self.tree_children[0,:] = -1 
        init_parent = -1 * np.ones(MAX_NODES, dtype=np.int32)
        self.tree_parent.copy_to_device(init_parent)
        init_depth = -1 * np.ones(MAX_NODES, dtype=np.int8); init_depth[0] = 0; self.tree_depth.copy_to_device(init_depth)
        self.cell_age.copy_to_device(np.full(MAX_NODES, 999999, dtype=np.int32))

    def _init_particles(self):
        u = np.random.rand(self.n_part).astype(np.float32)
        r = np.sqrt(u*(R_DOM**2 - R_CYL**2) + R_CYL**2).astype(np.float32)
        th = np.random.uniform(0, np.pi, self.n_part).astype(np.float32)
        self.x.copy_to_device((r * np.cos(th)).astype(np.float32))
        self.y.copy_to_device((r * np.sin(th)).astype(np.float32))
        vt = math.sqrt(KB * T_INF / MASS)
        self.vx.copy_to_device(np.random.normal(U_INF, vt, self.n_part).astype(np.float32))
        self.vy.copy_to_device(np.random.normal(0.0, vt, self.n_part).astype(np.float32))
        self.vz.copy_to_device(np.random.normal(0.0, vt, self.n_part).astype(np.float32))
        self.f_num = (RHO_INF / MASS) * (0.5 * np.pi * (R_DOM**2 - R_CYL**2)) / self.n_part

    def _load_dnn_weights_cupy(self):
        print("Loading ML Weights into CuPy...")
        d = np.load(MODEL_PARAMS_FILE)
        # Load weights directly to CuPy arrays
        self.cp_W1 = cp.asarray(d['W1'].astype(np.float32)) # (16, 256)
        self.cp_W2 = cp.asarray(d['W2'].astype(np.float32))
        self.cp_W3 = cp.asarray(d['W3'].astype(np.float32))
        self.cp_W4 = cp.asarray(d['W4'].astype(np.float32))
        self.cp_W5 = cp.asarray(d['W5'].astype(np.float32))
        self.cp_b1 = cp.asarray(d['b1'].astype(np.float32))
        self.cp_b2 = cp.asarray(d['b2'].astype(np.float32))
        self.cp_b3 = cp.asarray(d['b3'].astype(np.float32))
        self.cp_b4 = cp.asarray(d['b4'].astype(np.float32))
        self.cp_b5 = cp.asarray(d['b5'].astype(np.float32))
        # Scalers
        self.cp_mean_in = cp.asarray(d['mean_in'].astype(np.float32))
        self.cp_scale_in = cp.asarray(d['scale_in'].astype(np.float32))
        self.cp_mean_out = cp.asarray(d['mean_out'].astype(np.float32))
        self.cp_scale_out = cp.asarray(d['scale_out'].astype(np.float32))

    def compute_moments(self):
        n_act = min(int(self.next_free.copy_to_host()[0]), MAX_NODES)
        blk_n = (n_act+127)//128; tp=128
        blk_p = (self.n_part+127)//128
        
        # --- FIX 1: Reset ALL accumulators (MAX_NODES) ---
        blk_max = (MAX_NODES+127)//128
        reset_accumulators_kernel[blk_max, tp](self.sum_N, self.sum_U, self.sum_E, self.sum_Pij, self.sum_Q, MAX_NODES)
        cuda.synchronize()
        
        locate_particles_kernel[blk_p, tp](self.x, self.y, self.tree_children, self.tree_center, self.cell_id, R_CYL, R_DOM)
        moments_pass1_kernel[blk_p, tp](self.vx, self.vy, self.vz, self.cell_id, self.sum_N, self.sum_U, self.sum_E)
        for d in range(MAX_DEPTH-1, -1, -1): propagate_kernel[blk_n, tp](self.tree_children, self.tree_depth, self.sum_N, self.sum_U, self.sum_E, self.sum_Pij, self.sum_Q, d, n_act)
        finalize_mean_kernel[blk_n, tp](self.rho, self.U, self.T, self.sum_N, self.sum_U, self.sum_E, self.tree_children, self.tree_size, self.tree_center, MASS, KB, self.f_num, R_CYL, R_DOM)
        moments_pass2_kernel[blk_p, tp](self.vx, self.vy, self.vz, self.cell_id, self.U, self.sum_Pij, self.sum_Q)
        for d in range(MAX_DEPTH-1, -1, -1): propagate_kernel[blk_n, tp](self.tree_children, self.tree_depth, self.sum_N, self.sum_U, self.sum_E, self.sum_Pij, self.sum_Q, d, n_act)
        finalize_higher_kernel[blk_n, tp](self.stress, self.heat, self.sum_Pij, self.sum_Q, self.sum_N, self.rho, self.T, MASS, KB, self.tree_children)
        cuda.synchronize()

    def update_dt_cfl(self, n_active):
        max_vel = 7000.0 
        current_max_d = MAX_DEPTH 
        min_dx_xi = 0.5 ** current_max_d
        min_dx_phys = (R_DOM - R_CYL) * min_dx_xi
        dt_cfl = CFL_NUM * min_dx_phys / max_vel
        self.dt = dt_cfl
        return dt_cfl

    def step(self, step_count):
        t0 = time.time()
        tp = 128; blk_p = (self.n_part+tp-1)//tp
        n_act = min(int(self.next_free.copy_to_host()[0]), MAX_NODES)
        blk_n = (n_act+tp-1)//tp
        
        current_dt = self.update_dt_cfl(n_act)
        if step_count >= WARMUP_STEPS: self.sampling_time += current_dt

        # Reset averaging accumulators at start of warmup
        if step_count == WARMUP_STEPS:
             reset_averages_kernel[blk_n, tp](self.avg_rho, self.avg_U, self.avg_T, self.avg_N, MAX_NODES)

        update_age_kernel[blk_n, tp](self.tree_children, self.cell_age, n_act)
        move_boundary_kernel[blk_p, tp](self.x, self.y, self.vx, self.vy, self.vz, current_dt, R_CYL, R_DOM, T_WALL, MASS, KB, self.rng, self.surf_acc, np.pi/SURF_BINS, step_count, WARMUP_STEPS)
        cuda.synchronize()
        
        self.compute_moments()

        # Adaptivity
        self.did_refine.copy_to_device(np.array([0], dtype=np.int32))
        refine_flag_kernel[blk_n, tp](self.rho, self.U, self.T, self.tree_children, self.tree_parent, self.tree_depth, self.sum_N, self.refine_flags, GRAD_LIMIT, MAX_DEPTH, REFINE_MAX_PARTICLES, MASS, KB, P_INF, MIN_DEPTH, self.tree_size, self.tree_center)
        refine_apply_kernel[blk_n, tp](self.tree_children, self.tree_parent, self.tree_center, self.tree_size, self.tree_depth, self.cell_age, self.refine_flags, self.next_free, self.did_refine, MAX_NODES)
        cuda.synchronize()
        if int(self.did_refine.copy_to_host()[0]) == 1:
            self.compute_moments()
            n_act = min(int(self.next_free.copy_to_host()[0]), MAX_NODES)
            blk_n = (n_act+tp-1)//tp

        self.coarsen_parent[:] = 0
        mark_coarsen_parents[blk_n, tp](self.tree_children, self.sum_N, self.rho, self.T, self.cell_age, self.coarsen_parent, n_act, COARSEN_GRAD_LIMIT, MASS, KB, FREEZE_STEPS, self.tree_depth, MIN_DEPTH, self.tree_size, self.tree_center)
        apply_coarsen[blk_n, tp](self.tree_children, self.coarsen_parent, n_act)
        cuda.synchronize()
        self.compute_moments()
        n_act = min(int(self.next_free.copy_to_host()[0]), MAX_NODES)
        blk_n = (n_act+tp-1)//tp

        # Accumulate Averages
        if step_count >= WARMUP_STEPS:
            accumulate_averages_kernel[blk_n, tp](self.rho, self.U, self.T, self.avg_rho, self.avg_U, self.avg_T, self.avg_N, n_act)

        if self.mode == 'PHYSICS':
            calc_coeffs_physics_kernel[blk_n, tp](self.U, self.T, self.rho, self.stress, self.heat, self.c_C, self.c_Gamma, self.c_Beta, self.tau_arr, MASS, KB, self.vis_ref, T_REF, OMEGA, PR, self.tree_children)
        else:
            # --- FIX 2: CUPY NATIVE INFERENCE ---
            prepare_ml_inputs_kernel[blk_n, tp](self.rho, self.T, self.U, self.stress, self.heat, self.ml_input, self.tree_children, MASS, KB, self.vis_ref, T_REF, OMEGA)
            cuda.synchronize()
            
            X_cp = cp.asarray(self.ml_input)[:n_act] 
            
            # Normalize & Forward Pass
            # STAGE12A_AUGMENT_24_TO_34_BEGIN
# If the loaded Stage12A model expects 34 inputs but the legacy online solver
# builds only the 24 base inputs, append the same geometry/front/stagnation
# feature family used in Stage12A training. This is still pure ML:
# no physics fallback, no clipping/projection, no hybrid closure.
try:
    if int(self.cp_mean_in.shape[0]) == 34 and int(X_cp.shape[1]) == 24:
        _pi = np.float32(3.141592653589793)
        _eps = np.float32(1.0e-20)

        # Base Stage12A columns from the existing 24-vector:
        # 16 low-order features + log_rho + log_T + s_raw + stress_norm + heat_norm
        # + r_over_R + cos(theta) + sin(theta)
        _log_rho = X_cp[:, 16]
        _log_T   = X_cp[:, 17]
        _rR      = X_cp[:, 21]
        _cth     = X_cp[:, 22]
        _sth     = X_cp[:, 23]

        _theta = cp.arctan2(_sth, _cth)
        _theta = cp.where(_theta < np.float32(0.0), _theta + np.float32(2.0) * _pi, _theta)

        # Geometric reconstruction matching the 4-column geom convention:
        # geom0=x/R, geom1=y/R, geom2=r/R, geom3=theta.
        _geom0 = _rR * _cth
        _geom1 = _rR * _sth
        _geom2 = _rR
        _geom3 = _theta

        _sin_geom_last = cp.sin(_geom3)
        _cos_geom_last = cp.cos(_geom3)

        _dtheta = cp.arctan2(cp.sin(_theta - _pi), cp.cos(_theta - _pi))
        _abs_front_angle = cp.abs(_dtheta)

        # Smooth shock/front proxy used only as a learned input feature.
        # Centered on the bow-shock/stagnation band found in the diagnostics.
        _front_gate = cp.exp(-np.float32(0.5) * (_abs_front_angle / np.float32(0.65)) ** 2)
        _shock_radial = cp.exp(-np.float32(0.5) * ((_rR - np.float32(1.15)) / np.float32(0.065)) ** 2)
        _shock_indicator_explicit = _front_gate * _shock_radial

        # Front stagnation proxy: high near theta=pi and r/R around the shock/stagnation band.
        _front_stag_proxy = cp.exp(-np.float32(0.5) * (_abs_front_angle / np.float32(0.45)) ** 2) * \
                            cp.exp(-np.float32(0.5) * ((_rR - np.float32(1.10)) / np.float32(0.18)) ** 2)

        # Entropy-proxy-like local feature from log-rho/log-T.
        _local_sstar_like = np.float32(1.5) * _log_T - _log_rho

        _extra = cp.stack([
            _shock_indicator_explicit,
            _geom0,
            _geom1,
            _geom2,
            _geom3,
            _sin_geom_last,
            _cos_geom_last,
            _abs_front_angle,
            _front_stag_proxy,
            _local_sstar_like,
        ], axis=1).astype(cp.float32)

        X_cp = cp.concatenate([X_cp.astype(cp.float32), _extra], axis=1)

        if X_cp.shape[1] != self.cp_mean_in.shape[0]:
            raise RuntimeError(
                "Stage12A online feature augmentation failed: "
                f"X_cp has {X_cp.shape[1]} columns but mean_in has {self.cp_mean_in.shape[0]}"
            )
except Exception as _stage12A_exc:
    print("[Stage12A] online 24-to-34 feature augmentation failed:", repr(_stage12A_exc))
    raise
# STAGE12A_AUGMENT_24_TO_34_END
        X_norm = (X_cp - self.cp_mean_in) / self.cp_scale_in
            H1 = cp.maximum(0, cp.matmul(X_norm, self.cp_W1) + self.cp_b1)
            H2 = cp.maximum(0, cp.matmul(H1, self.cp_W2) + self.cp_b2)
            H3 = cp.maximum(0, cp.matmul(H2, self.cp_W3) + self.cp_b3)
            H4 = cp.maximum(0, cp.matmul(H3, self.cp_W4) + self.cp_b4)
            Y_out = cp.matmul(H4, self.cp_W5) + self.cp_b5
            Y_final = Y_out * self.cp_scale_out + self.cp_mean_out
            
            cp.asarray(self.ml_output)[:n_act] = Y_final
            
            apply_ml_outputs_kernel[blk_n, tp](self.c_C, self.c_Gamma, self.c_Beta, self.tau_arr, self.ml_output, self.rho, self.T, MASS, KB, self.vis_ref, T_REF, OMEGA, PR, self.tree_children)
        
        reset_stats_kernel[blk_n, tp](self.sum_N2, self.sum_MN, n_act)
        calc_stats_kernel[blk_p, tp](self.vx, self.vy, self.vz, self.cell_id, self.U, self.T, self.rho, self.heat, self.c_C, self.c_Gamma, self.c_Beta, self.tau_arr, self.sum_N2, self.sum_MN, MASS, KB)
        calc_alpha_kernel[blk_n, tp](self.sum_N, self.sum_N2, self.sum_MN, self.T, self.tau_arr, self.alpha_arr, current_dt, MASS, KB, self.tree_children)
        
        update_particles_kernel[blk_p, tp](self.vx, self.vy, self.vz, self.cell_id, self.rng, current_dt, self.U, self.T, self.rho, self.heat, self.tau_arr, self.alpha_arr, self.c_C, self.c_Gamma, self.c_Beta, MASS, KB)
        cuda.synchronize()
        
        self.total_compute_time += (time.time() - t0)

    def save_tecplot_field(self, filename):
        print(f"Saving Tecplot Field to {filename}...")
        n_max = int(self.next_free.copy_to_host()[0])
        n_max = min(n_max, MAX_NODES)
        
        # Finalize averages before saving
        blk_nodes = (n_max + 127) // 128
        finalize_averages_kernel[blk_nodes, 128](self.avg_rho, self.avg_U, self.avg_T, self.avg_N, n_max)
        cuda.synchronize()
        
        # Copy data to host
        h_rho = self.avg_rho.copy_to_host()[:n_max]
        h_U = self.avg_U.copy_to_host()[:n_max]
        h_T = self.avg_T.copy_to_host()[:n_max]
        h_stress = self.stress.copy_to_host()[:n_max]
        h_heat = self.heat.copy_to_host()[:n_max]
        h_C = self.tree_center.copy_to_host()[:n_max]
        h_S = self.tree_size.copy_to_host()[:n_max]
        h_Ch = self.tree_children.copy_to_host()[:n_max]
        
        # Filter leaves
        leaves = (h_Ch[:,0] == -1) & (h_rho > 0)
        leaf_indices = np.where(leaves)[0]
        final_indices = [idx for idx in leaf_indices if h_C[idx, 1] <= 1.01] # Clip domain height if needed
        num_leaves = len(final_indices)
        
        with open(filename, 'w') as f:
            f.write(f'TITLE="{filename}"\n')
            f.write('VARIABLES = "X", "Y", "Rho", "U", "V", "W", "T", "Mach", "Pxx", "Pxy", "Pxz", "Pyy", "Pyz", "Pzz", "Qx", "Qy", "Qz"\n')
            f.write(f'ZONE T="Leaves", N={4*num_leaves}, E={num_leaves}, F=FEPOINT, ET=QUADRILATERAL\n')
            
            for idx in final_indices:
                cx = h_C[idx, 0]; cy = h_C[idx, 1]
                hw = h_S[idx, 0]; hh = h_S[idx, 1]
                
                # Macroscopic props
                r = h_rho[idx]; t = h_T[idx]
                u = h_U[idx,0]; v = h_U[idx,1]; w = h_U[idx,2]
                
                # Mach
                vel_mag = math.sqrt(u**2 + v**2 + w**2)
                R_gas = KB / MASS
                gamma = 5.0/3.0
                sound_speed = math.sqrt(gamma * R_gas * t) if t > 0 else 1.0
                mach = vel_mag / sound_speed
                
                # Higher order
                pxx = h_stress[idx,0]; pxy = h_stress[idx,1]; pxz = h_stress[idx,2]
                pyy = h_stress[idx,3]; pyz = h_stress[idx,4]; pzz = h_stress[idx,5]
                qx = h_heat[idx,0]; qy = h_heat[idx,1]; qz = h_heat[idx,2]
                
                # Write 4 corners
                corners_logical = [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]
                for (xi, eta) in corners_logical:
                    rad = R_CYL + xi * (R_DOM - R_CYL)
                    theta = eta * math.pi
                    x_node = rad * math.cos(theta)
                    y_node = rad * math.sin(theta)
                    
                    f.write(f"{x_node:.6e} {y_node:.6e} {r:.6e} {u:.6e} {v:.6e} {w:.6e} {t:.6e} {mach:.6e} "
                            f"{pxx:.6e} {pxy:.6e} {pxz:.6e} {pyy:.6e} {pyz:.6e} {pzz:.6e} "
                            f"{qx:.6e} {qy:.6e} {qz:.6e}\n")
                            
            # Connectivity
            for i in range(num_leaves):
                base = i * 4
                f.write(f"{base+1} {base+2} {base+3} {base+4}\n")

    def get_surface_data(self):
        s_data = self.surf_acc.copy_to_host()
        d_theta = np.pi / SURF_BINS
        area_bin = R_CYL * d_theta
        fac = (self.f_num / (max(self.sampling_time, 1e-9) * area_bin)) * 2.0
        
        theta_sim = []; Cp = []; Cf = []; Ch = []
        for j in range(SURF_BINS):
            theta_rad = (j + 0.5) * d_theta
            deg = theta_rad * 180.0 / np.pi
            P_w = s_data[j, 0] * fac 
            Tau_w = s_data[j, 1] * fac 
            q_w = s_data[j, 2] * fac
            P_inf = RHO_INF * (KB/MASS) * T_INF
            theta_sim.append(deg)
            Cp.append((P_w - P_inf) / Q_DYN_REF)
            Cf.append(-1.0 * Tau_w / Q_DYN_REF)
            Ch.append(q_w / Q_HEAT_REF)
        return np.array(theta_sim), np.array(Cp), np.array(Cf), np.array(Ch)

def plot_mesh_png(solver, filename="mesh.png"):
    print(f"Generating Mesh PNG ({filename})...")
    try:
        n_max = int(solver.next_free.copy_to_host()[0])
        n_max = min(n_max, MAX_NODES)
        h_C = solver.tree_center.copy_to_host()[:n_max]
        h_S = solver.tree_size.copy_to_host()[:n_max]
        h_Ch = solver.tree_children.copy_to_host()[:n_max]
        h_rho = solver.rho.copy_to_host()[:n_max]
        leaves = (h_Ch[:,0] == -1) & (h_rho > 0)
        leaf_indices = np.where(leaves)[0]
        dr = R_DOM - R_CYL
        verts = []
        for idx in leaf_indices:
            if h_C[idx, 1] > 1.01: continue
            cx = h_C[idx, 0]; cy = h_C[idx, 1]
            hw = h_S[idx, 0]; hh = h_S[idx, 1]
            corners_log = [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]
            poly = []
            for (xi, eta) in corners_log:
                r = R_CYL + xi * dr
                theta = eta * math.pi
                poly.append((r * math.cos(theta), r * math.sin(theta)))
            verts.append(poly)
        fig, ax = plt.subplots(figsize=(10, 5))
        coll = PolyCollection(verts, edgecolors='black', facecolors='none', linewidths=0.5)
        ax.add_collection(coll)
        ax.set_aspect('equal')
        ax.set_xlim(-R_DOM, R_DOM)
        ax.set_ylim(0, R_DOM)
        ax.set_title(f"Mesh Visualization ({filename})", fontsize=16)
        plt.savefig(filename)
        print(f"Mesh saved as '{filename}'")
        plt.close()
    except Exception as e:
        print(f"Mesh plotting failed: {e}")

if __name__ == "__main__":
    print(">>> Starting SEQUENTIAL Comparison (Fully Fixed + Tecplot Export) <<<")
    
    # 1. Run Physics
    print("\n--- Running PHYSICS Solver ---")
    solver_phy = CubicFPSolver(mode='PHYSICS')
    for s in range(TOTAL_STEPS):
        solver_phy.step(s)
        if s % REPORT_INTERVAL == 0:
            n_act = int(solver_phy.next_free.copy_to_host()[0])
            print(f"Physics Step {s}/{TOTAL_STEPS} | Active: {n_act} | dt: {solver_phy.dt:.2e}")
    
    th_p, cp_p, cf_p, ch_p = solver_phy.get_surface_data()
    t_phy = solver_phy.total_compute_time
    
    # Export Physics Data
    solver_phy.save_tecplot_field("FlowField_Physics.dat")
    plot_mesh_png(solver_phy, "mesh_physics.png")
    
    del solver_phy
    gc.collect()
    
    # 2. Run ML
    print("\n--- Running ML Solver ---")
    solver_ml = CubicFPSolver(mode='ML')
    for s in range(TOTAL_STEPS):
        solver_ml.step(s)
        if s % REPORT_INTERVAL == 0:
            n_act = int(solver_ml.next_free.copy_to_host()[0])
            print(f"ML Step {s}/{TOTAL_STEPS} | Active: {n_act} | dt: {solver_ml.dt:.2e}")
            
    th_m, cp_m, cf_m, ch_m = solver_ml.get_surface_data()
    t_ml = solver_ml.total_compute_time
    
    # Export ML Data
    solver_ml.save_tecplot_field("FlowField_ML.dat")
    plot_mesh_png(solver_ml, "mesh_ml.png")
    
    print("-" * 40)
    print(f"Physics Time: {t_phy:.2f} s")
    print(f"ML Time:      {t_ml:.2f} s")
    print(f"Speedup:      {t_phy / t_ml:.2f}x")
    print("-" * 40)
    
    # Plotting
    th_plot_p = np.abs(th_p - 180.0); idx_p = np.argsort(th_plot_p)
    th_plot_m = np.abs(th_m - 180.0); idx_m = np.argsort(th_plot_m)
    
    plt.rcParams.update({'font.size': 22})
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    axes[0].plot(th_plot_p[idx_p], cp_p[idx_p], 'k-', linewidth=3, label='Physics')
    axes[0].plot(th_plot_m[idx_m], cp_m[idx_m], 'r--', linewidth=3, label='ML Surrogate')
    axes[0].set_ylabel(r'$C_p$', fontsize=22)
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(th_plot_p[idx_p], cf_p[idx_p], 'k-', linewidth=3)
    axes[1].plot(th_plot_m[idx_m], cf_m[idx_m], 'r--', linewidth=3)
    axes[1].set_ylabel(r'$C_f$', fontsize=22)
    axes[1].grid(True)
    
    axes[2].plot(th_plot_p[idx_p], ch_p[idx_p], 'k-', linewidth=3)
    axes[2].plot(th_plot_m[idx_m], ch_m[idx_m], 'r--', linewidth=3)
    axes[2].set_ylabel(r'$C_h$', fontsize=22)
    axes[2].set_xlabel(r'$\theta$ [deg]', fontsize=22)
    axes[2].set_xlim([0, 180])
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('Comparison_Result_Fixed.png')
    print("Comparison plot saved to 'Comparison_Result_Fixed.png'")