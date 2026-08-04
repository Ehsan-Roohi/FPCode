import numpy as np
import math
import time
import warnings
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import pandas as pd
from numba import cuda, float64
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_normal_float32,
    xoroshiro128p_uniform_float32
)
from numba.core.errors import NumbaPerformanceWarning

# =============================================================================
# 1) CONFIGURATION
# =============================================================================
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# Physics
MASS = 6.63e-26
KB = 1.38e-23
T_INF = 200.0
T_WALL = 500.0
U_INF = 2624.0
RHO_INF = 2.816e-5
LAMBDA_INF = 3.048e-3 

# Model (Maxwell)
OMEGA = 0.734 
PR = 0.667
T_REF = 200.0 

# Geometry
DIAMETER = 0.3048
R_CYL = DIAMETER / 2.0
R_DOM = 1.5 * DIAMETER  

# Reference
Q_DYN_REF = 0.5 * RHO_INF * (U_INF**2)
Q_HEAT_REF = 0.5 * RHO_INF * (U_INF**3)
P_INF = (RHO_INF / MASS) * KB * T_INF 

# Solver Settings (Updated for Finer Overall Mesh)
MAX_NODES = 2000000        # افزایش بافر نودها (چون تعداد سلول‌ها زیاد می‌شود)
MAX_DEPTH = 9              # افزایش ماکزیمم عمق (برای دقت بیشتر در شوک)
MIN_DEPTH = 5              # <--- تغییر اصلی: اجبار به ریز شدن در همه جا
INIT_PARTICLES = 6000000   # افزایش ذرات (چون تعداد سلول‌ها زیاد شده، ذرات بیشتری لازم است)  

# --- ADAPTIVE SETTINGS ---
GRAD_LIMIT = 0.10          
COARSEN_GRAD_LIMIT = 0.05  
REFINE_MAX_PARTICLES = 5000 
FREEZE_STEPS = 2000        

SURF_BINS = 90             
TOTAL_STEPS = 50000        
WARMUP_STEPS = 5000       

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

# --- FIX: FLOAT64 PRECISION FOR MOMENTS ---
@cuda.jit
def moments_pass1_kernel(vx, vy, vz, cell_id, sum_N, sum_U, sum_E):
    i = cuda.grid(1)
    if i < vx.shape[0]:
        idx = cell_id[i]
        if idx < 0: return
        
        # Explicit Cast to preserve precision for T calculation
        v_x = float64(vx[i])
        v_y = float64(vy[i])
        v_z = float64(vz[i])
        
        cuda.atomic.add(sum_N, idx, 1.0)
        cuda.atomic.add(sum_U, (idx, 0), v_x)
        cuda.atomic.add(sum_U, (idx, 1), v_y)
        cuda.atomic.add(sum_U, (idx, 2), v_z)
        cuda.atomic.add(sum_E, idx, v_x*v_x + v_y*v_y + v_z*v_z)

@cuda.jit
def moments_pass2_kernel(vx, vy, vz, cell_id, U, sum_Pij, sum_Q):
    i = cuda.grid(1)
    if i < vx.shape[0]:
        idx = cell_id[i]
        if idx < 0: return
        
        ux=float64(U[idx,0]); uy=float64(U[idx,1]); uz=float64(U[idx,2])
        v_x = float64(vx[i]); v_y = float64(vy[i]); v_z = float64(vz[i])
        
        cx=v_x-ux; cy=v_y-uy; cz=v_z-uz
        c2 = cx**2 + cy**2 + cz**2
        
        cuda.atomic.add(sum_Pij, (idx,0), cx*cx); cuda.atomic.add(sum_Pij, (idx,1), cx*cy)
        cuda.atomic.add(sum_Pij, (idx,2), cx*cz); cuda.atomic.add(sum_Pij, (idx,3), cy*cy)
        cuda.atomic.add(sum_Pij, (idx,4), cy*cz); cuda.atomic.add(sum_Pij, (idx,5), cz*cz)
        cuda.atomic.add(sum_Q, (idx,0), c2*cx); cuda.atomic.add(sum_Q, (idx,1), c2*cy); cuda.atomic.add(sum_Q, (idx,2), c2*cz)

@cuda.jit
def propagate_kernel(tree_children, tree_depth, sum_N, sum_U, sum_E, sum_Pij, sum_Q, target_d, n_limit):
    idx = cuda.grid(1)
    if idx >= n_limit: return
    if tree_depth[idx] != target_d: return
    if tree_children[idx, 0] == -1: return 
    for k in range(4):
        child = tree_children[idx, k]
        if child == -1: continue
        sum_N[idx] += sum_N[child]
        sum_E[idx] += sum_E[child]
        for j in range(3): 
            sum_U[idx, j] += sum_U[child, j]; sum_Q[idx, j] += sum_Q[child, j]
        for j in range(6): 
            sum_Pij[idx, j] += sum_Pij[child, j]

@cuda.jit
def finalize_mean_kernel(rho, U, T, sum_N, sum_U, sum_E, 
                         tree_children, tree_size, tree_center, 
                         mass, kb, f_num, R_cyl, R_dom):
    idx = cuda.grid(1)
    if idx >= rho.shape[0]: return
    n = sum_N[idx]
    if n < 1.0: 
        rho[idx]=0.0; T[idx]=0.0; return
    
    inv_n = 1.0 / n
    ux = sum_U[idx,0]*inv_n; uy = sum_U[idx,1]*inv_n; uz = sum_U[idx,2]*inv_n
    U[idx,0]=ux; U[idx,1]=uy; U[idx,2]=uz
    
    # High Precision Temperature Calculation
    v2_avg = sum_E[idx] * inv_n
    u2 = ux**2 + uy**2 + uz**2
    
    # Check for negative variance due to precision
    var_v = v2_avg - u2
    if var_v < 1e-10: var_v = 1e-10
    
    temp = (mass/(3.0*kb)) * var_v
    if temp < 10.0: temp = 10.0
    T[idx] = temp
    
    vol = calc_cell_volume(tree_size[idx], tree_center[idx], R_cyl, R_dom)
    rho[idx] = (n * f_num * mass) / vol

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
    for j in range(3):
        heat[idx, j] = 0.5 * rho[idx] * sum_Q[idx, j] * inv_n

# =============================================================================
# 4) ADAPTIVE MESH
# =============================================================================
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

    # Surface Protection
    xi = tree_center[idx, 0]; xi_min = xi - tree_size[idx, 0]
    if xi_min <= 0.05 and tree_depth[idx] < max_d:
        flags[idx] = 1; return

    if rho[idx] <= 0: return

    p_cell = (rho[idx] / mass) * kb * T[idx]
    # Shock Sensor
    if p_cell > 1.2 * p_inf and tree_depth[idx] < max_d:
        flags[idx] = 1; return

    # Gradient Logic
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
        
        # Protect Surface
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
        if dRho > grad_lim or dP > grad_lim:
            ok = 0
    
    if ok == 1:
        coarsen_parent[i] = 1

@cuda.jit
def apply_coarsen(tree_children, coarsen_parent, n_limit):
    i = cuda.grid(1)
    if i >= n_limit: return
    if coarsen_parent[i] == 1:
        for k in range(4):
            tree_children[i, k] = -1
        coarsen_parent[i] = 0

# =============================================================================
# 5) PHYSICS KERNELS
# =============================================================================
@cuda.jit
def calc_coeffs_kernel(U, T, rho, stress, heat, c_C, c_Gamma, c_Beta, tau_arr, 
                       mass, kb, vis0, t0, omega, pr, tree_children):
    idx = cuda.grid(1)
    if idx >= rho.shape[0]: return
    if tree_children[idx, 0] != -1: return 
    if rho[idx] <= 0: tau_arr[idx] = 1e20; return
    
    t_val = T[idx]
    p_val = (rho[idx]/mass) * kb * t_val
    if p_val < 1e-15: tau_arr[idx] = 1e20; return
    
    mu = vis0 * (t_val/t0)**omega
    tau = 2.0 * mu / p_val
    tau_arr[idx] = tau
    RT = kb * t_val / mass
    inv_tau = 1.0 / tau
    inv_p = 1.0 / p_val
    
    c_C[idx, 0] = inv_tau * (stress[idx, 0] * inv_p)
    c_C[idx, 1] = inv_tau * (stress[idx, 3] * inv_p)
    c_C[idx, 2] = inv_tau * (stress[idx, 5] * inv_p)
    c_C[idx, 3] = inv_tau * (stress[idx, 1] * inv_p)
    c_C[idx, 4] = inv_tau * (stress[idx, 2] * inv_p)
    c_C[idx, 5] = inv_tau * (stress[idx, 4] * inv_p)
    
    Pr_fac = (1.0 - pr) / pr
    Beta = inv_tau * Pr_fac * (1.0 / (10.0 * RT * RT))
    c_Beta[idx] = Beta
    
    q_x = heat[idx, 0]; q_y = heat[idx, 1]; q_z = heat[idx, 2]
    vth = math.sqrt(RT)
    q_scale = rho[idx] * RT * vth + 1e-30
    g0 = 0.05 * inv_tau * Pr_fac 
    c_Gamma[idx, 0] = g0 * (q_x / q_scale) * (1.0 / (RT + 1e-30))
    c_Gamma[idx, 1] = g0 * (q_y / q_scale) * (1.0 / (RT + 1e-30))
    c_Gamma[idx, 2] = g0 * (q_z / q_scale) * (1.0 / (RT + 1e-30))

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
        m2_lim = m2
        m2_max = 25.0 * rt 
        if m2_lim > m2_max: m2_lim = m2_max
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
        m2_lim = m2
        m2_max = 25.0 * rt
        if m2_lim > m2_max: m2_lim = m2_max
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
def move_boundary_kernel(x, y, vx, vy, vz, dt, R_cyl, R_dom, T_wall, mass, kb, rng, 
                         surf_acc, d_theta, step_count, warmup_steps):
    i = cuda.grid(1)
    if i < x.shape[0]:
        if math.isnan(x[i]): return
        
        xn = x[i] + vx[i]*dt
        yn = y[i] + vy[i]*dt
        
        if yn < 0.0:
            yn = -yn
            vy[i] = -vy[i]
        
        r2 = xn**2 + yn**2
        
        if r2 < R_cyl**2: 
            r = math.sqrt(r2)
            nx = xn/r; ny = yn/r
            x_surf = nx * R_cyl; y_surf = ny * R_cyl
            theta = math.atan2(y_surf, x_surf)
            bin_idx = int(theta / d_theta)
            if bin_idx >= surf_acc.shape[0]: bin_idx = surf_acc.shape[0] - 1

            vx_in = vx[i]; vy_in = vy[i]; vz_in = vz[i]
            v_n_in = vx_in*nx + vy_in*ny
            
            if v_n_in < 0.0:
                v_t_in = vx_in*(-ny) + vy_in*(nx)
                E_in = 0.5*mass*(vx_in**2 + vy_in**2 + vz_in**2)
                vth = math.sqrt(kb * T_wall / mass)
                u1 = max(xoroshiro128p_uniform_float32(rng, i), 1e-12)
                vn = math.sqrt(-2.0 * math.log(u1)) * vth 
                vt1 = xoroshiro128p_normal_float32(rng, i) * vth
                vt2 = xoroshiro128p_normal_float32(rng, i) * vth
                tx = -ny; ty = nx
                vx_out = vn*nx + vt1*tx
                vy_out = vn*ny + vt1*ty
                vz_out = vt2
                vx[i] = vx_out; vy[i] = vy_out; vz[i] = vz_out
                eps = 1e-5 * R_cyl 
                x[i] = x_surf + nx * eps
                y[i] = y_surf + ny * eps
                if step_count >= warmup_steps:
                    dMom_n = mass * (vn - v_n_in)
                    dMom_t = mass * (v_t_in - vt1)
                    dE = E_in - 0.5*mass*(vx_out**2 + vy_out**2 + vz_out**2)
                    cuda.atomic.add(surf_acc, (bin_idx, 0), dMom_n)
                    cuda.atomic.add(surf_acc, (bin_idx, 1), dMom_t)
                    cuda.atomic.add(surf_acc, (bin_idx, 2), dE)
                    cuda.atomic.add(surf_acc, (bin_idx, 3), 1.0)
            else:
                x[i] = xn; y[i] = yn
        elif r2 > R_dom**2: 
            theta = xoroshiro128p_uniform_float32(rng, i) * 3.14159
            x[i] = R_dom * math.cos(theta) * 0.999
            y[i] = R_dom * math.sin(theta) * 0.999
            vt = math.sqrt(kb * T_INF / mass)
            vx[i] = xoroshiro128p_normal_float32(rng, i)*vt + 2624.0 
            vy[i] = xoroshiro128p_normal_float32(rng, i)*vt
            vz[i] = xoroshiro128p_normal_float32(rng, i)*vt
        else:
            x[i] = xn; y[i] = yn

# =============================================================================
# 7) MAIN CLASS
# =============================================================================
class CubicFPSolver:
    def __init__(self):
        self.n_part = INIT_PARTICLES
        self.dt = 1e-7
        self._init_physics_constants()
        
        # Tree Data
        self.tree_children = cuda.device_array((MAX_NODES, 4), dtype=np.int32)
        self.tree_parent = cuda.device_array(MAX_NODES, dtype=np.int32)
        self.tree_center = cuda.device_array((MAX_NODES, 2), dtype=np.float32)
        self.tree_size = cuda.device_array((MAX_NODES, 2), dtype=np.float32)
        self.tree_depth = cuda.device_array(MAX_NODES, dtype=np.int8)
        self.cell_age = cuda.device_array(MAX_NODES, dtype=np.int32)
        
        # Accumulators
        self.sum_N = cuda.device_array(MAX_NODES, dtype=np.float64)
        self.sum_U = cuda.device_array((MAX_NODES, 3), dtype=np.float64)
        self.sum_E = cuda.device_array(MAX_NODES, dtype=np.float64)
        self.sum_Pij = cuda.device_array((MAX_NODES, 6), dtype=np.float64)
        self.sum_Q = cuda.device_array((MAX_NODES, 3), dtype=np.float64)
        self.sum_N2 = cuda.device_array(MAX_NODES, dtype=np.float64)
        self.sum_MN = cuda.device_array(MAX_NODES, dtype=np.float64)
        
        # Physics
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
        
        # Particles
        self.x = cuda.device_array(self.n_part, dtype=np.float32)
        self.y = cuda.device_array(self.n_part, dtype=np.float32)
        self.vx = cuda.device_array(self.n_part, dtype=np.float32)
        self.vy = cuda.device_array(self.n_part, dtype=np.float32)
        self.vz = cuda.device_array(self.n_part, dtype=np.float32)
        self.cell_id = cuda.device_array(self.n_part, dtype=np.int32)
        self.rng = create_xoroshiro128p_states(self.n_part, seed=42)
        
        self.next_free = cuda.to_device(np.array([1], dtype=np.int32))
        # اضافه کردن آرایه‌های گم‌شده برای مدیریت شبکه تطبیقی
        self.refine_flags = cuda.device_array(MAX_NODES, dtype=np.int32)
        self.avg_rho = cuda.device_array(MAX_NODES, dtype=np.float32)
        self.avg_U = cuda.device_array((MAX_NODES, 3), dtype=np.float32)
        self.avg_T = cuda.device_array(MAX_NODES, dtype=np.float32)
        self.avg_N = cuda.device_array(MAX_NODES, dtype=np.float32)
        self.did_refine = cuda.to_device(np.array([0], dtype=np.int32))
        self.coarsen_parent = cuda.device_array(MAX_NODES, dtype=np.int32)
        self.surf_acc = cuda.device_array((SURF_BINS, 4), dtype=np.float64)

        # *** LISTS FOR ML DATA COLLECTION ***
        self.ml_X = [] # Input features
        self.ml_y = [] # Target coefficients

        self._init_tree()
        self._init_particles()

    def _init_physics_constants(self):
        v_mean = math.sqrt(8.0 * KB * T_INF / (math.pi * MASS))
        mu_inf = 0.5 * RHO_INF * v_mean * LAMBDA_INF
        self.vis_ref = mu_inf * (T_REF / T_INF)**OMEGA

    def _init_tree(self):
        c = np.zeros((1,2), dtype=np.float32); c[0] = [0.5, 0.5]
        s = np.zeros((1,2), dtype=np.float32); s[0] = [0.5, 0.5]
        self.tree_center[:1].copy_to_device(c)
        self.tree_size[:1].copy_to_device(s)
        self.tree_children.copy_to_device(np.full((MAX_NODES, 4), -1, dtype=np.int32))
        self.tree_parent.copy_to_device(np.full(MAX_NODES, -1, dtype=np.int32))
        self.tree_depth.copy_to_device(np.full(MAX_NODES, 0, dtype=np.int8))
        self.alpha_arr.copy_to_device(np.ones(MAX_NODES, dtype=np.float32))

    def _init_particles(self):
        u = np.random.rand(self.n_part).astype(np.float32)
        r = np.sqrt(u*(R_DOM**2 - R_CYL**2) + R_CYL**2)
        th = np.random.uniform(0, np.pi, self.n_part)
        self.x.copy_to_device((r * np.cos(th)).astype(np.float32))
        self.y.copy_to_device((r * np.sin(th)).astype(np.float32))
        vt = math.sqrt(KB * T_INF / MASS)
        self.vx.copy_to_device(np.random.normal(U_INF, vt, self.n_part).astype(np.float32))
        self.vy.copy_to_device(np.random.normal(0.0, vt, self.n_part).astype(np.float32))
        self.vz.copy_to_device(np.random.normal(0.0, vt, self.n_part).astype(np.float32))
        vol_phys = 0.5 * np.pi * (R_DOM**2 - R_CYL**2)
        self.f_num = (RHO_INF / MASS) * vol_phys / self.n_part

    def compute_moments(self):
        n_active = min(int(self.next_free.copy_to_host()[0]), MAX_NODES)
        blk_nodes = (MAX_NODES + 127) // 128
        tp = 128
        blk_parts = (self.n_part + tp - 1) // tp
        reset_accumulators_kernel[blk_nodes, tp](self.sum_N, self.sum_U, self.sum_E, self.sum_Pij, self.sum_Q, MAX_NODES)
        cuda.synchronize()
        locate_particles_kernel[blk_parts, tp](self.x, self.y, self.tree_children, self.tree_center, self.cell_id, R_CYL, R_DOM)
        moments_pass1_kernel[blk_parts, tp](self.vx, self.vy, self.vz, self.cell_id, self.sum_N, self.sum_U, self.sum_E)
        for d in range(MAX_DEPTH-1, -1, -1):
            propagate_kernel[blk_nodes, tp](self.tree_children, self.tree_depth, self.sum_N, self.sum_U, self.sum_E, self.sum_Pij, self.sum_Q, d, n_active)
        finalize_mean_kernel[blk_nodes, tp](self.rho, self.U, self.T, self.sum_N, self.sum_U, self.sum_E, self.tree_children, self.tree_size, self.tree_center, MASS, KB, self.f_num, R_CYL, R_DOM)
        moments_pass2_kernel[blk_parts, tp](self.vx, self.vy, self.vz, self.cell_id, self.U, self.sum_Pij, self.sum_Q)
        for d in range(MAX_DEPTH-1, -1, -1):
            propagate_kernel[blk_nodes, tp](self.tree_children, self.tree_depth, self.sum_N, self.sum_U, self.sum_E, self.sum_Pij, self.sum_Q, d, n_active)
        finalize_higher_kernel[blk_nodes, tp](self.stress, self.heat, self.sum_Pij, self.sum_Q, self.sum_N, self.rho, self.T, MASS, KB, self.tree_children)
        cuda.synchronize()

    def step(self, step_count):
        tp = 128
        blk_parts = (self.n_part + tp - 1) // tp
        d_theta = np.pi / SURF_BINS
        n_active = min(int(self.next_free.copy_to_host()[0]), MAX_NODES)
        blk_nodes = (n_active + tp - 1) // tp
        
        if step_count == WARMUP_STEPS:
            reset_averages_kernel[blk_nodes, tp](self.avg_rho, self.avg_U, self.avg_T, self.avg_N, MAX_NODES)

        update_age_kernel[blk_nodes, tp](self.tree_children, self.cell_age, n_active)
        move_boundary_kernel[blk_parts, tp](self.x, self.y, self.vx, self.vy, self.vz, self.dt, R_CYL, R_DOM, T_WALL, MASS, KB, self.rng, self.surf_acc, d_theta, step_count, WARMUP_STEPS)
        cuda.synchronize()
        self.compute_moments()
        
        if step_count >= WARMUP_STEPS:
            accumulate_averages_kernel[blk_nodes, tp](self.rho, self.U, self.T, self.avg_rho, self.avg_U, self.avg_T, self.avg_N, n_active)

        self.did_refine.copy_to_device(np.array([0], dtype=np.int32))
        
        refine_flag_kernel[blk_nodes, tp](self.rho, self.U, self.T, self.tree_children, self.tree_parent, self.tree_depth, self.sum_N, self.refine_flags, GRAD_LIMIT, MAX_DEPTH, REFINE_MAX_PARTICLES, MASS, KB, P_INF, MIN_DEPTH, self.tree_size, self.tree_center)
        refine_apply_kernel[blk_nodes, tp](self.tree_children, self.tree_parent, self.tree_center, self.tree_size, self.tree_depth, self.cell_age, self.refine_flags, self.next_free, self.did_refine, MAX_NODES)
        cuda.synchronize()
        
        if int(self.did_refine.copy_to_host()[0]) == 1:
            self.compute_moments()
            n_active = min(int(self.next_free.copy_to_host()[0]), MAX_NODES)
            blk_nodes = (n_active + tp - 1) // tp

        self.coarsen_parent[:] = 0
        mark_coarsen_parents[blk_nodes, tp](self.tree_children, self.sum_N, self.rho, self.T, self.cell_age, self.coarsen_parent, n_active, COARSEN_GRAD_LIMIT, MASS, KB, FREEZE_STEPS, self.tree_depth, MIN_DEPTH, self.tree_size, self.tree_center)
        apply_coarsen[blk_nodes, tp](self.tree_children, self.coarsen_parent, n_active)
        cuda.synchronize()
        self.compute_moments()
        n_active = min(int(self.next_free.copy_to_host()[0]), MAX_NODES)
        blk_nodes = (n_active + tp - 1) // tp

        calc_coeffs_kernel[blk_nodes, tp](self.U, self.T, self.rho, self.stress, self.heat, self.c_C, self.c_Gamma, self.c_Beta, self.tau_arr, MASS, KB, self.vis_ref, T_REF, OMEGA, PR, self.tree_children)
        reset_stats_kernel[blk_nodes, tp](self.sum_N2, self.sum_MN, n_active)
        calc_stats_kernel[blk_parts, tp](self.vx, self.vy, self.vz, self.cell_id, self.U, self.T, self.rho, self.heat, self.c_C, self.c_Gamma, self.c_Beta, self.tau_arr, self.sum_N2, self.sum_MN, MASS, KB)
        calc_alpha_kernel[blk_nodes, tp](self.sum_N, self.sum_N2, self.sum_MN, self.T, self.tau_arr, self.alpha_arr, self.dt, MASS, KB, self.tree_children)
        update_particles_kernel[blk_parts, tp](self.vx, self.vy, self.vz, self.cell_id, self.rng, self.dt, self.U, self.T, self.rho, self.heat, self.tau_arr, self.alpha_arr, self.c_C, self.c_Gamma, self.c_Beta, MASS, KB)
        cuda.synchronize()
        
    def collect_ml_data(self):
        """
        این تابع سلول‌های برگ را از GPU خوانده و در لیست‌های آموزشی ذخیره می‌کند.
        """
        n_max = int(self.next_free.copy_to_host()[0])
        h_Ch = self.tree_children.copy_to_host()[:n_max]
        h_rho = self.rho.copy_to_host()[:n_max]
        leaves = (h_Ch[:, 0] == -1) & (h_rho > 0)
        leaf_indices = np.where(leaves)[0]
        
        if len(leaf_indices) == 0: return

        # کپی آرایه‌ها از GPU برای استخراج داده
        h_T = self.T.copy_to_host()[:n_max]
        h_U = self.U.copy_to_host()[:n_max]
        h_S = self.stress.copy_to_host()[:n_max]
        h_H = self.heat.copy_to_host()[:n_max]
        h_tau = self.tau_arr.copy_to_host()[:n_max]
        
        # ضرایب (Target)
        h_cC = self.c_C.copy_to_host()[:n_max]
        h_cG = self.c_Gamma.copy_to_host()[:n_max]
        h_cB = self.c_Beta.copy_to_host()[:n_max]

        for idx in leaf_indices:
            rt = (KB * h_T[idx]) / MASS
            dm2 = 3.0 * rt
            nu = 1.0 / (h_tau[idx] + 1e-30)
            
            # ساخت بردار ویژگی (۱۶ مورد مشابه Couette)
            feature = [
                h_rho[idx], h_T[idx], h_U[idx,0], h_U[idx,1], h_U[idx,2],
                h_S[idx,0], h_S[idx,1], h_S[idx,2], h_S[idx,3], h_S[idx,4], h_S[idx,5],
                h_H[idx,0], h_H[idx,1], h_H[idx,2],
                dm2, nu
            ]
            
            # ساخت بردار هدف (۱۰ مورد)
            target = [
                h_cC[idx,0], h_cC[idx,1], h_cC[idx,2], h_cC[idx,3], h_cC[idx,4], h_cC[idx,5],
                h_cG[idx,0], h_cG[idx,1], h_cG[idx,2],
                h_cB[idx]
            ]
            
            self.ml_X.append(feature)
            self.ml_y.append(target)

    def save_flow_field(self):
        print("Saving Flow Field (Averaged Data)...")
        n_max = int(self.next_free.copy_to_host()[0])
        n_max = min(n_max, MAX_NODES)
        blk_nodes = (n_max + 127) // 128
        finalize_averages_kernel[blk_nodes, 128](self.avg_rho, self.avg_U, self.avg_T, self.avg_N, n_max)
        cuda.synchronize()
        h_rho = self.avg_rho.copy_to_host()[:n_max]
        h_U = self.avg_U.copy_to_host()[:n_max]
        h_T = self.avg_T.copy_to_host()[:n_max]
        h_C = self.tree_center.copy_to_host()[:n_max]
        h_S = self.tree_size.copy_to_host()[:n_max]
        h_Ch = self.tree_children.copy_to_host()[:n_max]
        leaves = (h_Ch[:,0] == -1) & (h_rho > 0)
        leaf_indices = np.where(leaves)[0]
        final_indices = []
        for idx in leaf_indices:
            if h_C[idx, 1] <= 1.01: final_indices.append(idx)
        num_leaves = len(final_indices)
        if num_leaves == 0: return

        with open('Cylinder_Flow_Field.dat', 'w') as f:
            f.write('TITLE="Cylinder Flow Field"\n')
            f.write('VARIABLES = "X", "Y", "Rho", "U", "V", "W", "T"\n')
            num_nodes = 4 * num_leaves
            f.write(f'ZONE T="Leaves", N={num_nodes}, E={num_leaves}, F=FEPOINT, ET=QUADRILATERAL\n')
            for idx in final_indices:
                cx = h_C[idx, 0]; cy = h_C[idx, 1]
                hw = h_S[idx, 0]; hh = h_S[idx, 1]
                rho_val = h_rho[idx]; u_val = h_U[idx, 0]; v_val = h_U[idx, 1]; w_val = h_U[idx, 2]; t_val = h_T[idx]
                corners_logical = [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]
                for (xi, eta) in corners_logical:
                    r = R_CYL + xi * (R_DOM - R_CYL)
                    theta = eta * math.pi
                    x_node = r * math.cos(theta); y_node = r * math.sin(theta)
                    f.write(f"{x_node:.6e} {y_node:.6e} {rho_val:.6e} {u_val:.6e} {v_val:.6e} {w_val:.6e} {t_val:.6e}\n")
            for i in range(num_leaves):
                base = i * 4
                f.write(f"{base+1} {base+2} {base+3} {base+4}\n")


    def run(self, steps):
        print(f"Running Physics Solver for {steps} steps to collect ML data...")
        # اطمینان از خالی بودن لیست‌های جمع‌آوری داده در شروع اجرا
        self.ml_X = []
        self.ml_y = []
        
        self.surf_acc.copy_to_device(np.zeros((SURF_BINS, 4), dtype=np.float64))
        
        for s in range(steps):
            self.step(s)
            
            # ۱. گزارش وضعیت (هر ۱۰۰۰ استپ) - مطابق با دستور شما برای گزارش دما
            if s % 1000 == 0:
                n_act = int(self.next_free.copy_to_host()[0])
                T = self.T.copy_to_host()[:n_act]
                N = self.sum_N.copy_to_host()[:n_act]
                Ch = self.tree_children.copy_to_host()[:n_act, 0]
                # سلول‌هایی که برگ هستند و دارای ذره می‌باشند
                valid = (Ch == -1) & (N >= 1) & (T > 1.0)
                if valid.any():
                    T_avg = np.mean(T[valid])
                    T_max = np.max(T[valid])
                else:
                    T_avg = 0.0
                    T_max = 0.0
                print(f"Step {s} | Active Nodes: {n_act} | Avg T: {T_avg:.1f} K | Max T: {T_max:.1f} K")

            # ۲. جمع‌آوری داده برای ماشین لرنینگ (بعد از Warmup و هر ۱۰۰ گام)
            if s > WARMUP_STEPS and s % 100 == 0:
                self.collect_ml_data()
                print(f"--- [ML Data] Collected {len(self.ml_X)} samples so far (Step {s}) ---")

        # ۳. محاسبات و ذخیره‌سازی داده‌های سطح (بعد از اتمام حلقه)
        print("Saving Surface Data...")
        s_data = self.surf_acc.copy_to_host()
        d_theta = np.pi / SURF_BINS
        area_bin = R_CYL * d_theta
        total_time = (steps - WARMUP_STEPS) * self.dt
        # فاکتور تبدیل برای Cp, Cf, Ch
        fac = (self.f_num / (total_time * area_bin)) * 2.0
        
        with open('Cylinder_Full_Surface.dat', 'w') as f:
            f.write('TITLE="Surface Properties"\n')
            f.write('VARIABLES="Theta_deg", "Cp", "Cf", "Ch", "Cf_abs"\n')
            f.write(f'ZONE I={SURF_BINS}, F=POINT\n')
            for j in range(SURF_BINS):
                theta_rad = (j + 0.5) * d_theta
                deg = theta_rad * 180.0 / np.pi
                P_w = s_data[j, 0] * fac 
                Tau_w = s_data[j, 1] * fac 
                q_w = s_data[j, 2] * fac
                P_inf = RHO_INF * (KB/MASS) * T_INF
                Cp = (P_w - P_inf) / Q_DYN_REF
                Cf = -1.0 * Tau_w / Q_DYN_REF
                Ch = q_w / Q_HEAT_REF
                Cf_abs = abs(Tau_w) / Q_DYN_REF
                f.write(f"{deg:.4f} {Cp:.6e} {Cf:.6e} {Ch:.6e} {Cf_abs:.6e}\n")

        # ۴. ذخیره میدان جریان و رسم مش
        self.save_flow_field()
        plot_mesh_png(self)

        # ۵. ذخیره نهایی فایل داده‌های آموزشی ML
        if len(self.ml_X) > 0:
            print(f"Finalizing ML Dataset: Saving {len(self.ml_X)} samples to 'cylinder_training_data.npz'...")
            # تبدیل لیست‌ها به آرایه دان‌پای (NumPy) برای ذخیره‌سازی
            np.savez('cylinder_training_data.npz', X=np.array(self.ml_X), y=np.array(self.ml_y))
            print("File 'cylinder_training_data.npz' successfully generated.")
        else:
            print("Warning: No ML data was collected. Please check if TOTAL_STEPS > WARMUP_STEPS.")

    

def plot_mesh_png(solver):
    print("Generating Mesh PNG (mesh.png)...")
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
        ax.set_title("Adapted Mesh (Half Domain - Shock Protected)")
        plt.savefig("mesh.png")
        print("Mesh saved as 'mesh.png'")
        plt.close()
    except Exception as e:
        print(f"Mesh plotting failed: {e}")

def plot_surface_properties():
    try:
        plt.rcParams.update({'font.size': 22})
        try:
            df = pd.read_csv('Cylinder_Full_Surface.dat', skiprows=2, sep=r'\s+', names=["Theta_sim", "Cp", "Cf", "Ch", "Cf_abs"], engine='python')
        except:
             df = pd.read_csv('Cylinder_Full_Surface.dat', skiprows=3, sep=r'\s+', names=["Theta_sim", "Cp", "Cf", "Ch", "Cf_abs"], engine='python')
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        df["Theta_plot"] = np.abs(df["Theta_sim"] - 180.0)
        df = df.sort_values("Theta_plot")
        window = 7
        df["Cp"] = df["Cp"].rolling(window=window, center=True, min_periods=1).mean()
        df["Cf"] = df["Cf"].rolling(window=window, center=True, min_periods=1).mean()
        df["Ch"] = df["Ch"].rolling(window=window, center=True, min_periods=1).mean()
        theta = df["Theta_plot"].values
        fig, axes = plt.subplots(3, 1, figsize=(14, 20), sharex=True)
        axes[0].plot(theta, df["Cp"].values, 'k-', linewidth=2.5)
        axes[0].set_ylabel(r'$C_p$', fontsize=22)
        axes[0].set_title(r'(a) Pressure Coefficient ($C_p$)', fontsize=22)
        axes[0].grid(True)
        axes[0].set_xlim([0, 180])
        axes[1].plot(theta, df["Cf"].values, 'k-', linewidth=2.5)
        axes[1].set_ylabel(r'$C_f$', fontsize=22)
        axes[1].set_title(r'(b) Friction Coefficient ($C_f$)', fontsize=22)
        axes[1].grid(True)
        axes[2].plot(theta, df["Ch"].values, 'k-', linewidth=2.5)
        axes[2].set_ylabel(r'$C_h$', fontsize=22)
        axes[2].set_xlabel(r'$\theta$ [deg]', fontsize=22)
        axes[2].set_title(r'(c) Heat Transfer Coefficient ($C_h$)', fontsize=22)
        axes[2].set_xlim([0, 180])
        axes[2].grid(True)
        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=22)
        plt.tight_layout()
        plt.savefig('Surface_Properties.png')
        print("Plot saved as 'Surface_Properties.png'")
    except Exception as e:
        print(f"Plotting error: {e}")
            
if __name__ == "__main__":
    solver = CubicFPSolver()
    solver.run(TOTAL_STEPS)
    print("Exporting Particles...")
    x = solver.x.copy_to_host()
    y = solver.y.copy_to_host()
    u = solver.vx.copy_to_host()
    valid = ~np.isnan(x)
    np.savetxt("particles.csv", np.column_stack((x[valid], y[valid], u[valid])), delimiter=",", header="x,y,u", comments='')
    plot_surface_properties()