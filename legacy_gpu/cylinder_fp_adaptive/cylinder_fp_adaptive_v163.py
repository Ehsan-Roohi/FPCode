import numpy as np
import math
import time
import warnings
import matplotlib.pyplot as plt
from numba import cuda, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32
from numba.core.errors import NumbaPerformanceWarning
from mpi4py import MPI

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# =============================================================================
# MPI & DEVICE SETUP
# =============================================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

try:
    num_devices = len(cuda.list_devices())
    if num_devices > 0:
        cuda.select_device(rank % num_devices)
except:
    pass

# =============================================================================
# PHYSICAL PARAMETERS
# =============================================================================
MASS = 6.63e-26          
KB = 1.38e-23            
T_INF = 200.0            
T_WALL = 500.0           
U_INF = 2624.0           
RHO_INF = 2.816e-5       
NUM_DENS_INF = RHO_INF / MASS
P_INF = NUM_DENS_INF * KB * T_INF 
LAMBDA_INF = 3.048e-3    
OMEGA = 0.734            
PR = 2.0/3.0             
RT_INF = KB * T_INF / MASS
MU_INF = LAMBDA_INF * P_INF / math.sqrt(math.pi * RT_INF / 2.0)

# Geometry
DIAMETER = 0.3048        
R_CYL = DIAMETER / 2.0   
X_MIN_DOMAIN = -0.45     
X_MAX_DOMAIN = 0.65      
Y_MAX_DOMAIN = 0.65       
DOMAIN_LEN_X = X_MAX_DOMAIN - X_MIN_DOMAIN
DOMAIN_LEN_Y = Y_MAX_DOMAIN

# Grid Settings
N_CELLS_X = 1500          
N_CELLS_Y = 1000         
N_CELLS_TOTAL = N_CELLS_X * N_CELLS_Y

# Adaptive Settings
DELTA_MAX = 0.10
 
COARSE_LEVEL = 4

# Simulation Settings
GLOBAL_INIT_PARTICLES = 100000000 
LOCAL_PARTICLES = GLOBAL_INIT_PARTICLES // size
LOCAL_CAPACITY = int(LOCAL_PARTICLES * 1.95)

TOTAL_STEPS = 100000      
WARMUP_STEPS = 2000      
CFL_NUM = 0.75            
DT_INIT = 1e-9           
MACRO_UPDATE_FREQ = 10
OUTPUT_INTERVAL = 5000   

Q_DYN = 0.5 * RHO_INF * U_INF**2
Q_HEAT_REF = 0.5 * RHO_INF * U_INF**3

if rank == 0:
    print(f"MPI Run: {size} GPUs (v172: Local Noise Reduction + Unclamped Beta)")
    print(f"Grid: {N_CELLS_X}x{N_CELLS_Y}")
    print(f"Particles Init: {GLOBAL_INIT_PARTICLES}, Capacity per Rank: {LOCAL_CAPACITY}")

# =============================================================================
# DEVICE FUNCTIONS & KERNELS
# =============================================================================
@cuda.jit(device=True)
def solve_9x9_device_stable(A, b, x):
    n = 9
    for i in range(n):
        max_val = abs(A[i, i])
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k, i]) > max_val:
                max_val = abs(A[k, i])
                max_row = k
        if max_row != i:
            tmp_b = b[i]; b[i] = b[max_row]; b[max_row] = tmp_b
            for k in range(i, n):
                tmp_A = A[i, k]; A[i, k] = A[max_row, k]; A[max_row, k] = tmp_A
        pivot = A[i, i]
        if abs(pivot) < 1e-15: 
            x[0] = 1e30 
            return 
        for j in range(i + 1, n):
            factor = A[j, i] / pivot
            b[j] -= factor * b[i]
            for k in range(i, n):
                if k == i: A[j, k] = 0.0
                else: A[j, k] -= factor * A[i, k]
    for i in range(n - 1, -1, -1):
        sum_ax = 0.0
        for j in range(i + 1, n):
            sum_ax += A[i, j] * x[j]
        x[i] = (b[i] - sum_ax) / A[i, i]

@cuda.jit
def zero_1d_kernel(arr):
    i = cuda.grid(1)
    if i < arr.size: arr[i] = 0.0

@cuda.jit
def zero_2d_kernel(arr):
    i = cuda.grid(1)
    if i < arr.shape[0] * arr.shape[1]:
        r = i // arr.shape[1]; c = i % arr.shape[1]
        arr[r, c] = 0.0

@cuda.jit
def zero_surface_arrays_kernel(n_arr, p_arr, tau_arr, q_arr):
    i = cuda.grid(1)
    if i < n_arr.shape[0]:
        n_arr[i] = 0.0; p_arr[i] = 0.0; tau_arr[i] = 0.0; q_arr[i] = 0.0

@cuda.jit
def calc_volume_fraction_kernel(vol_frac, nx, ny, dx, dy, x_min, R_cyl, rng_states):
    idx = cuda.grid(1)
    if idx >= nx * ny: return
    ix = idx % nx; iy = idx // nx
    x_left = x_min + ix * dx; y_bottom = iy * dy 
    n_samples = 50; count_fluid = 0
    x_far = max(abs(x_left), abs(x_left+dx)); y_far = max(abs(y_bottom), abs(y_bottom+dy))
    if x_far**2 + y_far**2 < R_cyl**2: vol_frac[idx] = 0.0; return 
    if x_left**2 + y_bottom**2 > R_cyl**2 and (x_left+dx)**2 + y_bottom**2 > R_cyl**2 and x_left**2 + (y_bottom+dy)**2 > R_cyl**2:
        vol_frac[idx] = 1.0; return 
    for k in range(n_samples):
        rx = x_left + xoroshiro128p_uniform_float32(rng_states, idx) * dx
        ry = y_bottom + xoroshiro128p_uniform_float32(rng_states, idx) * dy
        if (rx**2 + ry**2) >= R_cyl**2: count_fluid += 1
    frac = count_fluid / n_samples
    vol_frac[idx] = max(frac, 0.01)

@cuda.jit
def init_particles_kernel(x, y, vx, vy, vz, cell_ids, active_mask, rng, x_min, x_max, y_max, R_cyl, U_inf, T_inf, mass, kb, n_init):
    i = cuda.grid(1)
    if i >= x.shape[0]: return
    if i < n_init:
        active_mask[i] = 1
        r1 = xoroshiro128p_uniform_float32(rng, i)
        r2 = xoroshiro128p_uniform_float32(rng, i)
        px = x_min + r1 * (x_max - x_min)
        py = 0.0 + r2 * (y_max - 0.0) 
        if (px**2 + py**2) < R_cyl**2:
            theta = r1 * 3.14159 
            px = (R_cyl + 0.001) * math.cos(theta)
            py = (R_cyl + 0.001) * math.sin(theta)
        x[i] = px; y[i] = py
        vth = math.sqrt(kb * T_inf / mass)
        vx[i] = U_inf + xoroshiro128p_normal_float32(rng, i) * vth
        vy[i] = xoroshiro128p_normal_float32(rng, i) * vth
        vz[i] = xoroshiro128p_normal_float32(rng, i) * vth
        cell_ids[i] = -1
    else:
        active_mask[i] = 0
        cell_ids[i] = -2
        x[i] = -1000.0

@cuda.jit
def bin_particles_kernel(x, y, cell_ids, active_mask, x_min, dx, dy, nx, ny, R_cyl):
    i = cuda.grid(1)
    if i >= x.shape[0]: return
    if active_mask[i] == 0: cell_ids[i] = -2; return
    if (x[i]**2 + y[i]**2) < R_cyl**2: cell_ids[i] = -1; return
    idx_x = int((x[i] - x_min) / dx)
    idx_y = int(y[i] / dy)
    if 0 <= idx_x < nx and 0 <= idx_y < ny:
        cell_ids[i] = idx_y * nx + idx_x
    else:
        cell_ids[i] = -2

@cuda.jit
def inject_particles_kernel(x, y, vx, vy, vz, cell_ids, active_mask, rng, dt, x_min, y_max, n_capacity, n_inject, U_inf, T_inf, mass, kb):
    tid = cuda.grid(1)
    if tid >= n_inject: return 
    stride = (n_capacity // n_inject) + 1
    start_idx = (tid * stride) % n_capacity
    for k in range(n_capacity):
        idx = (start_idx + k) % n_capacity
        if active_mask[idx] == 0:
            old_val = cuda.atomic.add(active_mask, idx, 1)
            if old_val == 0:
                vth = math.sqrt(kb * T_inf / mass)
                r_y = xoroshiro128p_uniform_float32(rng, idx)
                x[idx] = x_min + 1e-6; y[idx] = r_y * y_max
                vx[idx] = U_inf + xoroshiro128p_normal_float32(rng, idx) * vth
                vy[idx] = xoroshiro128p_normal_float32(rng, idx) * vth
                vz[idx] = xoroshiro128p_normal_float32(rng, idx) * vth
                cell_ids[idx] = -1
                return 

@cuda.jit
def move_particles_half_domain_kernel(x, y, vx, vy, vz, dt, R_cyl, x_min, x_max, y_max, T_wall, T_inf, U_inf, mass, kb, rng, surf_p, surf_tau, surf_q, surf_n, n_bins, active_mask, cell_ids):
    i = cuda.grid(1)
    if i >= x.shape[0]: return
    if active_mask[i] == 0: return 
    
    if math.isnan(x[i]) or math.isnan(y[i]) or math.isnan(vx[i]):
        active_mask[i] = 0; cell_ids[i] = -2; x[i] = -1000.0; return

    x_curr = x[i]; y_curr = y[i]; vx_curr = vx[i]; vy_curr = vy[i]; vz_curr = vz[i]
    dt_remaining = dt

    if vy_curr < 0:
        t_hit = -y_curr / vy_curr
        if t_hit < dt_remaining:
            x_curr += vx_curr * t_hit; y_curr = 0.0; vy_curr = -vy_curr; dt_remaining -= t_hit

    r2_new = (x_curr + vx_curr*dt_remaining)**2 + (y_curr + vy_curr*dt_remaining)**2
    if r2_new < R_cyl**2:
        a = vx_curr**2 + vy_curr**2; b = 2.0 * (x_curr * vx_curr + y_curr * vy_curr); c = x_curr**2 + y_curr**2 - R_cyl**2
        delta = b*b - 4.0*a*c
        t_cyl = (-b - math.sqrt(delta)) / (2.0*a)
        x_curr += vx_curr * t_cyl; y_curr += vy_curr * t_cyl
        dt_remaining -= t_cyl
        if dt_remaining < 0: dt_remaining = 0.0
        
        r_mag = math.sqrt(x_curr**2 + y_curr**2)
        nx = x_curr / r_mag; ny = y_curr / r_mag
        tx = ny; ty = -nx 
        
        vn_in = vx_curr*nx + vy_curr*ny
        vt_in = vx_curr*tx + vy_curr*ty
        E_in = 0.5 * mass * (vx_curr**2 + vy_curr**2 + vz_curr**2)
        
        vth_w = math.sqrt(kb * T_wall / mass)
        r1 = max(xoroshiro128p_uniform_float32(rng, i), 1e-10)
        vn_out = math.sqrt(-2.0 * math.log(r1)) * vth_w
        vt_out = xoroshiro128p_normal_float32(rng, i) * vth_w
        vz_out = xoroshiro128p_normal_float32(rng, i) * vth_w
        
        vx_curr = vn_out * nx + vt_out * tx
        vy_curr = vn_out * ny + vt_out * ty
        vz_curr = vz_out
        
        x_curr += nx * 1e-5; y_curr += ny * 1e-5
        x_curr += vx_curr * dt_remaining
        y_curr += vy_curr * dt_remaining
        
        E_out = 0.5 * mass * (vx_curr**2 + vy_curr**2 + vz_curr**2)
        raw_ang = math.atan2(abs(ny), nx)
        bin_idx = int(((math.pi - raw_ang) / math.pi) * n_bins)
        if bin_idx >= n_bins: bin_idx = n_bins - 1
        
        cuda.atomic.add(surf_p, bin_idx, mass * (vn_out - vn_in))
        cuda.atomic.add(surf_tau, bin_idx, mass * (vt_in - vt_out))
        cuda.atomic.add(surf_q, bin_idx, E_in - E_out)
        cuda.atomic.add(surf_n, bin_idx, 1.0)
    else:
        x_curr += vx_curr * dt_remaining
        y_curr += vy_curr * dt_remaining

    if x_curr > x_max or y_curr > y_max or x_curr < x_min:
        active_mask[i] = 0; cell_ids[i] = -2; x[i] = -1000.0
    else:
        x[i] = x_curr; y[i] = y_curr; vx[i] = vx_curr; vy[i] = vy_curr; vz[i] = vz_curr

@cuda.jit
def calc_raw_moments_kernel(vx, vy, vz, cell_ids, raw_moments):
    i = cuda.grid(1)
    if i >= vx.shape[0]: return
    idx = cell_ids[i]
    if idx < 0: return 
    cuda.atomic.add(raw_moments, (idx, 0), 1.0)
    cuda.atomic.add(raw_moments, (idx, 1), vx[i])
    cuda.atomic.add(raw_moments, (idx, 2), vy[i])
    cuda.atomic.add(raw_moments, (idx, 3), vz[i])
    cuda.atomic.add(raw_moments, (idx, 4), vx[i]**2 + vy[i]**2 + vz[i]**2)

@cuda.jit
def calc_central_moments_kernel(vx, vy, vz, cell_ids, U, packed_central): 
    i = cuda.grid(1)
    if i >= vx.shape[0]: return
    idx = cell_ids[i]
    if idx < 0: return
    cx = vx[i] - U[idx, 0]; cy = vy[i] - U[idx, 1]; cz = vz[i] - U[idx, 2]
    c2 = cx**2 + cy**2 + cz**2; c4 = c2 * c2
    cuda.atomic.add(packed_central, (idx, 0), cx*cx)
    cuda.atomic.add(packed_central, (idx, 1), cy*cy)
    cuda.atomic.add(packed_central, (idx, 2), cz*cz)
    cuda.atomic.add(packed_central, (idx, 3), cx*cy)
    cuda.atomic.add(packed_central, (idx, 4), cx*cz)
    cuda.atomic.add(packed_central, (idx, 5), cy*cz)
    cuda.atomic.add(packed_central, (idx, 6), c2*cx)
    cuda.atomic.add(packed_central, (idx, 7), c2*cy)
    cuda.atomic.add(packed_central, (idx, 8), c2*cz)
    cuda.atomic.add(packed_central, (idx, 9), cx*cx*cx)
    cuda.atomic.add(packed_central, (idx, 10), cy*cy*cy)
    cuda.atomic.add(packed_central, (idx, 11), cz*cz*cz)
    cuda.atomic.add(packed_central, (idx, 12), cx*cx*cy)
    cuda.atomic.add(packed_central, (idx, 13), cx*cx*cz)
    cuda.atomic.add(packed_central, (idx, 14), cx*cy*cy)
    cuda.atomic.add(packed_central, (idx, 15), cy*cy*cz)
    cuda.atomic.add(packed_central, (idx, 16), cx*cz*cz)
    cuda.atomic.add(packed_central, (idx, 17), cy*cz*cz)
    cuda.atomic.add(packed_central, (idx, 18), cx*cy*cz)
    cuda.atomic.add(packed_central, (idx, 19), c4)
    cuda.atomic.add(packed_central, (idx, 20), c4*cx)
    cuda.atomic.add(packed_central, (idx, 21), c4*cy)
    cuda.atomic.add(packed_central, (idx, 22), c4*cz)
    cuda.atomic.add(packed_central, (idx, 23), c2*cx*cx)
    cuda.atomic.add(packed_central, (idx, 24), c2*cy*cy)
    cuda.atomic.add(packed_central, (idx, 25), c2*cz*cz)
    cuda.atomic.add(packed_central, (idx, 26), c2*cx*cy)
    cuda.atomic.add(packed_central, (idx, 27), c2*cx*cz)
    cuda.atomic.add(packed_central, (idx, 28), c2*cy*cz)

@cuda.jit
def compute_adaptive_coefficients_kernel(packed_central, N_arr, adaptation_map, tau, rho, c_C, c_Gamma, c_Beta, nx, ny):
    idx = cuda.grid(1)
    if idx >= rho.shape[0]: return
    level = adaptation_map[idx]
    if level < 1: return
    ix = idx % nx; iy = idx // nx
    if (ix % level != 0) or (iy % level != 0): return
    start_x = ix; end_x = min(ix + level, nx)
    start_y = iy; end_y = min(iy + level, ny)
    m_N = 0.0; sum_tau_N = 0.0 
    m_M2 = cuda.local.array(6, float64); m_M3v = cuda.local.array(3, float64); m_M3f = cuda.local.array(10, float64)
    m_M4 = 0.0; m_M4v = cuda.local.array(3, float64); m_M2c2 = cuda.local.array(6, float64)
    for k in range(6): m_M2[k] = 0.0; m_M2c2[k] = 0.0
    for k in range(3): m_M3v[k] = 0.0; m_M4v[k] = 0.0
    for k in range(10): m_M3f[k] = 0.0
    for y in range(start_y, end_y):
        for x in range(start_x, end_x):
            nid = y * nx + x
            N_val = N_arr[nid]
            m_N += N_val
            sum_tau_N += tau[nid] * N_val 
            for k in range(6): m_M2[k] += packed_central[nid, 0+k]
            for k in range(3): m_M3v[k] += packed_central[nid, 6+k]
            for k in range(10): m_M3f[k] += packed_central[nid, 9+k]
            m_M4 += packed_central[nid, 19]
            for k in range(3): m_M4v[k] += packed_central[nid, 20+k]
            for k in range(6): m_M2c2[k] += packed_central[nid, 23+k]

    # حد پایین برای محاسبه (در شوک این مقدار مهم است)
    if m_N < 5: 
        for k in range(6): c_C[idx, k] = 0.0
        for k in range(3): c_Gamma[idx, k] = 0.0
        c_Beta[idx] = 0.0
        return

    inv_N = 1.0 / m_N
    tau_eff = sum_tau_N * inv_N
    if tau_eff < 1e-20: tau_eff = 1e-20
    
    u2_0 = m_M2[0] * inv_N; u2_1 = m_M2[1] * inv_N; u2_2 = m_M2[2] * inv_N
    u2_3 = m_M2[3] * inv_N; u2_4 = m_M2[4] * inv_N; u2_5 = m_M2[5] * inv_N
    u3_vec_0 = m_M3v[0] * inv_N; u3_vec_1 = m_M3v[1] * inv_N; u3_vec_2 = m_M3v[2] * inv_N
    u4_scalar = m_M4 * inv_N
    
    tr_u2 = u2_0 + u2_1 + u2_2
    if tr_u2 <= 1e-10:
        for k in range(6): c_C[idx, k] = 0.0
        for k in range(3): c_Gamma[idx, k] = 0.0
        c_Beta[idx] = 0.0
        return

   # --- اصلاح بزرگ (Factor 81 Fix) ---
    # محاسبه RT واقعی (نه Trace)
    RT = tr_u2 / 3.0
    
    # صورت کسر: دترمینان بخش انحرافی (طبق مقاله)
    Pxx = u2_0 - RT; Pyy = u2_1 - RT; Pzz = u2_2 - RT
    Pxy = u2_3; Pxz = u2_4; Pyz = u2_5
    det_dev = (Pxx*(Pyy*Pzz - Pyz**2) - Pxy*(Pxy*Pzz - Pyz*Pxz) + Pxz*(Pxy*Pyz - Pyy*Pxz))
    
    # مخرج کسر: استفاده از RT به جای tr_u2
    # این باعث می‌شود مقدار Lambda به اندازه 3^4 = 81 برابر بزرگتر (قوی‌تر) شود.
    denom = tau_eff * (RT**4) + 1e-30
    
    Lambda = -abs(det_dev) / denom
    
    # لیمیتر ایمنی (برای جلوگیری از NaN در گام‌های اولیه)
    safe_cap = 50000.0 / (tau_eff * RT * RT + 1e-20)
    if abs(Lambda) > safe_cap:
        Lambda = -safe_cap

    c_Beta[idx] = Lambda

    # --- Matrix Solver (با اصلاحاتی که قبلا تایید کردیم) ---
    L = cuda.local.array((9, 9), dtype=float64); R = cuda.local.array(9, dtype=float64); X = cuda.local.array(9, dtype=float64)
    for r in range(9):
        R[r] = 0.0; X[r] = 0.0
        for c in range(9): L[r, c] = 0.0
    
    for row in range(6):
        if row==0: i=0; j=0
        elif row==1: i=1; j=1
        elif row==2: i=2; j=2
        elif row==3: i=0; j=1
        elif row==4: i=0; j=2
        elif row==5: i=1; j=2
        
        # استفاده صحیح از m_M2c2 (گشتاور مرتبه 4)
        val_u2c2_ij = 0.0
        if row==0: val_u2c2_ij = m_M2c2[0] * inv_N
        elif row==1: val_u2c2_ij = m_M2c2[1] * inv_N
        elif row==2: val_u2c2_ij = m_M2c2[2] * inv_N
        elif row==3: val_u2c2_ij = m_M2c2[3] * inv_N
        elif row==4: val_u2c2_ij = m_M2c2[4] * inv_N
        elif row==5: val_u2c2_ij = m_M2c2[5] * inv_N
        
        R[row] = -2.0 * Lambda * val_u2c2_ij

        for l in range(3):
            val_u2_lj = 0.0
            if l==j: 
                if l==0: val_u2_lj = u2_0
                elif l==1: val_u2_lj = u2_1
                else: val_u2_lj = u2_2
            else:
                s_l = min(l,j); s_j = max(l,j)
                if s_l==0 and s_j==1: val_u2_lj = u2_3
                elif s_l==0 and s_j==2: val_u2_lj = u2_4
                else: val_u2_lj = u2_5
            c_idx_il = -1
            if min(i,l)==0 and max(i,l)==0: c_idx_il = 0
            elif min(i,l)==1 and max(i,l)==1: c_idx_il = 1
            elif min(i,l)==2 and max(i,l)==2: c_idx_il = 2
            elif min(i,l)==0 and max(i,l)==1: c_idx_il = 3
            elif min(i,l)==0 and max(i,l)==2: c_idx_il = 4
            elif min(i,l)==1 and max(i,l)==2: c_idx_il = 5
            L[row, c_idx_il] += val_u2_lj
            val_u2_li = 0.0
            if l==i:
                if l==0: val_u2_li = u2_0
                elif l==1: val_u2_li = u2_1
                else: val_u2_li = u2_2
            else:
                s_l = min(l,i); s_i = max(l,i)
                if s_l==0 and s_i==1: val_u2_li = u2_3
                elif s_l==0 and s_i==2: val_u2_li = u2_4
                else: val_u2_li = u2_5
            c_idx_jl = -1
            if min(j,l)==0 and max(j,l)==0: c_idx_jl = 0
            elif min(j,l)==1 and max(j,l)==1: c_idx_jl = 1
            elif min(j,l)==2 and max(j,l)==2: c_idx_jl = 2
            elif min(j,l)==0 and max(j,l)==1: c_idx_jl = 3
            elif min(j,l)==0 and max(j,l)==2: c_idx_jl = 4
            elif min(j,l)==1 and max(j,l)==2: c_idx_jl = 5
            L[row, c_idx_jl] += val_u2_li
        val_u3_j = u3_vec_0 if j==0 else (u3_vec_1 if j==1 else u3_vec_2)
        val_u3_i = u3_vec_0 if i==0 else (u3_vec_1 if i==1 else u3_vec_2)
        L[row, 6 + i] += val_u3_j
        L[row, 6 + j] += val_u3_i

    for i in range(3):
        row = 6 + i
        for l in range(3):
            val_u3_l = u3_vec_0 if l==0 else (u3_vec_1 if l==1 else u3_vec_2)
            c_idx_il = -1
            mn_i = min(i,l); mx_i = max(i,l)
            if mn_i==0 and mx_i==0: c_idx_il = 0
            elif mn_i==1 and mx_i==1: c_idx_il = 1
            elif mn_i==2 and mx_i==2: c_idx_il = 2
            elif mn_i==0 and mx_i==1: c_idx_il = 3
            elif mn_i==0 and mx_i==2: c_idx_il = 4
            elif mn_i==1 and mx_i==2: c_idx_il = 5
            L[row, c_idx_il] += val_u3_l
        for j in range(3):
            for l in range(3):
                c_idx_jl = -1
                mn_j = min(j,l); mx_j = max(j,l)
                if mn_j==0 and mx_j==0: c_idx_jl = 0
                elif mn_j==1 and mx_j==1: c_idx_jl = 1
                elif mn_j==2 and mx_j==2: c_idx_jl = 2
                elif mn_j==0 and mx_j==1: c_idx_jl = 3
                elif mn_j==0 and mx_j==2: c_idx_jl = 4
                elif mn_j==1 and mx_j==2: c_idx_jl = 5
                
                idx_list_0 = i; idx_list_1 = j; idx_list_2 = l
                if idx_list_0 > idx_list_1: t=idx_list_0; idx_list_0=idx_list_1; idx_list_1=t
                if idx_list_1 > idx_list_2: t=idx_list_1; idx_list_1=idx_list_2; idx_list_2=t
                if idx_list_0 > idx_list_1: t=idx_list_0; idx_list_0=idx_list_1; idx_list_1=t
                
                m3_idx = 0
                if idx_list_0==0 and idx_list_1==0 and idx_list_2==0: m3_idx=0
                elif idx_list_0==1 and idx_list_1==1 and idx_list_2==1: m3_idx=1
                elif idx_list_0==2 and idx_list_1==2 and idx_list_2==2: m3_idx=2
                elif idx_list_0==0 and idx_list_1==0 and idx_list_2==1: m3_idx=3
                elif idx_list_0==0 and idx_list_1==0 and idx_list_2==2: m3_idx=4
                elif idx_list_0==0 and idx_list_1==1 and idx_list_2==1: m3_idx=5
                elif idx_list_0==1 and idx_list_1==1 and idx_list_2==2: m3_idx=6
                elif idx_list_0==0 and idx_list_1==2 and idx_list_2==2: m3_idx=7
                elif idx_list_0==1 and idx_list_1==2 and idx_list_2==2: m3_idx=8
                elif idx_list_0==0 and idx_list_1==1 and idx_list_2==2: m3_idx=9
                val_u3_full = m_M3f[m3_idx] * inv_N
                L[row, c_idx_jl] += 2.0 * val_u3_full

        u4_scalar = m_M4 * inv_N
        L[row, 6 + i] += (u4_scalar - tr_u2**2)
        for j in range(3):
            mn_i = min(i,j); mx_i = max(i,j)
            m2c2_idx = 0
            if mn_i==0 and mx_i==0: m2c2_idx=0
            elif mn_i==1 and mx_i==1: m2c2_idx=1
            elif mn_i==2 and mx_i==2: m2c2_idx=2
            elif mn_i==0 and mx_i==1: m2c2_idx=3
            elif mn_i==0 and mx_i==2: m2c2_idx=4
            elif mn_i==1 and mx_i==2: m2c2_idx=5
            val_u2c2 = m_M2c2[m2c2_idx] * inv_N
            val_u2_ij = 0.0
            if mn_i==0 and mx_i==0: val_u2_ij=u2_0
            elif mn_i==1 and mx_i==1: val_u2_ij=u2_1
            elif mn_i==2 and mx_i==2: val_u2_ij=u2_2
            elif mn_i==0 and mx_i==1: val_u2_ij=u2_3
            elif mn_i==0 and mx_i==2: val_u2_ij=u2_4
            elif mn_i==1 and mx_i==2: val_u2_ij=u2_5
            L[row, 6 + j] += 2.0 * (val_u2c2 - tr_u2 * val_u2_ij)

        val_u3_i = u3_vec_0 if i==0 else (u3_vec_1 if i==1 else u3_vec_2)
        rhs_val = (5.0 / (3.0 * tau_eff)) * val_u3_i
        val_u4_i = m_M4v[i] * inv_N
        contract_vec_2 = 0.0
        for l in range(3):
            val_u3_l = u3_vec_0 if l==0 else (u3_vec_1 if l==1 else u3_vec_2)
            val_u2_il = 0.0
            mn_i = min(i,l); mx_i = max(i,l)
            if mn_i==0 and mx_i==0: val_u2_il=u2_0
            elif mn_i==1 and mx_i==1: val_u2_il=u2_1
            elif mn_i==2 and mx_i==2: val_u2_il=u2_2
            elif mn_i==0 and mx_i==1: val_u2_il=u2_3
            elif mn_i==0 and mx_i==2: val_u2_il=u2_4
            elif mn_i==1 and mx_i==2: val_u2_il=u2_5
            contract_vec_2 += val_u3_l * val_u2_il
        
        brack = 3.0 * val_u4_i - contract_vec_2 - val_u3_i * tr_u2
        R[row] = rhs_val - Lambda * brack

    solve_9x9_device_stable(L, R, X)
    
    max_coeff = 0.0
    for k in range(9):
        if abs(X[k]) > max_coeff: max_coeff = abs(X[k])
    
    if max_coeff > 5000000.0:
        for k in range(6): c_C[idx, k] = 0.0
        for k in range(3): c_Gamma[idx, k] = 0.0
        c_Beta[idx] = 0.0
    else:
        for k in range(6): c_C[idx, k] = X[k]
        for k in range(3): c_Gamma[idx, k] = X[6+k]

@cuda.jit
def broadcast_coeffs_kernel(adaptation_map, c_C, c_Gamma, c_Beta, nx, ny):
    idx = cuda.grid(1)
    if idx >= adaptation_map.shape[0]: return
    level = adaptation_map[idx]
    if level <= 1: return 
    ix = idx % nx; iy = idx // nx
    root_ix = (ix // level) * level
    root_iy = (iy // level) * level
    root_idx = root_iy * nx + root_ix
    if idx != root_idx:
        for k in range(6): c_C[idx, k] = c_C[root_idx, k]
        for k in range(3): c_Gamma[idx, k] = c_Gamma[root_idx, k]
        c_Beta[idx] = c_Beta[root_idx]

@cuda.jit
def calc_alpha_terms_kernel(vx, vy, vz, cell_ids, U, T, rho, c_C, c_Gamma, c_Beta, q, sum_NN, sum_MN, mass, kb):
    i = cuda.grid(1)
    if i >= vx.shape[0]: return
    idx = cell_ids[i]
    if idx < 0: return
    if rho[idx] <= 1e-20: return
    RT = kb * T[idx] / mass
    inv_rho = 1.0 / rho[idx]
    mx = vx[i] - U[idx, 0]; my = vy[i] - U[idx, 1]; mz = vz[i] - U[idx, 2]
    m2 = mx**2 + my**2 + mz**2; m2_lim = min(m2, 25.0 * RT)
    C = c_C[idx]; Gam = c_Gamma[idx]; Bet = c_Beta[idx]
    Nx = C[0]*mx + C[3]*my + C[4]*mz; Ny = C[3]*mx + C[1]*my + C[5]*mz; Nz = C[4]*mx + C[5]*my + C[2]*mz
    poly_g = m2_lim - 3.0 * RT 
    Nx += Gam[0] * poly_g; Ny += Gam[1] * poly_g; Nz += Gam[2] * poly_g
    Nx += Bet * (m2_lim * mx - 2.0 * q[idx,0] * inv_rho)
    Ny += Bet * (m2_lim * my - 2.0 * q[idx,1] * inv_rho)
    Nz += Bet * (m2_lim * mz - 2.0 * q[idx,2] * inv_rho)
    N2 = Nx**2 + Ny**2 + Nz**2; MN = mx*Nx + my*Ny + mz*Nz
    cuda.atomic.add(sum_NN, idx, N2); cuda.atomic.add(sum_MN, idx, MN)

# --- NEW: Noise Reduction Kernels (Cell-Wise Local) ---

# ---------------------------------------------------------
# KERNEL 1: تولید نویز خام و جمع‌آوری آمار (Sum, Sum^2)
# ---------------------------------------------------------
@cuda.jit
def generate_noise_stats_kernel(rng, n_part, active_mask, cell_ids, d_xi_x, d_xi_y, d_xi_z, d_noise_stats, d_N_local):
    i = cuda.grid(1)
    if i >= n_part: return
    if active_mask[i] == 0: return 
    idx = cell_ids[i]
    if idx < 0: return
    
    # 1. تولید اعداد تصادفی خام (Raw Gaussian)
    r1 = xoroshiro128p_normal_float32(rng, i)
    r2 = xoroshiro128p_normal_float32(rng, i)
    r3 = xoroshiro128p_normal_float32(rng, i)
    
    # 2. ذخیره موقت
    d_xi_x[i] = r1
    d_xi_y[i] = r2
    d_xi_z[i] = r3
    
    # 3. جمع‌زنی اتمیک برای محاسبه میانگین و واریانس سلول
    # Layout: [sum_x, sum_y, sum_z, sum_x2, sum_y2, sum_z2]
    cuda.atomic.add(d_noise_stats, (idx, 0), r1)
    cuda.atomic.add(d_noise_stats, (idx, 1), r2)
    cuda.atomic.add(d_noise_stats, (idx, 2), r3)
    cuda.atomic.add(d_noise_stats, (idx, 3), r1*r1)
    cuda.atomic.add(d_noise_stats, (idx, 4), r2*r2)
    cuda.atomic.add(d_noise_stats, (idx, 5), r3*r3)
    
    # 4. شمارش تعداد ذرات محلی (برای تقسیم صحیح)
    cuda.atomic.add(d_N_local, idx, 1.0)



@cuda.jit
def fp_update_kernel_correct(vx, vy, vz, cell_ids, dt, U, T, tau, alpha, c_C, c_Gamma, c_Beta, q, rho, mass, kb, active_mask, d_xi_x, d_xi_y, d_xi_z, d_noise_stats, d_N_local):
        i = cuda.grid(1)
        if i >= vx.shape[0]: return
        if active_mask[i] == 0: return 
        idx = cell_ids[i]
        if idx < 0: return
        
        # اگر شرایط فیزیکی نامعتبر است، آپدیت نکن
        if rho[idx] <= 1e-20: return
        
        # -----------------------------------------------------
        # 1. NOISE CORRECTION ALGORITHM (Local Cell-Wise)
        # -----------------------------------------------------
        count = d_N_local[idx]
        
        # خواندن نویز خام
        xi_x = d_xi_x[i]; xi_y = d_xi_y[i]; xi_z = d_xi_z[i]
        
        # فقط اگر بیش از 1 ذره در سلول باشد اصلاح می‌کنیم
        if count > 1:
            # A. محاسبه میانگین نمونه (Sample Mean)
            mu_x = d_noise_stats[idx, 0] / count
            mu_y = d_noise_stats[idx, 1] / count
            mu_z = d_noise_stats[idx, 2] / count
            
            # B. محاسبه واریانس نمونه (Sample Variance)
            # فرمول: (Sum(x^2)/N) - (Mean)^2
            var_x = (d_noise_stats[idx, 3] / count) - mu_x*mu_x
            var_y = (d_noise_stats[idx, 4] / count) - mu_y*mu_y
            var_z = (d_noise_stats[idx, 5] / count) - mu_z*mu_z
            
            # C. محاسبه ضریب اسکیل (Inverse STD)
            # چک می‌کنیم واریانس منفی یا صفر نشود (به خاطر خطای اعشار)
            scale_x = 1.0 / math.sqrt(var_x) if var_x > 1e-15 else 1.0
            scale_y = 1.0 / math.sqrt(var_y) if var_y > 1e-15 else 1.0
            scale_z = 1.0 / math.sqrt(var_z) if var_z > 1e-15 else 1.0
            
            # D. اعمال اصلاح: (Raw - Mean) * Scale
            # نتیجه این عملیات: میانگین = 0، واریانس = 1
            xi_x = (xi_x - mu_x) * scale_x
            xi_y = (xi_y - mu_y) * scale_y
            xi_z = (xi_z - mu_z) * scale_z
        
        # -----------------------------------------------------
        # 2. PHYSICS UPDATE (Full Cubic FP Drift)
        # -----------------------------------------------------
        RT = kb * T[idx] / mass
        
        # سرعت‌های نوسانی
        mx = vx[i] - U[idx, 0]
        my = vy[i] - U[idx, 1]
        mz = vz[i] - U[idx, 2]
        
        # محدود کردن m2 برای پایداری عددی (طبق استاندارد کدهای گرجی)
        m2 = mx**2 + my**2 + mz**2
        m2_lim = m2
        if m2_lim > 25.0 * RT: m2_lim = 25.0 * RT
        
        # خواندن ضرایب از حافظه
        C = c_C[idx]
        Gam = c_Gamma[idx]
        Bet = c_Beta[idx]
        
        # الف) بخش خطی (Linear Drift: C_ij * v_j)
        # چیدمان C در حافظه: 0:xx, 1:yy, 2:zz, 3:xy, 4:xz, 5:yz
        Nx = C[0]*mx + C[3]*my + C[4]*mz
        Ny = C[3]*mx + C[1]*my + C[5]*mz
        Nz = C[4]*mx + C[5]*my + C[2]*mz
        
        # ب) بخش درجه دو (Quadratic Drift: Gamma * (v^2 - 3RT))
        poly_g = m2_lim - 3.0 * RT
        Nx += Gam[0] * poly_g
        Ny += Gam[1] * poly_g
        Nz += Gam[2] * poly_g
        
        # ج) بخش درجه سه (Cubic Drift: Beta * (v v^2 - 2q/rho))
        inv_rho = 1.0 / rho[idx]
        Nx += Bet * (m2_lim * mx - 2.0 * q[idx,0] * inv_rho)
        Ny += Bet * (m2_lim * my - 2.0 * q[idx,1] * inv_rho)
        Nz += Bet * (m2_lim * mz - 2.0 * q[idx,2] * inv_rho)
        
        # -----------------------------------------------------
        # 3. TIME INTEGRATION (Exact Solution of OU Process)
        # -----------------------------------------------------
        arg = -dt / tau[idx]
        if arg < -20.0: arg = -20.0 
        
        exp_val = math.exp(arg)
        
        # محاسبه ضریب دیفیوژن (سیگما)
        sig_arg = 1.0 - math.exp(2.0 * arg)
        if sig_arg < 0: sig_arg = 0.0 # جلوگیری از خطای رادیکال
        sig = math.sqrt(RT * sig_arg)
        
        inv_alpha = 1.0 / alpha[idx]
        
        # آپدیت نهایی سرعت‌ها
        vx[i] = U[idx,0] + inv_alpha * (mx*exp_val + (1.0-exp_val)*tau[idx]*Nx + sig*xi_x)
        vy[i] = U[idx,1] + inv_alpha * (my*exp_val + (1.0-exp_val)*tau[idx]*Ny + sig*xi_y)
        vz[i] = U[idx,2] + inv_alpha * (mz*exp_val + (1.0-exp_val)*tau[idx]*Nz + sig*xi_z)

@cuda.jit
def verify_corrected_noise_kernel(n_part, active_mask, cell_ids, d_xi_x, d_noise_stats, d_N_local, d_check_result):
        # این کرنل دوباره همان اصلاحات را انجام می‌دهد و میانگین/واریانس ثانویه را حساب می‌کند
        # تا به شما ثابت کند که نتیجه 0 و 1 است.
        i = cuda.grid(1)
        if i >= n_part: return
        if active_mask[i] == 0: return 
        idx = cell_ids[i]
        if idx < 0: return
        
        count = d_N_local[idx]
        if count <= 1: return
        
        # بازسازی پارامترهای اصلاح
        sum_x = d_noise_stats[idx, 0]
        sum_x2 = d_noise_stats[idx, 3]
        mu = sum_x / count
        var = (sum_x2 / count) - mu*mu
        scale = 1.0 / math.sqrt(var + 1e-20)
        
        # نویز اصلاح شده
        raw_x = d_xi_x[i]
        corrected_x = (raw_x - mu) * scale
        
        # جمع زدن نویز اصلاح شده (باید 0 شود) و توان دوش (باید count شود)
        # result: [sum_corr, sum_corr_sq, total_samples]
        cuda.atomic.add(d_check_result, 0, corrected_x)
        cuda.atomic.add(d_check_result, 1, corrected_x**2)
        cuda.atomic.add(d_check_result, 2, 1.0)

# =============================================================================
# MAIN SOLVER (Full Class)
# =============================================================================
class CubicFPSolver:
    def __init__(self):
        self.dx = DOMAIN_LEN_X / N_CELLS_X
        self.dy = DOMAIN_LEN_Y / N_CELLS_Y
        self.x_min = X_MIN_DOMAIN
        self.x_max = X_MAX_DOMAIN
        self.y_max = Y_MAX_DOMAIN
        
        self.n_capacity = LOCAL_CAPACITY 
        self.n_part = LOCAL_PARTICLES 
        
        self.x = np.zeros(self.n_capacity, dtype=np.float32)
        self.y = np.zeros(self.n_capacity, dtype=np.float32)
        self.vx = np.zeros(self.n_capacity, dtype=np.float32)
        self.vy = np.zeros(self.n_capacity, dtype=np.float32)
        self.vz = np.zeros(self.n_capacity, dtype=np.float32)
        self.cell_ids = np.full(self.n_capacity, -1, dtype=np.int32)
        self.active_mask = np.zeros(self.n_capacity, dtype=np.int32)
        self.active_mask[:self.n_part] = 1 
        
        # Noise Arrays
        self.d_xi_x = cuda.device_array(self.n_capacity, dtype=np.float32)
        self.d_xi_y = cuda.device_array(self.n_capacity, dtype=np.float32)
        self.d_xi_z = cuda.device_array(self.n_capacity, dtype=np.float32)
        self.d_noise_stats = cuda.device_array((N_CELLS_TOTAL, 6), dtype=np.float64)
        self.d_N_local = cuda.device_array(N_CELLS_TOTAL, dtype=np.float64) # Local count for noise

        self.d_x = cuda.to_device(self.x)
        self.d_y = cuda.to_device(self.y)
        self.d_vx = cuda.to_device(self.vx)
        self.d_vy = cuda.to_device(self.vy)
        self.d_vz = cuda.to_device(self.vz)
        self.d_cell_ids = cuda.to_device(self.cell_ids)
        self.d_active_mask = cuda.to_device(self.active_mask)
        
        self.rng = create_xoroshiro128p_states(self.n_capacity, seed=42 + rank)
        
        self.P_tensor = np.zeros((N_CELLS_TOTAL, 6), dtype=np.float64)
        self.rho = np.zeros(N_CELLS_TOTAL)
        self.T = np.ones(N_CELLS_TOTAL) * T_INF
        self.U = np.zeros((N_CELLS_TOTAL, 3))
        self.q = np.zeros((N_CELLS_TOTAL, 3))
        self.tau = np.ones(N_CELLS_TOTAL)
        
        self.d_rho = cuda.device_array(N_CELLS_TOTAL, dtype=np.float64)
        self.d_T = cuda.device_array(N_CELLS_TOTAL, dtype=np.float64)
        self.d_U = cuda.device_array((N_CELLS_TOTAL, 3), dtype=np.float64)
        self.d_q = cuda.device_array((N_CELLS_TOTAL, 3), dtype=np.float64)
        self.d_tau = cuda.device_array(N_CELLS_TOTAL, dtype=np.float64)
        self.d_alpha = cuda.device_array(N_CELLS_TOTAL, dtype=np.float64)
        self.d_N = cuda.device_array(N_CELLS_TOTAL, dtype=np.float64)
        
        self.d_vol_frac = cuda.device_array(N_CELLS_TOTAL, dtype=np.float64)
        self.adaptation_map = np.ones(N_CELLS_TOTAL, dtype=np.int32)
        self.d_adaptation_map = cuda.to_device(self.adaptation_map)
        
        self.d_raw_moments = cuda.device_array((N_CELLS_TOTAL, 5), dtype=np.float64)
        self.d_packed_central = cuda.device_array((N_CELLS_TOTAL, 29), dtype=np.float64)
        
        self.d_c_C = cuda.device_array((N_CELLS_TOTAL, 6), dtype=np.float64)
        self.d_c_Gamma = cuda.device_array((N_CELLS_TOTAL, 3), dtype=np.float64)
        self.d_c_Beta = cuda.device_array(N_CELLS_TOTAL, dtype=np.float64)
        self.d_sum_NN = cuda.device_array(N_CELLS_TOTAL, dtype=np.float64)
        self.d_sum_MN = cuda.device_array(N_CELLS_TOTAL, dtype=np.float64)
        self.d_surf_n = cuda.device_array(90, dtype=np.float64)
        self.d_surf_p = cuda.device_array(90, dtype=np.float64)
        self.d_surf_tau = cuda.device_array(90, dtype=np.float64)
        self.d_surf_q = cuda.device_array(90, dtype=np.float64)
        
        blocks = (90 + 255) // 256
        zero_surface_arrays_kernel[blocks, 256](self.d_surf_n, self.d_surf_p, self.d_surf_tau, self.d_surf_q)
        
        self.samp_count = 0
        self.sum_rho = np.zeros(N_CELLS_TOTAL)
        self.sum_T = np.zeros(N_CELLS_TOTAL)
        self.sum_U = np.zeros((N_CELLS_TOTAL, 3))
        self.sum_P = np.zeros((N_CELLS_TOTAL, 6))
        self.sum_q = np.zeros((N_CELLS_TOTAL, 3))
        self.sim_time_accum = 0.0

        self.x_grid = self.x_min + (np.arange(N_CELLS_X) + 0.5) * self.dx
        self.y_grid = (np.arange(N_CELLS_Y) + 0.5) * self.dy
        XX, YY = np.meshgrid(self.x_grid, self.y_grid)
        R_dist = np.sqrt(XX**2 + YY**2)
        mask_near = R_dist < (R_CYL + 0.02) 
        mask_shock = (XX > -0.40) & (XX < 0.1) & (np.abs(YY) < 0.35)
        self.force_fine_mask = mask_near | mask_shock

        blocks_vol = (N_CELLS_TOTAL + 255) // 256
        calc_volume_fraction_kernel[blocks_vol, 256](
            self.d_vol_frac, N_CELLS_X, N_CELLS_Y, self.dx, self.dy, 
            self.x_min, R_CYL, self.rng
        )

        threads = 256
        blocks = (self.n_capacity + 255) // 256
        init_particles_kernel[blocks, threads](
            self.d_x, self.d_y, self.d_vx, self.d_vy, self.d_vz, self.d_cell_ids,
            self.d_active_mask, self.rng, self.x_min, self.x_max, self.y_max, R_CYL,
            U_INF, T_INF, MASS, KB, self.n_part
        )
        
        vol = (DOMAIN_LEN_X * DOMAIN_LEN_Y - 0.5*np.pi*R_CYL**2)
        self.f_num = (NUM_DENS_INF * vol) / self.n_part 
        self.cell_vol = self.dx * self.dy
        self.dt = DT_INIT 

    def check_noise_quality(self):
        # آرایه نتیجه روی GPU
        d_check = cuda.device_array(3, dtype=np.float64) # [sum, sum_sq, N]
        zero_1d_kernel[1, 1](d_check)
            
        # اجرای کرنل چک
        blocks = (self.n_capacity + 255) // 256
        verify_corrected_noise_kernel[blocks, 256](
        self.n_capacity, self.d_active_mask, self.d_cell_ids,
        self.d_xi_x, self.d_noise_stats, self.d_N_local, d_check
        )
            
            # دریافت نتیجه
        res = d_check.copy_to_host()
        total_sum = res[0]
        total_sum_sq = res[1]
        total_N = res[2]
            
        if total_N > 0:
            mean_error = total_sum / total_N     # باید بسیار نزدیک به 0.0 باشد (مثلاً 1e-15)
            var_value = total_sum_sq / total_N   # باید بسیار نزدیک به 1.0 باشد
            return mean_error, var_value
        return 0.0, 0.0


   

        

    def calculate_gradients_and_adapt(self):
        rho_grid = self.rho.reshape(N_CELLS_Y, N_CELLS_X)
        T_grid = self.T.reshape(N_CELLS_Y, N_CELLS_X)
        U_mag = np.sqrt(self.U[:,0]**2 + self.U[:,1]**2 + self.U[:,2]**2).reshape(N_CELLS_Y, N_CELLS_X)
        grad_rho = np.sqrt(np.gradient(rho_grid, axis=1)**2 + np.gradient(rho_grid, axis=0)**2) / self.dx
        grad_T = np.sqrt(np.gradient(T_grid, axis=1)**2 + np.gradient(T_grid, axis=0)**2) / self.dx
        grad_U = np.sqrt(np.gradient(U_mag, axis=1)**2 + np.gradient(U_mag, axis=0)**2) / self.dx
        delta_rho = (grad_rho * self.dx) / (rho_grid + 1e-20)
        delta_T = (grad_T * self.dx) / (T_grid + 1e-20)
        delta_U = (grad_U * self.dx) / (U_mag + 1e-20)
        delta_max_local = np.maximum(np.maximum(delta_rho, delta_T), delta_U)
        new_map = np.where(self.force_fine_mask | (delta_max_local > DELTA_MAX), 1, COARSE_LEVEL).flatten().astype(np.int32)
        self.d_adaptation_map.copy_to_device(new_map)

    def update_dt_cfl(self):
        vx = self.U[:,0]; vy = self.U[:,1]
        v_mag = np.sqrt(vx**2 + vy**2)
        c = np.sqrt(1.667 * KB * np.maximum(self.T, 1.0) / MASS)
        max_speed = np.max(v_mag + c)
        if max_speed < 100: max_speed = U_INF
        min_d = min(self.dx, self.dy)
        dt_cfl = CFL_NUM * min_d / max_speed
        if dt_cfl > 5e-8: dt_cfl = 5e-8
        max_speed_arr = np.array(max_speed, dtype=np.float64)
        max_speed_glob = np.array(0.0, dtype=np.float64)
        comm.Allreduce(max_speed_arr, max_speed_glob, op=MPI.MAX)
        self.dt = CFL_NUM * min_d / max_speed_glob
        if self.dt > 5e-8: self.dt = 5e-8
        return self.dt

    def run_step(self, step_idx):
        if step_idx > 10 and step_idx % 20 == 0:
            new_dt = self.update_dt_cfl()
            if not math.isnan(new_dt) and new_dt > 0.0:
                self.dt = new_dt
        
        dt = self.dt
        if step_idx >= WARMUP_STEPS: self.sim_time_accum += dt 

        blocks = (self.n_capacity + 255) // 256
        
        real_inject = (NUM_DENS_INF * U_INF * self.y_max * dt)
        if self.f_num > 0: sim_inject = real_inject / self.f_num
        else: sim_inject = 0.0
            
        if math.isnan(sim_inject) or math.isinf(sim_inject): n_inject_int = 0
        else:
            n_inject_int = int(sim_inject)
            if np.random.rand() < (sim_inject - n_inject_int): n_inject_int += 1
            
        if n_inject_int > 0:
            inj_blocks = (n_inject_int + 255) // 256
            inject_particles_kernel[inj_blocks, 256](
                self.d_x, self.d_y, self.d_vx, self.d_vy, self.d_vz, 
                self.d_cell_ids, self.d_active_mask, self.rng, dt, 
                self.x_min, self.y_max, self.n_capacity, n_inject_int, 
                U_INF, T_INF, MASS, KB
            )

        move_particles_half_domain_kernel[blocks, 256](
            self.d_x, self.d_y, self.d_vx, self.d_vy, self.d_vz, dt,
            R_CYL, self.x_min, self.x_max, self.y_max,
            T_WALL, T_INF, U_INF, MASS, KB, self.rng,
            self.d_surf_p, self.d_surf_tau, self.d_surf_q, self.d_surf_n, 90,
            self.d_active_mask, self.d_cell_ids
        )
        
        bin_particles_kernel[blocks, 256](
            self.d_x, self.d_y, self.d_cell_ids, self.d_active_mask,
            self.x_min, self.dx, self.dy,
            N_CELLS_X, N_CELLS_Y, R_CYL
        )
        
        do_update = (step_idx % MACRO_UPDATE_FREQ == 0) or (step_idx < WARMUP_STEPS)
        if do_update:
            blocks_raw = (N_CELLS_TOTAL * 5 + 255) // 256
            zero_2d_kernel[blocks_raw, 256](self.d_raw_moments)
            calc_raw_moments_kernel[blocks, 256](
                self.d_vx, self.d_vy, self.d_vz, self.d_cell_ids,
                self.d_raw_moments
            )
            local_raw = self.d_raw_moments.copy_to_host()
            global_raw = np.zeros_like(local_raw)
            comm.Allreduce(local_raw, global_raw, op=MPI.SUM)
            
            sum_N = global_raw[:, 0]
            sum_U = global_raw[:, 1:4]
            sum_E = global_raw[:, 4]
            mask = sum_N > 0
            
            self.rho[:] = 0; self.T[:] = T_INF; self.U[:] = 0
            
            vol_frac_host = self.d_vol_frac.copy_to_host()
            vol_frac_host = np.maximum(vol_frac_host, 0.01) 
            
            self.rho[mask] = (sum_N[mask] * self.f_num * MASS) / (self.cell_vol * vol_frac_host[mask])
            self.U[mask] = sum_U[mask] / sum_N[mask, None]
            u2 = np.sum(self.U**2, axis=1)
            v2_avg = sum_E / np.maximum(sum_N, 1)
            self.T[mask] = (MASS / (3.0*KB)) * (v2_avg[mask] - u2[mask])
            self.T[mask] = np.maximum(self.T[mask], 1.0) 
            
            if step_idx > WARMUP_STEPS and step_idx % (MACRO_UPDATE_FREQ*5) == 0:
                self.calculate_gradients_and_adapt()

            T_safe = np.maximum(self.T, 1.0) 
            mu = MU_INF * (T_safe / T_INF)**OMEGA
            p = self.rho * KB * T_safe / MASS
            p_safe = np.maximum(p, 1e-10)
            self.tau[mask] = 2.0 * mu[mask] / p_safe[mask]
            self.tau = np.maximum(self.tau, 1e-20)
            
            self.d_rho.copy_to_device(self.rho)
            self.d_T.copy_to_device(self.T)
            self.d_U.copy_to_device(self.U)
            self.d_tau.copy_to_device(self.tau)
            self.d_N.copy_to_device(sum_N) 
            
            blocks_central = (N_CELLS_TOTAL * 29 + 255) // 256
            zero_2d_kernel[blocks_central, 256](self.d_packed_central)
            
            calc_central_moments_kernel[blocks, 256](
                self.d_vx, self.d_vy, self.d_vz, self.d_cell_ids, self.d_U,
                self.d_packed_central
            )
            local_central = self.d_packed_central.copy_to_host()
            global_central = np.zeros_like(local_central)
            comm.Allreduce(local_central, global_central, op=MPI.SUM)
            self.d_packed_central.copy_to_device(global_central)
            
            m3v = global_central[:, 6:9]
            self.q[mask] = 0.5 * (self.f_num * MASS / (self.cell_vol * vol_frac_host[mask][:,None])) * m3v[mask]
            self.d_q.copy_to_device(self.q)
            d_N_arr = self.d_N 

            blocks_grid = (N_CELLS_TOTAL + 255) // 256
            compute_adaptive_coefficients_kernel[blocks_grid, 256](
                self.d_packed_central, d_N_arr, self.d_adaptation_map, self.d_tau, self.d_rho,
                self.d_c_C, self.d_c_Gamma, self.d_c_Beta,
                N_CELLS_X, N_CELLS_Y
            )
            broadcast_coeffs_kernel[blocks_grid, 256](
                self.d_adaptation_map, self.d_c_C, self.d_c_Gamma, self.d_c_Beta, 
                N_CELLS_X, N_CELLS_Y
            )
            
            zero_1d_kernel[blocks_grid, 256](self.d_sum_NN)
            zero_1d_kernel[blocks_grid, 256](self.d_sum_MN)

            calc_alpha_terms_kernel[blocks, 256](
                self.d_vx, self.d_vy, self.d_vz, self.d_cell_ids, self.d_U, self.d_T, self.d_rho,
                self.d_c_C, self.d_c_Gamma, self.d_c_Beta, self.d_q, 
                self.d_sum_NN, self.d_sum_MN, MASS, KB
            )
            
            NN_loc = self.d_sum_NN.copy_to_host(); NN_glob = np.zeros_like(NN_loc)
            MN_loc = self.d_sum_MN.copy_to_host(); MN_glob = np.zeros_like(MN_loc)
            comm.Allreduce(NN_loc, NN_glob, op=MPI.SUM)
            comm.Allreduce(MN_loc, MN_glob, op=MPI.SUM)
            E_NN = NN_glob / np.maximum(sum_N, 1)
            E_MN = MN_glob / np.maximum(sum_N, 1)
            term1 = self.tau * (1.0 - np.exp(-dt/self.tau))**2 * E_NN
            term2 = 2.0 * (np.exp(-dt/self.tau) - np.exp(-2.0*dt/self.tau)) * E_MN
            factor = (MASS * self.tau) / (3.0 * KB * T_safe)
            alpha_sq = 1.0 + factor * (term1 + term2)
            alpha_sq = np.where(mask, alpha_sq, 1.0)
            self.alpha = np.sqrt(np.maximum(alpha_sq, 0.1))
            self.d_alpha.copy_to_device(self.alpha)
        
        # --- NEW: Correct Local Noise Reduction ---
        # 1. Clear stats
        zero_1d_kernel[(N_CELLS_TOTAL + 255)//256, 256](self.d_N_local)
        zero_2d_kernel[(N_CELLS_TOTAL*6 + 255)//256, 256](self.d_noise_stats)
        
        # 2. Generate and Aggregate Locally
        generate_noise_stats_kernel[blocks, 256](
            self.rng, self.n_capacity, self.d_active_mask, self.d_cell_ids,
            self.d_xi_x, self.d_xi_y, self.d_xi_z, self.d_noise_stats, self.d_N_local
        )

        # 3. Update with Local Correction (No MPI Reduce needed here)
        fp_update_kernel_correct[blocks, 256](
            self.d_vx, self.d_vy, self.d_vz, self.d_cell_ids, dt,
            self.d_U, self.d_T, self.d_tau, self.d_alpha,
            self.d_c_C, self.d_c_Gamma, self.d_c_Beta, self.d_q, self.d_rho,
            MASS, KB, self.d_active_mask,
            self.d_xi_x, self.d_xi_y, self.d_xi_z, self.d_noise_stats, self.d_N_local
        )

        if step_idx >= WARMUP_STEPS:
            self.samp_count += 1
            self.sum_rho += self.rho
            self.sum_T += self.T
            self.sum_U += self.U
            if do_update:
                m2 = global_central[:, 0:6]
                P_curr = np.zeros((N_CELLS_TOTAL, 6), dtype=np.float64)
                P_curr[mask] = self.rho[mask][:, None] * (m2[mask] / sum_N[mask][:, None])
                self.sum_P += P_curr
                self.sum_q += self.q
                
        return dt

    def save_output(self, total_time_sim):
        surf_p_loc = self.d_surf_p.copy_to_host(); surf_p_glob = np.zeros_like(surf_p_loc)
        surf_tau_loc = self.d_surf_tau.copy_to_host(); surf_tau_glob = np.zeros_like(surf_tau_loc)
        surf_q_loc = self.d_surf_q.copy_to_host(); surf_q_glob = np.zeros_like(surf_q_loc)
        surf_n_loc = self.d_surf_n.copy_to_host(); surf_n_glob = np.zeros_like(surf_n_loc)
        
        comm.Reduce(surf_p_loc, surf_p_glob, op=MPI.SUM, root=0)
        comm.Reduce(surf_tau_loc, surf_tau_glob, op=MPI.SUM, root=0)
        comm.Reduce(surf_q_loc, surf_q_glob, op=MPI.SUM, root=0)
        comm.Reduce(surf_n_loc, surf_n_glob, op=MPI.SUM, root=0)
        
        if rank == 0:
            print("\nSaving Output...")
            
            rho_h = self.rho
            U_h = self.U
            T_h = self.T
            total_mass = np.sum(rho_h * self.cell_vol)
            total_mom_x = np.sum(rho_h * U_h[:,0] * self.cell_vol)
            E_int = 1.5 * (KB/MASS) * T_h
            E_kin = 0.5 * (U_h[:,0]**2 + U_h[:,1]**2 + U_h[:,2]**2)
            total_energy = np.sum(rho_h * (E_int + E_kin) * self.cell_vol)
            
            print("--- CONSERVATION CHECK (Snapshot) ---")
            print(f"Total Mass: {total_mass:.4e} kg")
            print(f"Total Mom-X: {total_mom_x:.4e} kg m/s")
            print(f"Total Energy: {total_energy:.4e} J")
            print("---------------------------------------")

            d_theta = (1.0 * np.pi) / 90
            area_bin = R_CYL * d_theta * 1.0 
            factor = self.f_num / (total_time_sim * area_bin)
            
            Cp = (surf_p_glob * factor - P_INF) / Q_DYN
            Cf = (surf_tau_glob * factor) / Q_DYN
            Ch = (surf_q_glob * factor) / Q_HEAT_REF
            theta_arr = np.linspace(0, 180, 90)
            
            with open('surface_data.dat', 'w') as f:
                f.write('VARIABLES = "Theta", "Cp", "Cf", "Ch"\n')
                f.write(f'ZONE I={90}, F=POINT\n')
                for i in range(90):
                    f.write(f'{theta_arr[i]:.4f} {Cp[i]:.6f} {Cf[i]:.6f} {Ch[i]:.6f}\n')
            
            plt.figure(figsize=(10, 12))
            plt.rcParams.update({'font.size': 22})
            plt.subplot(3,1,1); plt.plot(theta_arr, Cp, 'o-'); plt.ylabel(r'$C_p$'); plt.grid()
            plt.subplot(3,1,2); plt.plot(theta_arr, Cf, 'o-'); plt.ylabel(r'$C_f$'); plt.grid()
            plt.subplot(3,1,3); plt.plot(theta_arr, Ch, 'o-'); plt.ylabel(r'$C_h$'); plt.xlabel('Theta')
            plt.tight_layout()
            plt.savefig('Surface_Properties.jpg')
            plt.close()

            if self.samp_count > 0:
                avg_rho = self.sum_rho / self.samp_count
                avg_T = self.sum_T / self.samp_count
                avg_U = self.sum_U / self.samp_count
                avg_P = self.sum_P / self.samp_count
                avg_q = self.sum_q / self.samp_count
                
                gamma = 1.667
                sound_speed = np.sqrt(gamma * (KB/MASS) * np.maximum(avg_T, 1.0))
                vel_mag = np.sqrt(avg_U[:,0]**2 + avg_U[:,1]**2 + avg_U[:,2]**2)
                avg_Mach = vel_mag / sound_speed
                
                with open('field_data.dat', 'w') as f:
                    f.write('VARIABLES = "X", "Y", "Rho", "U", "V", "W", "T", "Mach", "Pxx", "Pyy", "Pzz", "Pxy", "Qx", "Qy"\n')
                    f.write(f'ZONE I={N_CELLS_X}, J={N_CELLS_Y}, F=POINT\n')
                    for j in range(N_CELLS_Y):
                        for i in range(N_CELLS_X):
                            idx = j * N_CELLS_X + i
                            xc = self.x_min + (i + 0.5) * self.dx
                            yc = 0.0 + (j + 0.5) * self.dy
                            
                            pxx = self.sum_P[idx, 0]
                            pyy = self.sum_P[idx, 1]
                            pzz = self.sum_P[idx, 2]
                            pxy = self.sum_P[idx, 3]
                            
                            f.write(f'{xc:.4f} {yc:.4f} {avg_rho[idx]:.4e} '
                                    f'{avg_U[idx,0]:.2f} {avg_U[idx,1]:.2f} {avg_U[idx,2]:.2f} '
                                    f'{avg_T[idx]:.2f} {avg_Mach[idx]:.3f} '
                                    f'{pxx:.2e} {pyy:.2e} {pzz:.2e} {pxy:.2e} '
                                    f'{avg_q[idx,0]:.2e} {avg_q[idx,1]:.2e}\n')

            if rank == 0:
                try:
                    adapt_map_cpu = self.d_adaptation_map.copy_to_host().reshape(N_CELLS_Y, N_CELLS_X)
                    plt.figure(figsize=(12, 6))
                    plt.pcolormesh(self.x_grid, self.y_grid, adapt_map_cpu, cmap='coolwarm', shading='nearest')
                    cbar = plt.colorbar()
                    cbar.set_label('Refinement Level (1=Fine, 4=Coarse)')
                    th = np.linspace(0, np.pi, 200)
                    plt.plot(R_CYL*np.cos(th), R_CYL*np.sin(th), 'k-', lw=2)
                    plt.title(f"Adaptive Mesh Status (Time: {total_time_sim:.2e}s)")
                    plt.xlabel("X [m]")
                    plt.ylabel("Y [m]")
                    plt.axis('equal')
                    plt.xlim(self.x_min, self.x_max)
                    plt.ylim(0, self.y_max)
                    plt.tight_layout()
                    plt.savefig('Adaptation_Map.jpg', dpi=100)
                    plt.close()
                except Exception as e:
                    print(f"Error plotting adaptation map: {e}")

if __name__ == "__main__":
    solver = CubicFPSolver()
    t0 = time.time()
    
    if rank == 0:
        print(f"Starting MPI Simulation (v172: Local Noise Reduction + Unclamped Beta): {TOTAL_STEPS} steps...")
    
    for i in range(TOTAL_STEPS):
        current_dt = solver.run_step(i)
        
        # Periodic Report (Every 100 steps)
        if i % 100 == 0:
            mask_host = solver.d_active_mask.copy_to_host()
            active_count = np.sum(mask_host)
            active_glob = comm.reduce(active_count, op=MPI.SUM, root=0)
            
            valid_T = solver.T[solver.rho > 0]
            T_mean_local = np.mean(valid_T) if len(valid_T) > 0 else 0.0
            T_max_local = np.max(solver.T)
            
            T_max_glob = comm.reduce(T_max_local, op=MPI.MAX, root=0)
            T_mean_sum = comm.reduce(T_mean_local, op=MPI.SUM, root=0)
            
            if rank == 0:
                rho_h = solver.rho 
                U_h = solver.U
                total_mass = np.sum(rho_h * solver.cell_vol)
                total_mom_x = np.sum(rho_h * U_h[:,0] * solver.cell_vol)
                
                T_mean_glob = T_mean_sum / size
                capacity_pct = (active_glob / (solver.n_capacity * size)) * 100.0
                
                print(f"Step {i} | T: {time.time()-t0:.1f}s | N: {active_glob} ({capacity_pct:.1f}%) | dt: {current_dt:.2e}")
                print(f"   Stats: T_max: {T_max_glob:.1f} K | T_avg: {T_mean_glob:.1f} K")
                print(f"   Consv: Mass: {total_mass:.4e} | Mom-X: {total_mom_x:.4e}")
                print("-" * 60)
                
                if capacity_pct > 98.0:
                    print("!!! CRITICAL WARNING: Capacity Full! Injection Stalled !!!")
        
        # Periodic Output Save (File Write)
        if i > WARMUP_STEPS and i % OUTPUT_INTERVAL == 0:
            if rank == 0: print(f"--- Saving Interim Output at Step {i} ---")
            solver.save_output(solver.sim_time_accum)
            
        if i % 100 == 0:
            # چک کردن کیفیت نویز
            err_mean, val_var = solver.check_noise_quality()
            if rank == 0:
                print(f"   [NOISE CHECK] Mean Error: {err_mean:.2e} (Target: 0.0) | Variance: {val_var:.6f} (Target: 1.0)")
    
    if rank == 0:
        print(f"Finished. Total Wall Time: {time.time()-t0:.2f}s")
        
    solver.save_output(solver.sim_time_accum)
    comm.Barrier()
    if rank == 0:
        print("Files saved.")
