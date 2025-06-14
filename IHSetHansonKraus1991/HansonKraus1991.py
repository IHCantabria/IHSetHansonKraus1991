import numpy as np
from numba import njit
# from datetime import datetime
from IHSetUtils.libjit.geometry import nauticalDir2cartesianDir, abs_pos, shore_angle
from IHSetUtils.libjit.waves import BreakingPropagation
from IHSetUtils.libjit.morfology import ALST
import math

# @njit(fastmath=True, cache=True)
# def hansonKraus1991(yi, dt, dx, hs, tp, dir, depth, doc, kal, X0, Y0, phi, bctype, Bcoef):
#     """
#     Function OneLine calculates the evolution of a system over time using a numerical method.

#     Args:
#         yi (ndarray): Initial conditions.
#         dt (float): Time step.
#         dx (float): Spatial step.
#         hs (ndarray): Matrix representing sea surface height.
#         tp (ndarray): Matrix representing wave periods.
#         dir (ndarray): Matrix representing wave directions. [**CARTESIAN**]
#         depth (float): Water depth.
#         doc (ndarray): Matrix representing diffusivity coefficients.
#         kal (bool): Flag indicating if Kalman filter is used.
#         X0 (ndarray): Initial x coordinates.
#         Y0 (ndarray): Initial y coordinates.
#         phi (ndarray): Phase angles.
#         bctype (int): Boundary condition type.

#     Returns:
#         ysol (ndarray): Solution matrix.
#         q (ndarray): Matrix representing some quantity.
#     """

    
#     nti = hs.shape[0]
#     # tii = 1
#     desl = 1

#     n1 = len(X0)
#     mt, n2 = tp.shape

#     ysol = np.zeros((mt, n1))
#     ysol[0, :] = yi

#     q = np.zeros((mt, n2))

#     ## Create the ghost condition

#     # time_init = datetime.now()
#     for pos in range(0, nti-1):
        
#         ti = pos+desl

#         # p1 = ti - desl
#         # p2 = ti + desl

#         ysol[ti,:], q[ti,:] =  ydir_L(ysol[ti-1, :], dt[ti-1], dx, hs[ti, :], tp[ti, :], dir[ti, :], depth[ti, :], doc[ti, :], kal, X0, Y0, phi, bctype, Bcoef)

#     # Make the interpolation for the actual transects!!!



#     return ysol, q

# @njit(fastmath=True, cache=True)
# def ydir_L(y, dt, dx, hs, tp, dire, depth, doc, kal, X0, Y0, phi, bctype, Bcoef):
#     """
#     Function ydir_L calculates the propagation of waves and other quantities at a specific time step.

#     Args:
#         y (ndarray): Initial conditions.
#         dt (ndarray): Time step.
#         dx (float): Spatial step.
#         ti (int): Time index.
#         hs (ndarray): Matrix representing sea surface height.
#         tp (ndarray): Matrix representing wave periods.
#         dire (ndarray): Matrix representing wave directions.
#         depth (float): Water depth.
#         hb (ndarray): Matrix representing breaking height.
#         dirb (ndarray): Matrix representing breaking direction.
#         depthb (ndarray): Matrix representing breaking depth.
#         q (ndarray): Matrix representing some quantity.
#         doc (ndarray): Matrix representing diffusivity coefficients.
#         kal (ndarray): Matrix representing Kalman filter.
#         X0 (ndarray): Initial x coordinates.
#         Y0 (ndarray): Initial y coordinates.
#         phi (ndarray): Phase angles.
#         bctype (str): Boundary condition type.

#     Returns:
#         Tuple containing:
#         - ynew (ndarray): Updated conditions.
#         - hb (ndarray): Updated breaking height.
#         - dirb (ndarray): Updated breaking direction.
#         - depthb (ndarray): Updated breaking depth.
#         - q (ndarray): Updated quantity matrix.
#         - q0 (ndarray): Updated quantity.
#     """
    
#     XN, YN = abs_pos(X0, Y0, phi * np.pi / 180.0, y)

#     alfas = np.zeros_like(hs)
#     # alfas_ = shore_angle(XN, YN, dire)
#     alfas_ = _compute_normals(XN, YN, phi)
#     alfas[1:-1] = alfas_
#     alfas[0] = alfas[1]
#     alfas[-1] = alfas[-2]

#     # try:
#     hb, dirb, depthb = BreakingPropagation(hs, tp, dire, depth, alfas, Bcoef)
#     # except:
#     #     flag_dig = True
#     #     print("Waves diverged -- Q_tot = 0")
#     #     hb = np.zeros_like(hs) + 0.01
#     #     dirb = np.zeros_like(dire) + alfas + 90
#     #     depthb = np.zeros_like(depth) + 0.01
        
#     dc = 0.5 * (doc[1:] + doc[:-1])

#     # if flag_dig:
#     #     q_now = np.zeros_like(hs)
#     #     q0 = np.zeros_like(hs)
#     # else:
#     q_now, q0 = ALST(hb, dirb, depthb, alfas, kal)

#     if bctype[0] == 0:
#         q_now[0] = 0
#     elif bctype[0] == 1:
#         q_now[0] = q_now[1]
#     if bctype[1] == 0:
#         q_now[-1] = 0
#     elif bctype[1] == 1:
#         q_now[-1] = q_now[-2]


#     if (np.mean(dx)**2 * np.min(dc) / (4 * np.max(q0) + 1e-6)) < (dt*60*60):
#         print("WARNING: COURANT CONDITION VIOLATED")
#         return y, q_now
    

#     ynew = y - (dt * 60 * 60) / dc * (q_now[1:] - q_now[:-1]) / dx


#     return ynew, q_now

# import numpy as np
# from numba import jit
# from IHSetUtils.libjit.geometry import abs_pos, cartesianDir2nauticalDir
# from IHSetUtils.libjit.waves import BreakingPropagation
# from IHSetUtils.libjit.morfology import ALST

# ###############################################################################
# # 0. Funções utilitárias (fantasmas) ---------------------------------------- #
# ###############################################################################

@njit(fastmath=True, cache=True)
def _compute_normals(X, Y, phi):
    """Devolve vetor de ângulos **normais** (perpendiculares) à costa, em graus.

    Para cada segmento entre os nós *i*‑*i+1* calcula‑se duas normais
    (θ ± 90°) e escolhe‑se aquela cujo desvio em módulo de 360° em relação
    a *phi[i]* seja < 45 graus.

    *Entrada*
        X, Y : coordenadas dos nós (len = N)
        phi  : orientação local fornecida pelo usuário (len = N)

    *Saída*
        alfas (len = N‑1) – ângulo normal para cada segmento
    """
    n = X.shape[0] - 1  # número de segmentos (faces)
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        dx = X[i+1] - X[i]
        dy = Y[i+1] - Y[i]
        theta = np.arctan2(dy, dx) * 180.0 / np.pi  # orientação da costa

        n1 = theta + 90.0  # primeira normal
        n2 = theta - 90.0  # normal oposta

        # diferença angular mínima (−180, 180]
        d1 = abs(((n1 - phi[i] + 180.0) % 360.0) - 180.0)
        d2 = abs(((n2 - phi[i] + 180.0) % 360.0) - 180.0)

        out[i] = n1 if d1 <= d2 else n2
    return out

@njit(fastmath=True, cache=True)
def hansonKraus1991(yi, dt, dx, hs, tp, dir, depth, doc, kal, X0, Y0, phi, bctype, Bcoef, formula, mb, D50):

    n1 = len(X0)
    mt, n2 = tp.shape

    # preallocate output and buffers
    ysol = np.zeros((mt, n1))
    q    = np.zeros((mt, n2))
    alfas = np.empty(n2, dtype=np.float64)      # reuse buffer for alfas

    ysol[0, :] = yi

    phi_rad = np.empty(n1, dtype=np.float64)
    for i in range(n1):
        phi_rad[i] = phi[i] * np.pi / 180.0
    
    for t in range(1, mt):
        # compute alfas once per time step
        XN, YN = abs_pos(X0, Y0, phi_rad, ysol[t-1,:])
        normals = _compute_normals(XN, YN, phi)
        # fill alfas buffer
        alfas[1:-1] = normals
        alfas[0] = normals[0]
        alfas[-1] = normals[-1]

        # propagate waves and compute transport
        hb, dirb, depthb = BreakingPropagation(hs[t,:], tp[t,:], dir[t,:], depth[t,:], alfas, Bcoef)
        # ALST(Hb, Tp, Dirb, hb, bathy_angle, K, D50, mb, formula)
        q_now, _ = ALST(hb, tp[t,:],dirb, depthb, alfas, kal, mb, D50, formula)

        # apply boundary conditions
        if bctype[0] == 0:
            q_now[0] = 0.0
        else:
            q_now[0] = q_now[1]
        if bctype[1] == 0:
            q_now[-1] = 0.0
        else:
            q_now[-1] = q_now[-2]

        # diffusion midpoints
        dc = 0.5 * (doc[t, 1:] + doc[t, :-1])

        ynew = np.empty(n1, dtype=np.float64)
        for i in range(n1):
            inv_i = dt[t-1] * 3600.0 / dx[i]
            ynew[i] = ysol[t-1, i] - inv_i * (q_now[i+1] - q_now[i]) / dc[i]


        ysol[t,:] = ynew
        q[t,:]     = q_now

    return ysol, q

# @jit
# def _extrapolate_ghost_nodes(X0, Y0):
#     dXL = X0[1] - X0[0]
#     dYL = Y0[1] - Y0[0]
#     dXR = X0[-1] - X0[-2]
#     dYR = Y0[-1] - Y0[-2]

#     n = X0.shape[0]
#     X0_calc = np.empty(n + 2, dtype=X0.dtype)
#     Y0_calc = np.empty(n + 2, dtype=Y0.dtype)
#     X0_calc[0] = X0[0] - dXL
#     Y0_calc[0] = Y0[0] - dYL
#     X0_calc[1:-1] = X0
#     Y0_calc[1:-1] = Y0
#     X0_calc[-1] = X0[-1] + dXR
#     Y0_calc[-1] = Y0[-1] + dYR
#     return X0_calc, Y0_calc

# @jit
# def _extend_1d(vec):
#     n = vec.shape[0]
#     ext = np.empty(n + 2, dtype=vec.dtype)
#     ext[0] = vec[0]
#     ext[1:-1] = vec
#     ext[-1] = vec[-1]
#     return ext

# @jit
# def _extend_forcing(mat):
#     nt, nc = mat.shape
#     out = np.empty((nt, nc + 2), dtype=mat.dtype)
#     for t in range(nt):
#         out[t, 0] = mat[t, 0]
#         out[t, -1] = mat[t, -1]
#         for j in range(nc):
#             out[t, j + 1] = mat[t, j]
#     return out

# ###############################################################################
# # 1. Núcleo dinâmico --------------------------------------------------------- #
# ###############################################################################

# # @jit
# def ydir_L(y, dt, dx_seg,
#            hs, tp, dire, depth, doc, kal,
#            X0, Y0, phi,
#            bctype, Bcoef):
#     """Avança um passo de tempo.

#     Retorna `ynew`, transporte nas faces (`q_now`) **e** ΔQ/Δs já calculado
#     sobre cada transecto (`dq_nodes`).
#     """
#     # --- Geometria local --------------------------------------------------
#     XN, YN = abs_pos(X0, Y0, phi * np.pi / 180.0, y)
    
#     # --- Ângulo de praia ---------------------------------------------------

#     alfas = np.zeros_like(hs)
#     alfas_ = _compute_normals(XN, YN, phi)
#     alfas[1:] = alfas_
#     alfas[0] = alfas[1]
#     alfas[-1] = alfas[-2]

#     # --- Ondas → transporte ----------------------------------------------
#     hb, dirb, depthb = BreakingPropagation(hs, tp, dire, depth, alfas, Bcoef)
#     hb[hb < 0.1] = 0.1  # não pode ser zero
#     depthb[hb < 0.1] = 0.1/Bcoef  # não pode ser zero
#     dc = 0.5 * (doc[1:] + doc[:-1])
#     q_now, q0 = ALST(hb, dirb, depthb, cartesianDir2nauticalDir(alfas), kal)

#     # --- BC em q_now ------------------------------------------------------
#     if bctype[0] == "Dirichlet":
#         q_now[0] = 0.0
#     else:
#         q_now[0] = q_now[1]
#     if bctype[1] == "Dirichlet":
#         q_now[-1] = 0.0
#     else:
#         q_now[-1] = q_now[-2]

#     # --- ΔQ sobre cada transecto (mesmo comprimento de y) -----------------
#     dq_nodes = np.zeros_like(y)
#     for i in range(1, y.shape[0]-1):
#         dq_nodes[i] = (q_now[i] - q_now[i-1])  # ΔQ (sem dividir por Δs)

#     # --- Estabilidade de Courant -----------------------------------------
#     # try:
#     if (dx_seg.min()**2 * dc.min() / (4.0 * q0.max())) < dt:
#         print("WARNING: COURANT CONDITION VIOLATED")
#         print('q max:' , q_now.max())
#         print('dire max:' , dire.max())
#         print('dire min:' , dire.min())
#         print('alfas max:' , alfas.max()+90)
#         print('alfas min:' , alfas.min()+90)
        
#     # except:
#     #     pass

#     # --- Atualiza linha de costa (η) nos nós ------------------------------
#     ynew = y.copy()
#     for i in range(1, y.shape[0]-1):
#         ynew[i] = y[i] - (dt * 3600.0) / dc[i-1] * dq_nodes[i] / dx_seg[i-1]

#     return ynew, q_now, dq_nodes

# ###############################################################################
# # 2. Função de alto nível ---------------------------------------------------- #
# ###############################################################################

# # @jit
# def hansonKraus1991(yi, dt, dx,
#                     hs, tp, dire, depth, doc, kal,
#                     X0, Y0, phi,
#                     bctype, Bcoef):
#     """Modelo one‑line com fantasmas, retornando ΔQ em cada transecto.

#     Saídas:
#         η  (ysol_trim) – shape (nt, len(X0))
#         ΔQ (dq_trim)   – mesma shape, ΔQ/Δs em cada transecto.
#     """
#     # --- 2.1  Criar malha estendida --------------------------------------
#     X0_calc, Y0_calc = _extrapolate_ghost_nodes(X0, Y0)
#     dx_seg = np.sqrt((X0_calc[1:] - X0_calc[:-1])**2 + (Y0_calc[1:] - Y0_calc[:-1])**2)
#     phi_calc = _extend_1d(phi)

#     hs_calc    = _extend_forcing(hs)
#     tp_calc    = _extend_forcing(tp)
#     dire_calc  = _extend_forcing(dire)
#     depth_calc = _extend_forcing(depth)
#     doc_calc   = _extend_forcing(doc)

#     nt, n_centers = hs_calc.shape  # n_centers = len(X0)+2

#     # --- 2.2  Alocar saídas ----------------------------------------------
#     ysol   = np.zeros((nt, n_centers))
#     dq_all = np.zeros((nt, n_centers))
#     q_face = np.zeros_like(hs_calc)

#     # --- 2.3  Condição inicial -------------------------------------------
#     ysol[0, 0]   = yi[0]
#     ysol[0, -1]  = yi[-1]
#     ysol[0, 1:-1] = yi

#     # --- 2.4  Loop temporal ----------------------------------------------
#     for n in range(nt-1):
#         (ysol[n+1, :],
#          q_face[n+1, :],
#          dq_all[n+1, :]) = ydir_L(
#             ysol[n, :], dt[n], dx_seg,
#             hs_calc[n+1, :], tp_calc[n+1, :], dire_calc[n+1, :],
#             depth_calc[n+1, :], doc_calc[n+1, :], kal,
#             X0_calc, Y0_calc, phi_calc,
#             bctype, Bcoef)

#     # --- 2.5  Remover fantasmas antes de retornar -------------------------
#     eta_nodes = ysol[:, 1:-1]
#     dq_nodes  = dq_all[:, 1:-1]
#     return eta_nodes, dq_nodes
