import numpy as np
from numba import njit
# from datetime import datetime
from IHSetUtils.libjit.geometry import nauticalDir2cartesianDir, abs_pos, shore_angle
from IHSetUtils.libjit.waves import BreakingPropagation
import math

@njit(cache=True)
def extrapolate_baseline(X0, Y0, dx):
    """
    Recebe:
      X0, Y0 : arrays (n1,) da linha base original
      dx     : array (n1,) dos comprimentos de segmento
    Devolve:
      X0_p, Y0_p : arrays (n1+2,) com um ponto fantasma em cada extremidade,
                   extrapolado ao longo da linha base.
    """
    n1 = X0.shape[0]
    X0_p = np.empty(n1+1, dtype=np.float64)
    Y0_p = np.empty(n1+1, dtype=np.float64)

    # 1) Primeiro segmento
    dx0 = X0[1] - X0[0]
    dy0 = Y0[1] - Y0[0]
    L0  = math.hypot(dx0, dy0)
    tx0 = dx0 / L0
    ty0 = dy0 / L0
    # ponto fantasma “antes” de X0[0]
    X0_p[0] = X0[0] - tx0 * dx[0]/2
    Y0_p[0] = Y0[0] - ty0 * dx[0]/2

    # 2) interior = segmentos intermediários
    for i in range(n1-1):
        dxi = X0[i+1] - X0[i]
        dyi = Y0[i+1] - Y0[i]
        Li  = math.hypot(dxi, dyi)
        txi = dxi / Li
        tyi = dyi / Li
        X0_p[i+1] = X0[i] + txi * dx[i]/2 
        Y0_p[i+1] = Y0[i] + tyi * dx[i]/2

    # 3) Último segmento
    dxE = X0[-1] - X0[-2]
    dyE = Y0[-1] - Y0[-2]
    LE  = math.hypot(dxE, dyE)
    txE = dxE / LE
    tyE = dyE / LE
    # ponto fantasma “após” X0[-1]
    X0_p[-1] = X0[-1] + txE * dx[-1]/2
    Y0_p[-1] = Y0[-1] + tyE * dx[-1]/2

    return X0_p, Y0_p


@njit(fastmath=True, cache=True)
def _compute_normals(X, Y, phi):
    """
    Devolve vetor de ângulos **normais** (perpendiculares) à costa, em graus.

    Para cada segmento entre os nós i‑i+1 calcula-se duas normais (θ±90°) e escolhe-se
    aquela cuja diferença angular circular em relação a phi[i] seja mínima.

    Entrada:
        X, Y : coordenadas dos nós (len = N)
        phi  : orientação local fornecida pelo usuário (len = N)

    Saída:
        alfas (len = N‑1) – ângulo normal para cada segmento
    """
    n = X.shape[0] - 1
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        # orientação da costa (°)
        theta = np.arctan2(Y[i+1] - Y[i], X[i+1] - X[i]) * 180.0 / np.pi
        # duas normais candidatas
        n1 = (theta + 90.0) % 360.0
        n2 = (theta - 90.0) % 360.0
        # diferença angular circular mínima
        # δ = (a - b + 180) % 360 - 180 ∈ (−180, 180]
        d1 = (n1 - phi[i] + 180.0) % 360.0 - 180.0
        d2 = (n2 - phi[i] + 180.0) % 360.0 - 180.0
        # escolhe a normal de menor desvio absoluto
        out[i] = n1 if abs(d1) <= abs(d2) else n2

    return out

@njit(fastmath=True, cache=True)
def hansonKraus1991(yi, dt,  hs, tp, dire, depth, doc, kal, X0, Y0, phi, bctype, Bcoef, mb, D50, lstf):

    n1 = len(X0)
    mt, n2 = tp.shape

    # preallocate output and buffers
    ysol  = np.zeros((mt, n1))
    q     = np.zeros((mt, n2))
    alfas = np.empty(n2, dtype=np.float64)      # reuse buffer for alfas
    ynew  = np.empty(n1, dtype=np.float64)

    ysol[0, :] = yi
    
    dx = np.sqrt((X0[1:] - X0[:-1])**2 + (Y0[1:] - Y0[:-1])**2)
    dx = np.hstack((np.array([dx[0]]), dx))  # add ghost points at the ends

    phi_rad = np.empty(n1, dtype=np.float64)

    for i in range(n1):
        phi_rad[i] = phi[i] * np.pi / 180.0
    
    for t in range(1, mt):

        # compute alfas once per time step
        XN, YN  = abs_pos(X0, Y0, phi_rad, ysol[t-1,:])
        normals = _compute_normals(XN, YN, phi)
        dx      = ((XN[1:] - XN[:-1])**2 + (YN[1:] - YN[:-1])**2)**0.5

        # fill alfas buffer
        alfas[1:-1] = normals
        alfas[0]    = normals[0]
        alfas[-1]   = normals[-1]
        
        # propagate waves and compute transport
        hb, dirb, depthb = BreakingPropagation(hs[t,:], tp[t,:], dire[t,:], depth, alfas, Bcoef)
        # (Hb, Tp, Dirb, hb, bathy_angle, K, mb, D50)
        q_now, _ = lstf(hb, tp[t,:], dirb, depthb, alfas, kal, mb, D50)

        # apply boundary conditions
        if bctype[0]  == 0:
            q_now[0]  = 0.0
        else:
            q_now[0]  = q_now[1]
        if bctype[1]  == 0:
            q_now[-1] = 0.0
        else:
            q_now[-1] = q_now[-2]

        # diffusion midpoints
        dc = 0.5 * (doc[t, 1:] + doc[t, :-1])

        inv_i = dt[t-1] / dc[0]  # inverse of dc[1] for first element
        ynew[0] = ysol[t-1, 0] - inv_i * (q_now[1] - q_now[0]) / dx[0]

        inv_i = dt[t-1] / dc[-1]  # inverse of dc[-1] for last element
        ynew[-1] = ysol[t-1, -1] - inv_i * (q_now[-1] - q_now[-2]) / dx[-1]

        for i in range(1,n2-2):
            inv_i   = dt[t-1] / dc[i]  # inverse of dx[i+1] for current element
            ynew[i] = ysol[t-1,i] - inv_i * (q_now[i+1] - q_now[i]) / dx[i-1]

        ysol[t,:]  = ynew

        q[t,:]     = q_now  

    return ysol, q



# @njit(fastmath=True, cache=True)
# def hansonKraus1991_cerq(yi, dt,  hs, tp, dire, depth, doc, kal, X0, Y0, phi, bctype, Bcoef, mb, D50):

#     n1 = len(X0)
#     mt, n2 = tp.shape

#     # preallocate output and buffers
#     ysol  = np.zeros((mt, n1))
#     q     = np.zeros((mt, n2))
#     alfas = np.empty(n2, dtype=np.float64)      # reuse buffer for alfas
#     ynew  = np.empty(n1, dtype=np.float64)

#     ysol[0, :] = yi
    
#     dx = np.sqrt((X0[1:] - X0[:-1])**2 + (Y0[1:] - Y0[:-1])**2)
#     dx = np.hstack((np.array([dx[0]]), dx))  # add ghost points at the ends

#     phi_rad = np.empty(n1, dtype=np.float64)

#     for i in range(n1):
#         phi_rad[i] = phi[i] * np.pi / 180.0
    
#     for t in range(1, mt):

#         # compute alfas once per time step
#         XN, YN  = abs_pos(X0, Y0, phi_rad, ysol[t-1,:])
#         normals = _compute_normals(XN, YN, phi)
#         dx      = ((XN[1:] - XN[:-1])**2 + (YN[1:] - YN[:-1])**2)**0.5

#         # fill alfas buffer
#         alfas[1:-1] = normals
#         alfas[0]    = normals[0]
#         alfas[-1]   = normals[-1]
        
#         # propagate waves and compute transport
#         hb, dirb, depthb = BreakingPropagation(hs[t,:], tp[t,:], dire[t,:], depth, alfas, Bcoef)
#         # ALST(Hb, Tp, Dirb, hb, bathy_angle, K, D50, mb, formula)
#         q_now, _         = CERQ_ALST(hb, dirb, depthb, alfas, kal)

#         # apply boundary conditions
#         if bctype[0]  == 0:
#             q_now[0]  = 0.0
#         else:
#             q_now[0]  = q_now[1]
#         if bctype[1]  == 0:
#             q_now[-1] = 0.0
#         else:
#             q_now[-1] = q_now[-2]

#         # diffusion midpoints
#         dc = 0.5 * (doc[t, 1:] + doc[t, :-1])

#         inv_i = dt[t-1] / dc[0]  # inverse of dc[1] for first element
#         ynew[0] = ysol[t-1, 0] - inv_i * (q_now[1] - q_now[0]) / dx[0]

#         inv_i = dt[t-1] / dc[-1]  # inverse of dc[-1] for last element
#         ynew[-1] = ysol[t-1, -1] - inv_i * (q_now[-1] - q_now[-2]) / dx[-1]

#         for i in range(1,n2-2):
#             inv_i   = dt[t-1] / dc[i]  # inverse of dx[i+1] for current element
#             ynew[i] = ysol[t-1,i] - inv_i * (q_now[i+1] - q_now[i]) / dx[i-1]

#         ysol[t,:]  = ynew

#         q[t,:]     = q_now  

#     return ysol, q


# @njit(fastmath=True, cache=True)
# def hansonKraus1991_komar(yi, dt,  hs, tp, dire, depth, doc, kal, X0, Y0, phi, bctype, Bcoef, mb, D50):

#     n1 = len(X0)
#     mt, n2 = tp.shape

#     # preallocate output and buffers
#     ysol  = np.zeros((mt, n1))
#     q     = np.zeros((mt, n2))
#     alfas = np.empty(n2, dtype=np.float64)      # reuse buffer for alfas
#     ynew  = np.empty(n1, dtype=np.float64)

#     ysol[0, :] = yi
    
#     dx = np.sqrt((X0[1:] - X0[:-1])**2 + (Y0[1:] - Y0[:-1])**2)
#     dx = np.hstack((np.array([dx[0]]), dx))  # add ghost points at the ends

#     phi_rad = np.empty(n1, dtype=np.float64)

#     for i in range(n1):
#         phi_rad[i] = phi[i] * np.pi / 180.0
    
#     for t in range(1, mt):

#         # compute alfas once per time step
#         XN, YN  = abs_pos(X0, Y0, phi_rad, ysol[t-1,:])
#         normals = _compute_normals(XN, YN, phi)
#         dx      = ((XN[1:] - XN[:-1])**2 + (YN[1:] - YN[:-1])**2)**0.5

#         # fill alfas buffer
#         alfas[1:-1] = normals
#         alfas[0]    = normals[0]
#         alfas[-1]   = normals[-1]
        
#         # propagate waves and compute transport
#         hb, dirb, depthb = BreakingPropagation(hs[t,:], tp[t,:], dire[t,:], depth, alfas, Bcoef)
#         # ALST(Hb, Tp, Dirb, hb, bathy_angle, K, D50, mb, formula)
#         q_now, _         = Komar_ALST(hb, dirb, depthb, alfas, kal)

#         # apply boundary conditions
#         if bctype[0]  == 0:
#             q_now[0]  = 0.0
#         else:
#             q_now[0]  = q_now[1]
#         if bctype[1]  == 0:
#             q_now[-1] = 0.0
#         else:
#             q_now[-1] = q_now[-2]

#         # diffusion midpoints
#         dc = 0.5 * (doc[t, 1:] + doc[t, :-1])

#         inv_i = dt[t-1] / dc[0]  # inverse of dc[1] for first element
#         ynew[0] = ysol[t-1, 0] - inv_i * (q_now[1] - q_now[0]) / dx[0]

#         inv_i = dt[t-1] / dc[-1]  # inverse of dc[-1] for last element
#         ynew[-1] = ysol[t-1, -1] - inv_i * (q_now[-1] - q_now[-2]) / dx[-1]

#         for i in range(1,n2-2):
#             inv_i   = dt[t-1] / dc[i]  # inverse of dx[i+1] for current element
#             ynew[i] = ysol[t-1,i] - inv_i * (q_now[i+1] - q_now[i]) / dx[i-1]

#         ysol[t,:]  = ynew

#         q[t,:]     = q_now  

#     return ysol, q


# @njit(fastmath=True, cache=True)
# def hansonKraus1991_kamphuis(yi, dt,  hs, tp, dire, depth, doc, kal, X0, Y0, phi, bctype, Bcoef, mb, D50):

#     n1 = len(X0)
#     mt, n2 = tp.shape

#     # preallocate output and buffers
#     ysol  = np.zeros((mt, n1))
#     q     = np.zeros((mt, n2))
#     alfas = np.empty(n2, dtype=np.float64)      # reuse buffer for alfas
#     ynew  = np.empty(n1, dtype=np.float64)

#     ysol[0, :] = yi
    
#     dx = np.sqrt((X0[1:] - X0[:-1])**2 + (Y0[1:] - Y0[:-1])**2)
#     dx = np.hstack((np.array([dx[0]]), dx))  # add ghost points at the ends

#     phi_rad = np.empty(n1, dtype=np.float64)

#     for i in range(n1):
#         phi_rad[i] = phi[i] * np.pi / 180.0
    
#     for t in range(1, mt):

#         # compute alfas once per time step
#         XN, YN  = abs_pos(X0, Y0, phi_rad, ysol[t-1,:])
#         normals = _compute_normals(XN, YN, phi)
#         dx      = ((XN[1:] - XN[:-1])**2 + (YN[1:] - YN[:-1])**2)**0.5

#         # fill alfas buffer
#         alfas[1:-1] = normals
#         alfas[0]    = normals[0]
#         alfas[-1]   = normals[-1]
        
#         # propagate waves and compute transport
#         hb, dirb, depthb = BreakingPropagation(hs[t,:], tp[t,:], dire[t,:], depth, alfas, Bcoef)
#         # ALST(Hb, Tp, Dirb, hb, bathy_angle, K, D50, mb, formula)
#         q_now, _         = Kamphuis_ALST(hb, tp[t,:], dirb, depthb, alfas, kal, mb, D50)

#         # apply boundary conditions
#         if bctype[0]  == 0:
#             q_now[0]  = 0.0
#         else:
#             q_now[0]  = q_now[1]
#         if bctype[1]  == 0:
#             q_now[-1] = 0.0
#         else:
#             q_now[-1] = q_now[-2]

#         # diffusion midpoints
#         dc = 0.5 * (doc[t, 1:] + doc[t, :-1])

#         inv_i = dt[t-1] / dc[0]  # inverse of dc[1] for first element
#         ynew[0] = ysol[t-1, 0] - inv_i * (q_now[1] - q_now[0]) / dx[0]

#         inv_i = dt[t-1] / dc[-1]  # inverse of dc[-1] for last element
#         ynew[-1] = ysol[t-1, -1] - inv_i * (q_now[-1] - q_now[-2]) / dx[-1]

#         for i in range(1,n2-2):
#             inv_i   = dt[t-1] / dc[i]  # inverse of dx[i+1] for current element
#             ynew[i] = ysol[t-1,i] - inv_i * (q_now[i+1] - q_now[i]) / dx[i-1]

#         ysol[t,:]  = ynew

#         q[t,:]     = q_now  

#     return ysol, q


# @njit(fastmath=True, cache=True)
# def hansonKraus1991_vanrijn(yi, dt,  hs, tp, dire, depth, doc, kal, X0, Y0, phi, bctype, Bcoef, mb, D50):

#     n1 = len(X0)
#     mt, n2 = tp.shape

#     # preallocate output and buffers
#     ysol  = np.zeros((mt, n1))
#     q     = np.zeros((mt, n2))
#     alfas = np.empty(n2, dtype=np.float64)      # reuse buffer for alfas
#     ynew  = np.empty(n1, dtype=np.float64)

#     ysol[0, :] = yi
    
#     dx = np.sqrt((X0[1:] - X0[:-1])**2 + (Y0[1:] - Y0[:-1])**2)
#     dx = np.hstack((np.array([dx[0]]), dx))  # add ghost points at the ends

#     phi_rad = np.empty(n1, dtype=np.float64)

#     for i in range(n1):
#         phi_rad[i] = phi[i] * np.pi / 180.0
    
#     for t in range(1, mt):

#         # compute alfas once per time step
#         XN, YN  = abs_pos(X0, Y0, phi_rad, ysol[t-1,:])
#         normals = _compute_normals(XN, YN, phi)
#         dx      = ((XN[1:] - XN[:-1])**2 + (YN[1:] - YN[:-1])**2)**0.5

#         # fill alfas buffer
#         alfas[1:-1] = normals
#         alfas[0]    = normals[0]
#         alfas[-1]   = normals[-1]
        
#         # propagate waves and compute transport
#         hb, dirb, depthb = BreakingPropagation(hs[t,:], tp[t,:], dire[t,:], depth, alfas, Bcoef)
#         # ALST(Hb, Tp, Dirb, hb, bathy_angle, K, D50, mb, formula)
#         q_now, _         = VanRijn_ALST(hb, dirb, depthb, alfas, kal, mb, D50)

#         # apply boundary conditions
#         if bctype[0]  == 0:
#             q_now[0]  = 0.0
#         else:
#             q_now[0]  = q_now[1]
#         if bctype[1]  == 0:
#             q_now[-1] = 0.0
#         else:
#             q_now[-1] = q_now[-2]

#         # diffusion midpoints
#         dc = 0.5 * (doc[t, 1:] + doc[t, :-1])

#         inv_i = dt[t-1] / dc[0]  # inverse of dc[1] for first element
#         ynew[0] = ysol[t-1, 0] - inv_i * (q_now[1] - q_now[0]) / dx[0]

#         inv_i = dt[t-1] / dc[-1]  # inverse of dc[-1] for last element
#         ynew[-1] = ysol[t-1, -1] - inv_i * (q_now[-1] - q_now[-2]) / dx[-1]

#         for i in range(1,n2-2):
#             inv_i   = dt[t-1] / dc[i]  # inverse of dx[i+1] for current element
#             ynew[i] = ysol[t-1,i] - inv_i * (q_now[i+1] - q_now[i]) / dx[i-1]

#         ysol[t,:]  = ynew

#         q[t,:]     = q_now  

#     return ysol, q





# @njit(fastmath=True, cache=True)
# def hansonKraus1991_cerq(yi, dt,  hs, tp, dire, depth, doc, kal, X0, Y0, phi, bctype, Bcoef, mb, D50): #dx,

#     n1 = len(X0)
#     mt, n2 = tp.shape

#     # preallocate output and buffers
#     ysol  = np.zeros((mt, n1))
#     q     = np.zeros((mt, n2))
#     alfas = np.empty(n2, dtype=np.float64)      # reuse buffer for alfas
#     ynew = np.empty(n1, dtype=np.float64)
    
#     ysol[0, :] = yi

#     phi_rad = np.empty(n1, dtype=np.float64)

#     for i in range(n1):
#         phi_rad[i] = phi[i] * np.pi / 180.0
    
#     for t in range(1, mt):

#         # compute alfas once per time step
#         XN, YN  = abs_pos(X0, Y0, phi_rad, ysol[t-1,:])
#         normals = _compute_normals(XN, YN, phi)
#         dx      = ((XN[1:] - XN[:-1])**2 + (YN[1:] - YN[:-1])**2)**0.5

#         # fill alfas buffer
#         alfas[1:-1] = normals
#         alfas[0]    = normals[0]
#         alfas[-1]   = normals[-1]
        
#         # propagate waves and compute transport
#         hb, dirb, depthb = BreakingPropagation(hs[t,:], tp[t,:], dire[t,:], depth, alfas, Bcoef)
#         # ALST(Hb, Tp, Dirb, hb, bathy_angle, K, D50, mb, formula)
#         q_now, _         = CERQ_ALST(hb, dirb, depthb, alfas, kal)

#         # apply boundary conditions
#         if bctype[0]  == 0:
#             q_now[0]  = 0.0
#         else:
#             q_now[0]  = q_now[1]
#         if bctype[1]  == 0:
#             q_now[-1] = 0.0
#         else:
#             q_now[-1] = q_now[-2]

#         # diffusion midpoints
#         dc = 0.5 * (doc[t, 1:] + doc[t, :-1])

#         inv_i = dt[t-1] / dc[0]  # inverse of dx[1] for first element
#         ynew[0] = ysol[t-1, 0] - inv_i * (q_now[1] - q_now[0]) / dx[0]

#         for i in range(1,n1):
#             inv_i   = dt[t-1] / dc[i]  # inverse of dx[i+1] for current element
#             ynew[i] = ysol[t-1,i] - inv_i * (q_now[i+1] - q_now[i]) / dx[i-1]

#         ysol[t,:]  = ynew

#         q[t,:]     = q_now  

#     return ysol , q

# @njit(fastmath=True, cache=True)
# def hansonKraus1991_komar(yi, dt, dx, hs, tp, dir, depth, doc, kal, X0, Y0, phi, bctype, Bcoef, mb, D50):

#     n1 = len(X0)
#     mt, n2 = tp.shape

#     # preallocate output and buffers
#     ysol = np.zeros((mt, n1))
#     q    = np.zeros((mt, n2))
#     alfas = np.empty(n2, dtype=np.float64)      # reuse buffer for alfas

#     ysol[0, :] = yi

#     phi_rad = np.empty(n1, dtype=np.float64)
#     for i in range(n1):
#         phi_rad[i] = phi[i] * np.pi / 180.0
    
#     for t in range(1, mt):
#         # compute alfas once per time step
#         XN, YN = abs_pos(X0, Y0, phi_rad, ysol[t-1,:])
#         normals = _compute_normals(XN, YN, phi)
#         # fill alfas buffer
#         alfas[1:-1] = normals
#         alfas[0] = normals[0]
#         alfas[-1] = normals[-1]

#         # propagate waves and compute transport
#         hb, dirb, depthb = BreakingPropagation(hs[t,:], tp[t,:], dir[t,:], depth, alfas, Bcoef)
#         # ALST(Hb, Tp, Dirb, hb, bathy_angle, K, D50, mb, formula)
#         q_now, _ = Komar_ALST(hb, dirb, depthb, alfas, kal)

#         # apply boundary conditions
#         if bctype[0] == 0:
#             q_now[0] = 0.0
#         else:
#             q_now[0] = q_now[1]
#         if bctype[1] == 0:
#             q_now[-1] = 0.0
#         else:
#             q_now[-1] = q_now[-2]

#         # diffusion midpoints
#         dc = 0.5 * (doc[t, 1:] + doc[t, :-1])

#         ynew = np.empty(n1, dtype=np.float64)
#         for i in range(n1):
#             inv_i = dt[t-1] * 3600.0 / dx[i]
#             ynew[i] = ysol[t-1, i] - inv_i * (q_now[i+1] - q_now[i]) / dc[i]


#         ysol[t,:] = ynew
#         q[t,:]     = q_now

#     return ysol, q


# @njit(fastmath=True, cache=True)
# def hansonKraus1991_kamphuis(yi, dt, dx, hs, tp, dir, depth, doc, kal, X0, Y0, phi, bctype, Bcoef, mb, D50):

#     n1 = len(X0)
#     mt, n2 = tp.shape

#     # preallocate output and buffers
#     ysol = np.zeros((mt, n1))
#     q    = np.zeros((mt, n2))
#     alfas = np.empty(n2, dtype=np.float64)      # reuse buffer for alfas

#     ysol[0, :] = yi

#     phi_rad = np.empty(n1, dtype=np.float64)
#     for i in range(n1):
#         phi_rad[i] = phi[i] * np.pi / 180.0
    
#     for t in range(1, mt):
#         # compute alfas once per time step
#         XN, YN = abs_pos(X0, Y0, phi_rad, ysol[t-1,:])
#         normals = _compute_normals(XN, YN, phi)
#         # fill alfas buffer
#         alfas[1:-1] = normals
#         alfas[0] = normals[0]
#         alfas[-1] = normals[-1]

#         # propagate waves and compute transport
#         hb, dirb, depthb = BreakingPropagation(hs[t,:], tp[t,:], dir[t,:], depth, alfas, Bcoef)
#         # ALST(Hb, Tp, Dirb, hb, bathy_angle, K, D50, mb, formula)
#         q_now, _ = Kamphuis_ALST(hb, tp[t,:], dirb, depthb, alfas, kal, mb, D50)

#         # apply boundary conditions
#         if bctype[0] == 0:
#             q_now[0] = 0.0
#         else:
#             q_now[0] = q_now[1]
#         if bctype[1] == 0:
#             q_now[-1] = 0.0
#         else:
#             q_now[-1] = q_now[-2]

#         # diffusion midpoints
#         dc = 0.5 * (doc[t, 1:] + doc[t, :-1])

#         ynew = np.empty(n1, dtype=np.float64)
#         for i in range(n1):
#             inv_i = dt[t-1] * 3600.0 / dx[i]
#             ynew[i] = ysol[t-1, i] - inv_i * (q_now[i+1] - q_now[i]) / dc[i]


#         ysol[t,:] = ynew
#         q[t,:]     = q_now

#     return ysol, q


# @njit(fastmath=True, cache=True)
# def hansonKraus1991_vanrijn(yi, dt, dx, hs, tp, dir, depth, doc, kal, X0, Y0, phi, bctype, Bcoef, mb, D50):

#     n1 = len(X0)
#     mt, n2 = tp.shape

#     # preallocate output and buffers
#     ysol = np.zeros((mt, n1))
#     q    = np.zeros((mt, n2))
#     alfas = np.empty(n2, dtype=np.float64)      # reuse buffer for alfas

#     ysol[0, :] = yi

#     phi_rad = np.empty(n1, dtype=np.float64)
#     for i in range(n1):
#         phi_rad[i] = phi[i] * np.pi / 180.0
    
#     for t in range(1, mt):
#         # compute alfas once per time step
#         XN, YN = abs_pos(X0, Y0, phi_rad, ysol[t-1,:])
#         normals = _compute_normals(XN, YN, phi)
#         # fill alfas buffer
#         alfas[1:-1] = normals
#         alfas[0] = normals[0]
#         alfas[-1] = normals[-1]

#         # propagate waves and compute transport
#         hb, dirb, depthb = BreakingPropagation(hs[t,:], tp[t,:], dir[t,:], depth, alfas, Bcoef)
#         # ALST(Hb, Tp, Dirb, hb, bathy_angle, K, D50, mb, formula)
#         q_now, _ = VanRijn_ALST(hb, dirb, depthb, alfas, kal, mb, D50)

#         # apply boundary conditions
#         if bctype[0] == 0:
#             q_now[0] = 0.0
#         else:
#             q_now[0] = q_now[1]
#         if bctype[1] == 0:
#             q_now[-1] = 0.0
#         else:
#             q_now[-1] = q_now[-2]

#         # diffusion midpoints
#         dc = 0.5 * (doc[t, 1:] + doc[t, :-1])

#         ynew = np.empty(n1, dtype=np.float64)
#         for i in range(n1):
#             inv_i = dt[t-1] * 3600.0 / dx[i]
#             ynew[i] = ysol[t-1, i] - inv_i * (q_now[i+1] - q_now[i]) / dc[i]


#         ysol[t,:] = ynew
#         q[t,:]     = q_now

#     return ysol, q