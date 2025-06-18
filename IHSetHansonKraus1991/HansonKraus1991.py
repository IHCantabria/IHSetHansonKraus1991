import numpy as np
from numba import njit
# from datetime import datetime
from IHSetUtils.libjit.geometry import nauticalDir2cartesianDir, abs_pos, shore_angle
from IHSetUtils.libjit.waves import BreakingPropagation
from IHSetUtils.libjit.morfology import CERQ_ALST, Komar_ALST, Kamphuis_ALST, VanRijn_ALST
import math


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
def hansonKraus1991_cerq(yi, dt, dx, hs, tp, dir, depth, doc, kal, X0, Y0, phi, bctype, Bcoef, mb, D50):

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
        hb, dirb, depthb = BreakingPropagation(hs[t,:], tp[t,:], dir[t,:], depth, alfas, Bcoef)
        # ALST(Hb, Tp, Dirb, hb, bathy_angle, K, D50, mb, formula)
        q_now, _ = CERQ_ALST(hb, dirb, depthb, alfas, kal)

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


@njit(fastmath=True, cache=True)
def hansonKraus1991_komar(yi, dt, dx, hs, tp, dir, depth, doc, kal, X0, Y0, phi, bctype, Bcoef, mb, D50):

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
        hb, dirb, depthb = BreakingPropagation(hs[t,:], tp[t,:], dir[t,:], depth, alfas, Bcoef)
        # ALST(Hb, Tp, Dirb, hb, bathy_angle, K, D50, mb, formula)
        q_now, _ = Komar_ALST(hb, dirb, depthb, alfas, kal)

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


@njit(fastmath=True, cache=True)
def hansonKraus1991_kamphuis(yi, dt, dx, hs, tp, dir, depth, doc, kal, X0, Y0, phi, bctype, Bcoef, mb, D50):

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
        hb, dirb, depthb = BreakingPropagation(hs[t,:], tp[t,:], dir[t,:], depth, alfas, Bcoef)
        # ALST(Hb, Tp, Dirb, hb, bathy_angle, K, D50, mb, formula)
        q_now, _ = Kamphuis_ALST(hb, tp[t,:], dirb, depthb, alfas, kal, mb, D50)

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


@njit(fastmath=True, cache=True)
def hansonKraus1991_vanrijn(yi, dt, dx, hs, tp, dir, depth, doc, kal, X0, Y0, phi, bctype, Bcoef, mb, D50):

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
        hb, dirb, depthb = BreakingPropagation(hs[t,:], tp[t,:], dir[t,:], depth, alfas, Bcoef)
        # ALST(Hb, Tp, Dirb, hb, bathy_angle, K, D50, mb, formula)
        q_now, _ = VanRijn_ALST(hb, dirb, depthb, alfas, kal, mb, D50)

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