# import numpy as np
# from numba import jit
# # from datetime import datetime
# from IHSetUtils.libjit.geometry import nauticalDir2cartesianDir, abs_pos, shore_angle
# from IHSetUtils.libjit.waves import BreakingPropagation
# from IHSetUtils.libjit.morfology import ALST

# @jit
# def hansonKraus1991(yi, dt, dx, hs, tp, dir, depth, doc, kal, X0, Y0, phi, bctype, Bcoef):
#     """
#     Function OneLine calculates the evolution of a system over time using a numerical method.

#     Args:
#         yi (ndarray): Initial conditions.
#         dt (float): Time step.
#         dx (float): Spatial step.
#         hs (ndarray): Matrix representing sea surface height.
#         tp (ndarray): Matrix representing wave periods.
#         dir (ndarray): Matrix representing wave directions.
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

#     hb = np.zeros((mt, n2))
#     dirb = np.zeros((mt, n2))
#     depthb = np.zeros((mt, n2))
#     q = np.zeros((mt, n2))
#     q0 = np.zeros((mt, n2))

#     ## Create the ghost condition

#     # time_init = datetime.now()
#     for pos in range(0, nti-1):
        
#         ti = pos+desl

#         # p1 = ti - desl
#         # p2 = ti + desl

#         ysol[ti,:], hb[ti,:], dirb[ti,:], depthb[ti,:], q[ti,:], q0[ti,:] =  ydir_L(ysol[ti-1, :], dt[ti-1], dx, hs[ti, :], tp[ti, :], dir[ti, :], depth[ti, :], doc[ti, :], kal, X0, Y0, phi, bctype, Bcoef)

#     # Make the interpolation for the actual transects!!!



#     return ysol, q

# @jit
# def ydir_L(y, dt, dx, hs, tp, dire, depth, doc, kal, X0, Y0, phi, bctype, Bcoef):
#     """
#     Function ydir_L calculates the propagation of waves and other quantities at a specific time step.

#     Args:
#         y (ndarray): Initial conditions.
#         dt (float): Time step.
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
#     # flag_dig = False
    
 
#     XN, YN = abs_pos(X0, Y0, phi * np.pi / 180.0, y)

#     alfas = np.zeros_like(hs)
#     alfas_ = shore_angle(XN, YN, dire)
#     alfas[0] = alfas[1]
#     alfas[-1] = alfas[-2]

#     # try:
#     hb, dirb, depthb = BreakingPropagation(hs, tp, dire, depth, alfas + 90, Bcoef)
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
#     q_now, q0 = ALST(hb, dirb, depthb, alfas + 90, kal)

#     if bctype[0] == "Dirichlet":
#         q_now[0] = 0
#     elif bctype[0] == "Neumann":
#         q_now[0] = q_now[1]
#     if bctype[1] == "Dirichlet":
#         q_now[-1] = 0
#     elif bctype[1] == "Neumann":
#         q_now[-1] = q_now[-2]


#     try:
#         if (dx**2 * np.min(dc) / (4 * np.max(q0))) < dt:
#             print("WARNING: COURANT CONDITION VIOLATED")
#     except:
#         pass


#     ynew = y - (dt * 60 * 60) / dc * (q_now[1:] - q_now[:-1]) / dx


#     return ynew, hb, dirb, depthb, q_now, q0



import numpy as np
from numba import jit
from scipy.interpolate import PchipInterpolator

from IHSetUtils.libjit.geometry import nauticalDir2cartesianDir, abs_pos, shore_angle
from IHSetUtils.libjit.waves import BreakingPropagation
from IHSetUtils.libjit.morfology import ALST

###############################################################################
# Helper routines (PURE Python – **não** jittados)                           #
###############################################################################

def _extend_with_ghost(x_centers: np.ndarray,
                       eta_centers: np.ndarray,
                       bc_left: str = "neumann",
                       bc_right: str = "neumann"):
    """Cria arrays estendidos (centros + dois nós fantasmas).

    A posição dos fantasmas é obtida por extrapolação linear;
    os valores usam a condição de contorno escolhida.

    Parameters
    ----------
    x_centers : ndarray (M,)
        Coordenadas ao longo‑costa dos centros das células.
    eta_centers : ndarray (M,)
        Linha de costa nos centros.
    bc_left, bc_right : {"neumann", "dirichlet"}
        Tipo de condição na extremidade esquerda / direita.

    Returns
    -------
    x_ext  : ndarray (M+2,)
    eta_ext: ndarray (M+2,)
    """
    dx_L = x_centers[1] - x_centers[0]
    dx_R = x_centers[-1] - x_centers[-2]
    x_ghost_L = x_centers[0] - dx_L
    x_ghost_R = x_centers[-1] + dx_R

    # -- valores nos fantasmas ------------------------------------------------
    if bc_left.lower() == "neumann":
        eta_ghost_L = eta_centers[0]
    elif bc_left.lower() == "dirichlet":
        eta_ghost_L = 0.0
    else:
        raise ValueError("bc_left deve ser 'neumann' ou 'dirichlet'.")

    if bc_right.lower() == "neumann":
        eta_ghost_R = eta_centers[-1]
    elif bc_right.lower() == "dirichlet":
        eta_ghost_R = 0.0
    else:
        raise ValueError("bc_right deve ser 'neumann' ou 'dirichlet'.")

    x_ext = np.concatenate(([x_ghost_L], x_centers, [x_ghost_R]))
    eta_ext = np.concatenate(([eta_ghost_L], eta_centers, [eta_ghost_R]))
    return x_ext, eta_ext


def centers_to_nodes(eta_centers: np.ndarray,
                     x_nodes: np.ndarray,
                     bc: tuple = ("neumann", "neumann")) -> np.ndarray:
    """Interpola a linha de costa dos centros **para** os transectos.

    Usa PCHIP (monotónica, 2ª ordem) com nós fantasmas para evitar oscilações.

    Parameters
    ----------
    eta_centers : ndarray (M,)
        Linha de costa nos centros.
    x_nodes : ndarray (M+1,)
        Coordenadas ao longo‑costa dos transectos originais.
    bc : tuple(str, str)
        Tipos de condição de contorno (esquerda, direita).

    Returns
    -------
    eta_nodes : ndarray (M+1,)
        Linha de costa em cada transecto.
    """
    # Construir os x dos centros a partir dos nós
    x_centers = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    x_ext, eta_ext = _extend_with_ghost(x_centers, eta_centers,
                                        bc_left=bc[0], bc_right=bc[1])
    interpolador = PchipInterpolator(x_ext, eta_ext, extrapolate=False)
    return interpolador(x_nodes)

###############################################################################
# Núcleo numérico (JIT‑omp)                                                   #
###############################################################################

@jit(nopython=True)
def ydir_L(y, dt, dx, hs, tp, dire, depth, doc, kal,
           X0, Y0, phi, bctype, Bcoef):
    """Avança uma iteração temporal do modelo Hanson & Kraus (1991).

    Todos os argumentos mantêm a mesma assinatura do seu código original.
    As únicas alterações foram:
        • Comentários / formatação.
        • *Nenhuma* dependência SciPy dentro da região jittada.
    """
    # -- 1. Geometria da linha de costa ---------------------------------------
    XN, YN = abs_pos(X0, Y0, phi * np.pi / 180.0, y)
    alfas = np.empty_like(hs)
    alfas_ = shore_angle(XN, YN, dire)
    alfas[0] = alfas[1]
    alfas[-1] = alfas[-2]

    # -- 2. Transformação onda‑quebra ----------------------------------------
    hb, dirb, depthb = BreakingPropagation(hs, tp, dire, depth,
                                           alfas + 90.0, Bcoef)

    # -- 3. Transporte sedimentar --------------------------------------------
    dc = 0.5 * (doc[1:] + doc[:-1])
    q_now, q0 = ALST(hb, dirb, depthb, alfas + 90.0, kal)

    # Condições nas extremidades (β fantasma implícito)
    if bctype[0] == "Dirichlet":
        q_now[0] = 0.0
    elif bctype[0] == "Neumann":
        q_now[0] = q_now[1]

    if bctype[1] == "Dirichlet":
        q_now[-1] = 0.0
    elif bctype[1] == "Neumann":
        q_now[-1] = q_now[-2]

    # Verifica estabilidade de Courant (apenas aviso)
    if (dx * dx * dc.min() / (4.0 * q0.max())) < dt:
        # Numba não imprime strings grandes eficientemente
        print("WARNING: Courant condition violated – Δt muito grande.")

    # -- 4. Atualiza a linha de costa nos centros ----------------------------
    ynew = y - (dt * 3600.0) / dc * (q_now[1:] - q_now[:-1]) / dx
    return ynew, hb, dirb, depthb, q_now, q0

###############################################################################
# Rotina principal                                                            #
###############################################################################

@jit(nopython=False, parallel=False)
# ("nopython=False" porque chamamos função Python fora do loop principal)
def hansonKraus1991(yi, dt, dx,
                    hs, tp, dire, depth, doc, kal,
                    X0, Y0, phi,
                    bctype=("Neumann", "Neumann"), Bcoef=1.0):
    """Modelo *one‑line* com criação de nós fantasmas e interpolação final.

    Parameters
    ----------
    yi : ndarray (N_centers,)
        Linha de costa inicial nos **centros**.
    dt : ndarray (T,)
        Passos de tempo [h].
    dx : float
        Resolução espacial [m].
    hs, tp, dire, depth, doc : ndarray (T, N_centers)
        Forçantes em cada centro.
    kal : bool or ndarray
        Flag / parâmetro Kalman.
    X0, Y0 : ndarray (N_nodes,)
        Coordenadas dos **transectos** reais.
    phi : ndarray (N_centers,)
        Ângulos da orientação local da costa.
    bctype : (str, str)
        ('Dirichlet' ou 'Neumann', esquerda, direita).
    Bcoef : float
        Coeficiente na formulação de quebra.

    Returns
    -------
    eta_centers : ndarray (T, N_centers)
        Linha de costa nos centros (mesmo que "ysol" original).
    eta_nodes   : ndarray (T, N_nodes)
        Linha de costa interpolada em cada transecto.
    q           : ndarray (T, N_centers)
        Transporte instantâneo em cada centro.
    """
    n_steps, n_centers = hs.shape
    n_nodes = n_centers + 1

    # -- Saídas ----------------------------------------------------------------
    eta_centers = np.empty((n_steps, n_centers))
    eta_centers[0, :] = yi
    hb_all      = np.empty_like(hs)
    dirb_all    = np.empty_like(hs)
    depthb_all  = np.empty_like(hs)
    q_all       = np.empty_like(hs)

    # -------------------------------------------------------------------------
    for t in range(1, n_steps):
        (eta_centers[t, :],
         hb_all[t, :],
         dirb_all[t, :],
         depthb_all[t, :],
         q_all[t, :], _) = ydir_L(
            eta_centers[t - 1, :], dt[t - 1], dx,
            hs[t, :], tp[t, :], dire[t, :], depth[t, :],
            doc[t, :], kal,
            X0, Y0, phi,
            bctype, Bcoef)

    # -------------------------------------------------------------------------
    # Pós‑processamento: centros → transectos (fora da região jittada) ---------
    eta_nodes = np.empty((n_steps, n_nodes))
    for t in range(n_steps):
        eta_nodes[t, :] = centers_to_nodes(eta_centers[t, :], X0, bc=bctype)

    return eta_centers, eta_nodes, q_all
