import numpy as np
from numba import jit
from IHSetUtils.libjit.geometry import nauticalDir2cartesianDir, abs_pos, shore_angle
from IHSetUtils.libjit.waves import BreakingPropagation
from IHSetUtils.libjit.morfology import ALST

@jit
def hansonKraus1991(yi, dt, dx, hs, tp, dir, depth, doc, kal, X0, Y0, phi, bctype, Bcoef):
    """
    Function OneLine calculates the evolution of a system over time using a numerical method.

    Args:
        yi (ndarray): Initial conditions.
        dt (float): Time step.
        dx (float): Spatial step.
        hs (ndarray): Matrix representing sea surface height.
        tp (ndarray): Matrix representing wave periods.
        dir (ndarray): Matrix representing wave directions.
        depth (float): Water depth.
        doc (ndarray): Matrix representing diffusivity coefficients.
        kal (bool): Flag indicating if Kalman filter is used.
        X0 (ndarray): Initial x coordinates.
        Y0 (ndarray): Initial y coordinates.
        phi (ndarray): Phase angles.
        bctype (int): Boundary condition type.

    Returns:
        ysol (ndarray): Solution matrix.
        q (ndarray): Matrix representing some quantity.
    """
    
    nti = hs.shape[0]
    # tii = 1
    desl = 1

    n1 = len(X0)
    mt, n2 = tp.shape

    ysol = np.zeros((mt, n1))
    ysol[0, :] = yi

    hb = np.zeros((mt, n2))
    dirb = np.zeros((mt, n2))
    depthb = np.zeros((mt, n2))
    q = np.zeros((mt, n2))
    q0 = np.zeros((mt, n2))

    # time_init = datetime.now()
    for pos in range(0, nti-1):
        
        ti = pos+desl

        # p1 = ti - desl
        # p2 = ti + desl

        ysol[ti,:], hb[ti,:], dirb[ti,:], depthb[ti,:], q[ti,:], q0[ti,:] =  ydir_L(ysol[ti-1, :], dt, dx, hs[ti, :], tp[ti, :], dir[ti, :], depth, doc[ti, :], kal, X0, Y0, phi, bctype, Bcoef)


        # if pos % 1000 == 0:
            # elp_t = (datetime.now() - time_init).total_seconds()
            # print("Progress of %.2f %% - " % (pos/(nti-1) * 100), end="")
    #         if pos > 0:
    #             print("Average time per step: %.2f [ms] - " % (elp_t/pos), end="")
    #             print("Estimated time to finish: %.2f [s] - " % ((elp_t/pos)*(nti-2-pos)), end="")
    #             print("Elapsed time: %.2f [s]" % (elp_t))

    # print("\n***************************************************************")
    # print("End of simulation")
    # print("***************************************************************")
    # print("\nElapsed simulation time: %.2f seconds \n" % ((datetime.now() - time_init).total_seconds()))
    # print("***************************************************************")

    return ysol, q

@jit
def ydir_L(y, dt, dx, hs, tp, dire, depth, doc, kal, X0, Y0, phi, bctype, Bcoef):
    """
    Function ydir_L calculates the propagation of waves and other quantities at a specific time step.

    Args:
        y (ndarray): Initial conditions.
        dt (float): Time step.
        dx (float): Spatial step.
        ti (int): Time index.
        hs (ndarray): Matrix representing sea surface height.
        tp (ndarray): Matrix representing wave periods.
        dire (ndarray): Matrix representing wave directions.
        depth (float): Water depth.
        hb (ndarray): Matrix representing breaking height.
        dirb (ndarray): Matrix representing breaking direction.
        depthb (ndarray): Matrix representing breaking depth.
        q (ndarray): Matrix representing some quantity.
        doc (ndarray): Matrix representing diffusivity coefficients.
        kal (ndarray): Matrix representing Kalman filter.
        X0 (ndarray): Initial x coordinates.
        Y0 (ndarray): Initial y coordinates.
        phi (ndarray): Phase angles.
        bctype (str): Boundary condition type.

    Returns:
        Tuple containing:
        - ynew (ndarray): Updated conditions.
        - hb (ndarray): Updated breaking height.
        - dirb (ndarray): Updated breaking direction.
        - depthb (ndarray): Updated breaking depth.
        - q (ndarray): Updated quantity matrix.
        - q0 (ndarray): Updated quantity.
    """
    # flag_dig = False

    XN, YN = abs_pos(X0, Y0, nauticalDir2cartesianDir(phi) * np.pi / 180.0, y)

    alfas = np.zeros_like(hs)
    alfas[1:-1] = shore_angle(XN, YN, dire)
    alfas[0] = alfas[1]
    alfas[-1] = alfas[-2]

    # try:
    hb, dirb, depthb = BreakingPropagation(hs, tp, dire, depth, alfas + 90, Bcoef)
    # except:
    #     flag_dig = True
    #     print("Waves diverged -- Q_tot = 0")
    #     hb = np.zeros_like(hs) + 0.01
    #     dirb = np.zeros_like(dire) + alfas + 90
    #     depthb = np.zeros_like(depth) + 0.01
        
    dc = 0.5 * (doc[1:] + doc[:-1])

    # if flag_dig:
    #     q_now = np.zeros_like(hs)
    #     q0 = np.zeros_like(hs)
    # else:
    q_now, q0 = ALST(hb, dirb, depthb, alfas + 90, kal)

    if bctype == "Dirichlet":
        q_now[0] = 0
        q_now[-1] = 0
    elif bctype == "Neumann":
        q_now[0] = q_now[1]
        q_now[-1] = q_now[-2]

    try:
        if (dx**2 * np.min(dc) / (4 * np.max(q0))) < dt:
            print("WARNING: COURANT CONDITION VIOLATED")
    except:
        pass

    ynew = y - (dt * 60 * 60) / dc * (q_now[1:] - q_now[:-1]) / dx

    return ynew, hb, dirb, depthb, q_now, q0