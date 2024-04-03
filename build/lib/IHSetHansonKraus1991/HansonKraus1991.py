import numpy as np
from numba import jit
from datetime import datetime
from IHSetUtils import nauticalDir2cartesianDir, abs_pos, shore_angle, BreakingPropagation_1L, ALST

@jit
def hansonKraus1991(yi, dt, dx, hs, tp, dir, depth, doc, kal, X0, Y0, phi, bctype):
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
    
    nti = hs.shape[1]
    tii = 2
    desl = 1

    n1 = len(X0)
    n2, mt = tp.shape

    ysol = np.zeros((n1, mt))
    ysol[:, 0] = yi

    hb = np.zeros((n2, mt))
    dirb = np.zeros((n2, mt))
    depthb = np.zeros((n2, mt))
    q = np.zeros((n2, mt))
    q0 = np.zeros((n2, mt))

    time_init = datetime.now()
    for pos in range(1, nti-1):

        ti = pos+desl

        p1 = ti - desl
        p2 = ti + desl

        ynew, hb[:, ti], dirb[:, ti], depthb[:, ti], q[:, ti], q0[:, ti] = \
            ydir_L(ysol[:, ti-1], dt, dx, tii, hs[:, p1:p2], tp[:, p1:p2], dir[:, p1:p2], depth, \
                   hb[:, p1:p2], dirb[:, p1:p2], depthb[:, p1:p2], q[:, p1:p2], doc[:, p1:p2], kal, X0, Y0, phi, bctype)

        ysol[:, ti] = ynew

        p1 += 1
        p2 += 1

        if pos % 1000 == 0:
            elp_t = (datetime.now() - time_init).total_seconds() * 1000
            print("\n Progress of %.2f %% - " % (pos/(nti-1) * 100), end="")
            print("Average time per step: %.2f [ms] - " % (elp_t/pos), end="")
            print("Estimated time to finish: %.2f [s] - " % ((elp_t/1000/pos)*(nti-2-pos)), end="")
            print("Elapsed time: %.2f [s]" % (elp_t/1000))

    print("\n***************************************************************")
    print("End of simulation")
    print("***************************************************************")
    print("\nElapsed simulation time: %.2f seconds \n" % ((datetime.now() - time_init).total_seconds() * 1000))
    print("***************************************************************")

    return ysol, q

@jit
def ydir_L(y, dt, dx, ti, hs, tp, dire, depth, hb, dirb, depthb, q, doc, kal, X0, Y0, phi, bctype):
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
    XN, YN = abs_pos(X0, Y0, nauticalDir2cartesianDir(phi) * np.pi / 180.0, y)

    alfas = np.zeros(hs.shape[0])
    alfas[1:-1] = shore_angle(XN, YN, dire[:, ti])
    alfas[0] = alfas[1]
    alfas[-1] = alfas[-2]

    try:
        hb[:, ti], dirb[:, ti], depthb[:, ti] = BreakingPropagation_1L(hs[:, ti], tp[:, ti], dire[:, ti], depth, alfas + 90, "spectral")
    except Exception as e:
        print("Waves diverged -- Q_tot = 0")
        print(e)

    dc = 0.5 * (doc[1:, ti-1:ti+2] + doc[:-1, ti-1:ti+2])

    q[:, ti], q0 = ALST(hb[:, ti], tp[:, ti], dirb[:, ti], depthb[:, ti], alfas + 90, kal)

    if bctype == "Dirichlet":
        q[0, ti] = 0
        q[-1, ti] = 0
    elif bctype == "Neumann":
        q[0, ti] = q[1, ti]
        q[-1, ti] = q[-2, ti]

    if (dx**2 * np.min(dc) / (4 * np.max(q0))) < dt:
        print("WARNING: COURANT CONDITION VIOLATED")

    ynew = y - (dt * 60 * 60) / dc[:, 1] * (q[1:, ti] - q[:-1, ti]) / dx

    return ynew, hb[:, ti], dirb[:, ti], depthb[:, ti], q[:, ti], q0

