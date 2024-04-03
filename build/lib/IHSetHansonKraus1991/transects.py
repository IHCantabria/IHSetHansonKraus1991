
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
import scipy.optimize as so
import matplotlib.pyplot as plt

class shoreline:
    def __init__(self, shores, dateType, interpN, flagLim, xy):

        #store data
        self.shores = shores
        self.datetype = dateType
        self.interpN = interpN
        self.flagLim = flagLim
        self.xy = xy


        if dateType == 'yyyy':
            self.dates = [int(key) for key in shores.keys()]

        self.shorelines = interpShores(shores, interpN, flagLim, xy)

        self.N = int(interpN)
    
    def setDomain(self, seaPoint, mode, despl, dx):

        #store data
        self.dx = dx
        self.despl = despl
        self.mode = mode
        self.seaPoint = seaPoint

        if mode == 'pol1':
            domain = pol1(self.shorelines, seaPoint, despl, dx)

        self.refline = refLine(domain)
    
    def setTransects(self, length):

        self.length = length

        self.trs = transects(self.refline, length)
    
    def setShorelinePositions(self):

        auxX = np.zeros((self.N, len(self.shorelines)))
        auxY = np.zeros((self.N, len(self.shorelines)))

        for key, i in zip(self.shorelines.keys(), range(self.N)):
            auxX[:,i] = self.shorelines[key]['x']
            auxY[:,i] = self.shorelines[key]['y']


        self.ts = getTimeseries(auxX, auxY,
                                self.trs.xi05, self.trs.yi05,
                                self.trs.xf05, self.trs.yf05)
        
    def setProfiles(self, xx, yy, zz, minDepth):
        
        prof = getProfiles(self.trs.xi, self.trs.yi, self.trs.xf, self.trs.yf, xx, yy , zz)
        # prof05 = getProfiles(self.trs.xi05, self.trs.yi05, self.trs.xf05, self.trs.yf05, xx, yy , zz)

        self.profiles = {}
        for i in range(len(self.trs.xi)):
            jj = prof[i,:,2] < minDepth
            ii = np.isnan(prof[i,:,2])
            self.profiles[str(i+1)] = prof[i,(~ii & jj),:]
        
        self.aDean = deanProfiler(self.profiles)

        prof05 = getProfiles(self.trs.xi05, self.trs.yi05, self.trs.xf05, self.trs.yf05, xx, yy , zz)
        # prof05 = getProfiles(self.trs.xi05, self.trs.yi05, self.trs.xf05, self.trs.yf05, xx, yy , zz)

        self.profiles05 = {}
        for i in range(len(self.trs.xi05)):
            jj = prof05[i,:,2] < minDepth
            ii = np.isnan(prof05[i,:,2])
            self.profiles05[str(i+1)] = prof05[i,(~ii & jj),:]
        
        self.aDean05 = deanProfiler(self.profiles05)

    def interpClimate(self, wavec):

        prof = np.zeros((len(self.profiles), 2))
        for i, key in zip(range(len(self.profiles)),self.profiles.keys()):

            if self.refline.flagSea == 1:
                prof[i, 0] = self.profiles[key][0,0]
                prof[i, 1] = self.profiles[key][0,1]
            elif self.refline.flagSea == -1:
                prof[i, 0] = self.profiles[key][-1,0]
                prof[i, 1] = self.profiles[key][-1,1]

        self.waves = interpWaves(prof[:,0],
                                 prof[:,1],
                                 wavec.x.values,
                                 wavec.y.values,
                                 wavec.hm0.values,
                                 wavec.theta.values,
                                 wavec.tp.values,
                                 wavec.ss.values,
                                 wavec.tide.values,
                                 wavec.depth.values)
        
        self.waves['time'] = wavec.time.values

    # def interpWaves(self, waves):

    #     self.waves = interpWaves(waves, self.refline)

class refLine:
    def __init__(self, ref):
        self.xyi = ref['xyi']
        self.xyf = ref['xyf']
        self.nTrs = int(ref['nTrs'])
        self.b = ref['b']
        self.m = ref['m']
        self.mp = ref['mp']
        self.alpha = ref['alpha']
        self.alphap = ref['alphap']
        self.meanPoint = ref['meanPoint']
        self.posIni = ref['posIni']
        self.posFin = ref['posFin']
        self.dx = ref['dx']
        self.despl = ref['despl']
        self.mode = ref['mode']
        self.flagSea = ref['flagSea']
        self.ii = ref['ii']
        self.ff = ref['ff']

class transects:
    def __init__(self, refline, length):
        
        n = refline.nTrs
        xyi = refline.xyi
        xyf = refline.xyf
        b = refline.b
        m = refline.m
        mp = refline.mp
        alpha = refline.alpha
        alphap = refline.alphap
        dx = refline.dx
        config = {'length': length, 'n': n,
                  'xyi': xyi, 'xyf': xyf,
                  'b': b, 'm': m, 'mp': mp,
                  'alpha': alpha, 'alphap': alphap,
                  'dx': dx, 'mode': refline.mode,
                  'flagSea': refline.flagSea}

        trs = getTrs(config)
        trs05 = getTrs05(config)


        self.xi = trs['xi']
        self.yi = trs['yi']
        self.xf = trs['xf']
        self.yf = trs['yf']
        if refline.flagSea == 1:
            self.phi = np.rad2deg(trs['phi'])
        else:
            self.phi = 360 - (90 - np.rad2deg(trs['phi']))
        self.n = trs['n']

        self.xi05 = trs05['xi']
        self.yi05 = trs05['yi']
        self.xf05 = trs05['xf']
        self.yf05 = trs05['yf']
        if refline.flagSea == 1:
            self.phi05 = np.rad2deg(trs05['phi'])
        else:
            self.phi05 = 360 - (90 - np.rad2deg(trs05['phi']))
        self.n05 = trs05['n']

def pol1(shorelines, seaPoint, despl, dx):
    auxX = np.array([])
    auxY = np.array([])

    for key in shorelines.keys():
        auxX = np.concatenate((auxX, shorelines[key]['x']))
        auxY = np.concatenate((auxY, shorelines[key]['y']))

    meanPoint = np.vstack((np.mean(auxX), np.mean(auxY)))
    auxX = auxX
    auxY = auxY

    xy = np.vstack((auxX, auxY))

    ii = np.argmax(cdist(meanPoint.T, xy.T, 'euclidean'))
    
    posIni = np.vstack((auxX[ii], auxY[ii]))

    jj = np.argmax(cdist(posIni.T, xy.T, 'euclidean'))

    posFin = np.vstack((auxX[jj], auxY[jj]))            

    if posFin[0]<posIni[0]:
        aux = posFin
        posFin = posIni
        posIni = aux

    m = (posIni[1]-posFin[1])/(posIni[0]-posFin[0])

    mp = -1/m

    b = posFin[1] - m * posFin[0]

    alpha = np.arctan(m)

    alphap = np.arctan(mp)

    xyi = np.vstack((posIni[0] + np.cos(alphap) * despl, posIni[1] + np.sin(alphap) * despl))

    if cdist(xyi.T, seaPoint.T, 'euclidean') > cdist(posIni.T, seaPoint.T, 'euclidean'):
        xyf = np.vstack((posFin[0] + np.cos(alphap) * despl, posFin[1] + np.sin(alphap) * despl))
        flagSea = -1
    else:
        xyi = np.vstack((posIni[0] + np.cos(alphap) * -despl, posIni[1] + np.sin(alphap) * -despl))
        xyf = np.vstack((posFin[0] + np.cos(alphap) * -despl, posFin[1] + np.sin(alphap) * -despl))
        flagSea = 1

    b = xyf[1] - m * xyf[0]

    nTrs = np.floor(cdist(xyi.T, xyf.T, 'euclidean')/dx)

    ddxy = .5 * (dx - (cdist(xyi.T, xyf.T, 'euclidean') - nTrs * dx))

    nTrs = nTrs + 2

    if xyi[1] > xyf[1]:
        xyi = np.vstack((xyi[0] - np.cos(alpha) * ddxy, xyi[1] + np.sin(alpha) * ddxy))
        xyf = np.vstack((xyf[0] + np.cos(alpha) * ddxy, xyf[1] - np.sin(alpha) * ddxy))
    else:
        xyi = np.vstack((xyi[0] - np.cos(alpha) * ddxy, xyi[1] - np.sin(alpha) * ddxy))
        xyf = np.vstack((xyf[0] + np.cos(alpha) * ddxy, xyf[1] + np.sin(alpha) * ddxy))

    
    auxi = np.vstack((xyi[0] + np.cos(alphap) * -despl, xyi[1] + np.sin(alphap) * -despl))
    auxf = np.vstack((xyf[0] + np.cos(alphap) * -despl, xyf[1] + np.sin(alphap) * -despl))

    ref = {'xyi': xyi, 'xyf': xyf,
           'nTrs': nTrs, 'b': b,
           'm': m, 'mp': mp,
           'alpha': alpha, 'alphap': alphap,
           'meanPoint': meanPoint, 'posIni': posIni,
           'posFin': posFin, 'dx': dx,
           'despl': despl, 'mode': 'pol1',
           'flagSea': flagSea, 'ii': auxi, 'ff': auxf}

    return ref

def getTrs(config):
    mode = config['mode']
    length = config['length']
    nTrs = config['n']
    xyi = config['xyi']
    xyf = config['xyf']
    b = config['b']
    m = config['m']
    mp = config['mp']
    alpha = config['alpha']
    alphap = config['alphap']
    dx = config['dx']
    flagSea = config['flagSea']

    xi = np.zeros(nTrs)
    yi = np.zeros(nTrs)
    xf = np.zeros(nTrs)
    yf = np.zeros(nTrs)
    phi = np.zeros(nTrs)
    n = np.zeros(nTrs)

    if mode == 'pol1':
        for i in range(nTrs):
            n[i] = i+1
            xi[i] = xyi[0] + i * dx * np.cos(alpha)
            if m > 0:
                yi[i] = xyi[1] + i * dx * np.sin(alpha)
            else:
                yi[i] = xyi[1] - i * dx * np.sin(alpha)

            xf[i] = xi[i] + flagSea * np.cos(alphap) * length
            yf[i] = yi[i] + flagSea * np.sin(alphap) * length

            phi[i] = alpha


    
    trs = {'xi': xi, 'yi': yi, 'xf': xf, 'yf': yf, 'n': n, 'phi': phi}

    return trs

def getTrs05(config):
    mode = config['mode']
    length = config['length']
    nTrs = config['n'] - 1
    xyi = config['xyi']
    xyf = config['xyf']
    b = config['b']
    m = config['m']
    mp = config['mp']
    alpha = config['alpha']
    alphap = config['alphap']
    dx = config['dx']
    flagSea = config['flagSea']

    xi = np.zeros(nTrs)
    yi = np.zeros(nTrs)
    xf = np.zeros(nTrs)
    yf = np.zeros(nTrs)
    phi = np.zeros(nTrs)
    n = np.zeros(nTrs)
    
    if mode == 'pol1':

        xyi[0] = xyi[0] + 1 * dx/2 * np.cos(alpha)
        if m > 0:
            xyi[1] = xyi[1] + 1 * dx/2 * np.sin(alpha)
        else:
            xyi[1] = xyi[1] - 1 * dx/2 * np.sin(alpha)

        for i in range(nTrs):
            n[i] = i+1
            xi[i] = xyi[0] + i * dx * np.cos(alpha)
            if m > 0:
                yi[i] = xyi[1] + i * dx * np.sin(alpha)
            else:
                yi[i] = xyi[1] - i * dx * np.sin(alpha)

            xf[i] = xi[i] + flagSea * np.cos(alphap) * length
            yf[i] = yi[i] + flagSea * np.sin(alphap) * length

            phi[i] = alpha
    
    trs = {'xi': xi, 'yi': yi, 'xf': xf, 'yf': yf, 'n': n, 'phi': phi}

    return trs
    
# @jit(nopython = True)
def getIntersection(slx, sly, xi, yi, xf, yf):

    m = (yi-yf)/(xi - xf)
    b = yi - m * xi
    ii = np.argmin(np.abs(m * slx - sly + b) / (m ** 2 + 1) ** 0.5)

    if np.sqrt((slx[ii] - xi) ** 2 + (sly[ii] - yi) ** 2) > np.sqrt((slx[ii-1] - xi) ** 2 + (sly[ii-1] - yi) ** 2):
        
        ma = (sly[ii]-sly[ii-1])/(slx[ii] - slx[ii-1])
        
    else:
        
        ma = (sly[ii]-sly[ii+1])/(slx[ii] - slx[ii+1])
        
    ba = sly[ii] - ma * slx[ii]
    x = (ba - b) / (m - ma)
    y = m * x + b
    
    return np.sqrt((x - xi) ** 2 + (y - yi) ** 2)

# @jit(nopython = True)
def getTimeseries(slx, sly, xi, yi, xf, yf):

    ts = np.zeros((slx.shape[1], len(xi)))

    for i in range(slx.shape[1]):
        for j in range(1, len(xi)-1):
            ts[i, j] = getIntersection(slx[:,i], sly[:,i], xi[j], yi[j], xf[j], yf[j])
    
    ts[:,0] = ts[:,1]
    ts[:,-1] = ts[:,-2]

    return ts

def interpShores(shores, N, flagLim, xy):

    if flagLim == 'uu':
        for key in shores.keys():
            aux = np.linspace(shores[key]['x'][0], xy, N)
            shores[key]['y'] = np.interp(aux, shores[key]['x'], shores[key]['y'])
            shores[key]['x'] = aux
    elif flagLim == 'uull':
        for key in shores.keys():
            aux = np.linspace(xy[0], xy[1], N)
            shores[key]['y'] = np.interp(aux, shores[key]['x'], shores[key]['y'])
            shores[key]['x'] = aux
    else:
        for key in shores.keys():
            aux = np.linspace(shores[key]['x'][0], shores[key]['x'][-1], N)
            shores[key]['y'] = np.interp(aux, shores[key]['x'], shores[key]['y'])
            shores[key]['x'] = aux

    return shores

@jit(nopython = True)
def getProfiles(xi, yi, xf, yf, x, y , z):

    prof = np.zeros((len(xi),1000, 4))

    for i in range(len(xi)):
        m = (yf[i] - yi[i]) / (xf[i] - xi[i])
        b = yi[i] - m * xi[i]
        xx = np.linspace(xi[i], xf[i], 1000)
        yy = m * xx + b
        for j in range(len(xx)):
            ii = np.argmin(np.abs(xx[j] - x[0,:]))
            jj = np.argmin(np.abs(yy[j] - y[:,0]))
            prof[i, j, 0] = x[jj,ii]
            prof[i, j, 1] = y[jj,ii]
            prof[i, j, 2] = -z[jj,ii]
            prof[i, j, 3] = np.sqrt((x[jj,ii] - xi[i]) ** 2 + (y[jj,ii] - yi[i]) ** 2)


    return prof

def deanProfiler(prof):

    aDean = np.zeros(len(prof.keys()))
    deanProf = lambda x, A: A * x ** (2/3)
    
    for key, i in zip(prof.keys(), range(len(prof.keys()))):
        popt, _ = so.curve_fit(deanProf, prof[key][:,3]-prof[key][0,3], prof[key][:,2])
        aDean[i] = popt[0]

    return aDean

def deanProfile(A, x):
    return A * x ** (2/3)

def plotDeanProfiles(domain, minDepth, maxLen, saveDir):

    font = {'family': 'serif',
        'weight': 'bold',
        'size': 8}
    for i in range(len(domain.aDean)):
        dist = np.linspace(0, domain.profiles[str(i+1)][:,3].max()-domain.profiles[str(i+1)][0,3], 1000)
        h = deanProfile(domain.aDean[i], dist)
        
        plt.figure(figsize=(8, 5), dpi=200, linewidth=5, edgecolor="#04253a")
        plt.plot(domain.profiles[str(i+1)][:,3]-domain.profiles[str(i+1)][0,3],
                domain.profiles[str(i+1)][:,2], 'k-', label = 'Data')
        plt.plot(dist, h, 'r--', label = 'Fitted Dean Profile')
        plt.grid(visible=True, which='major', axis='both', c = "black", linestyle = "--", linewidth = 0.3, zorder = 51)
        plt.ylim(0, minDepth)
        plt.xlim(0, maxLen)
        plt.gca().invert_yaxis()
        plt.legend(loc = 'upper right')
        plt.ylabel('Depth [m]', fontdict=font)
        plt.xlabel('Distance [m]', fontdict=font)
        plt.savefig(saveDir + '\\Prof_' + str(i+1) + '.png')
        plt.close()

def interpWaves(x, y, xw, yw, hm0, theta, tp, ss, tide, depth):

    # mkii = np.vectorize(lambda xww, yww: np.argmin(np.sqrt((x-xww) ** 2 + (y-yww) ** 2 )))
    
    # ii = mkii(xw, yw)

    d = np.sqrt((x-x[0]) ** 2 + (y-y[0]) ** 2)

    dd = np.sqrt((x[0]-xw) ** 2 + (y[0]-yw) ** 2)

    # wavec = {
    #     'depth': np.interp(d, d[ii], depth),
    #     'hm0': np.zeros((hm0.shape[0], len(x))),
    #     'theta': np.zeros((hm0.shape[0], len(x))),
    #     'tp': np.zeros((hm0.shape[0], len(x))),
    #     'ss': np.zeros((hm0.shape[0], len(x))),
    #     'tide': np.zeros((hm0.shape[0], len(x))),
    # }

    # for i in range(hm0.shape[0]):
    #     wavec['hm0'][i,:] =  np.interp(d, d[ii], hm0[i,:])
    #     wavec['theta'][i,:] = np.interp(d, d[ii], theta[i,:])
    #     wavec['tp'][i,:] = np.interp(d, d[ii], tp[i,:])
    #     wavec['ss'][i,:] = np.interp(d, d[ii], ss[i,:])
    #     wavec['tide'][i,:] = np.interp(d, d[ii], tide[i,:])

    wavec = {
        'depth': np.interp(d, dd, depth),
        # 'depth': np.zeros((len(x), hm0.shape[1])),
        'hm0': np.zeros((hm0.shape[1], len(x))),
        'doc': np.zeros((hm0.shape[1], len(x))),
        'theta': np.zeros((hm0.shape[1], len(x))),
        'tp': np.zeros((hm0.shape[1], len(x))),
        'ss': np.zeros((hm0.shape[1], len(x))),
        'tide': np.zeros((hm0.shape[1], len(x))),
    }

    for i in range(hm0.shape[1]):
        wavec['hm0'][i,:] =  np.interp(d, dd, hm0[:,i])
        wavec['theta'][i,:] = np.interp(d, dd, theta[:,i])
        wavec['tp'][i,:] = np.interp(d, dd, tp[:,i])
        wavec['ss'][i,:] = np.interp(d, dd, ss[:,i])
        wavec['tide'][i,:] = np.interp(d, dd, tide[:,i])
        # wavec['depth'][:,i] = np.interp(d, dd, depth[:,i])
    

    return wavec

