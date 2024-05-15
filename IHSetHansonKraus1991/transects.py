
import numpy as np
from scipy.spatial.distance import cdist
import scipy.interpolate as spi
from numba import jit
import scipy.optimize as so
import matplotlib.pyplot as plt
from IHSetUtils import depthOfClosure, Hs12Calc, ADEAN, wast
from IHSetUtils import nauticalDir2cartesianDirP as n2c

class shoreline:
    def __init__(self, shores, timeShores, **kwargs):

        #store data
        self.shores = shores
        self.timeShores = timeShores
        
        self.epsg = kwargs['epsg']
        self.corners = kwargs['corners']
        self.rounder = kwargs['rounder']
        self.server = kwargs['server']
        try:
            interpN = kwargs['interpN']
        except:
            interpN = 5000       

        self.shorelines, self.flag_f = interpShores(shores, interpN)
            
        # self.N = int(interpN)
        self.N = interpN
    
    def setDomain(self, seaPoint, mode, dx, **kwargs):

        #store data
        self.dx = dx
        self.mode = mode
        self.seaPoint = seaPoint

        if mode == 'pol1':
            self.despl = kwargs['despl']
            domain = pol1(self.shorelines, seaPoint, self.despl, dx)
        elif mode == 'draw':
            domain = draw(seaPoint, dx, kwargs['refPoints'], self.flag_f)

        self.refline = refLine(domain)
    
    def setTransects(self, length):

        self.length = length

        self.trs = transects(self.refline, length)
    
    def setShorelinePositions(self):

        auxX = np.zeros((self.N, len(self.shorelines)))
        auxY = np.zeros((self.N, len(self.shorelines)))

        for key in self.shorelines.keys():
            i = int(key) - 1
            auxX[:,i] = self.shorelines[key]['x']
            auxY[:,i] = self.shorelines[key]['y']


        self.ts = getTimeseries(auxX, auxY,
                                self.trs.xi, self.trs.yi,
                                self.trs.xf, self.trs.yf)
        self.ts05 = getTimeseries(auxX, auxY,
                                self.trs.xi05, self.trs.yi05,
                                self.trs.xf05, self.trs.yf05)
        
    def setProfiles(self, mode, minDepth, **kwargs):

        self.profMode = mode
        self.minDepth = minDepth  
        
        if mode == 'bathy':
            prof = getProfiles(self.trs.xi, self.trs.yi, self.trs.xf, self.trs.yf, kwargs['xx'], kwargs['yy'] , kwargs['zz'])
            # prof05 = getProfiles(self.trs.xi05, self.trs.yi05, self.trs.xf05, self.trs.yf05, xx, yy , zz)

            self.profiles = {}
            for i in range(len(self.trs.xi)):
                jj = prof[i,:,2] < minDepth
                ii = np.isnan(prof[i,:,2])
                self.profiles[str(i+1)] = prof[i,(~ii & jj),:]        
            self.aDean = deanProfiler(self.profiles)

            prof05 = getProfiles(self.trs.xi05, self.trs.yi05, self.trs.xf05, self.trs.yf05, kwargs['xx'], kwargs['yy'] , kwargs['zz'])
            # prof05 = getProfiles(self.trs.xi05, self.trs.yi05, self.trs.xf05, self.trs.yf05, xx, yy , zz)

            self.profiles05 = {}
            for i in range(len(self.trs.xi05)):
                jj = prof05[i,:,2] < minDepth
                ii = np.isnan(prof05[i,:,2])
                self.profiles05[str(i+1)] = prof05[i,(~ii & jj),:]
            
            self.aDean05 = deanProfiler(self.profiles05)
        elif mode == 'D50':

            self.aDean = np.zeros_like(self.trs.xi) + ADEAN(kwargs['D50'])
            self.aDean05 = np.zeros_like(self.trs.xi05) + ADEAN(kwargs['D50'])

            self.profiles = {}
            for i in range(len(self.trs.xi)):
                dist_i = self.ts[0,i]
                dist_f = wast(minDepth, kwargs['D50'])
                dist = np.linspace(dist_i, dist_i+dist_f, 100)
                xi , xf = self.trs.xi[i], self.trs.xi[i] + np.cos(np.deg2rad(n2c(self.trs.phi[i]))) * dist_f
                yi, yf = self.trs.yi[i], self.trs.yi[i] + np.sin(np.deg2rad(n2c(self.trs.phi[i]))) * dist_f
                x = np.linspace(xi, xf, 100)
                y = np.linspace(yi, yf, 100)
                z = deanProfile(self.aDean[i], np.linspace(0.0001, dist_f, 100))
                self.profiles[str(i+1)] = np.column_stack((x, y, z, dist))
                
            self.profiles05 = {}
            for i in range(len(self.trs.xi05)):
                dist_i = self.ts05[0,i]
                dist_f = wast(minDepth, kwargs['D50'])
                dist = np.linspace(dist_i, dist_i+dist_f, 100)
                xi , xf = self.trs.xi05[i], self.trs.xi05[i] + np.cos(np.deg2rad(n2c(self.trs.phi05[i]))) * dist_f
                yi, yf = self.trs.yi05[i], self.trs.yi05[i] + np.sin(np.deg2rad(n2c(self.trs.phi05[i]))) * dist_f
                x = np.linspace(xi, xf, 100)
                y = np.linspace(yi, yf, 100)
                z = deanProfile(self.aDean05[i], np.linspace(0.0001, dist_f, 100))
                self.profiles05[str(i+1)] = np.column_stack((x, y, z, dist))





    def interpClimate(self, wavec):

        prof = np.zeros((len(self.profiles), 2))
        for i, key in zip(range(len(self.profiles)),self.profiles.keys()):

                ii = np.argmax(np.abs(self.profiles[key][:,2]))
                prof[i, 0] = self.profiles[key][ii,0]
                prof[i, 1] = self.profiles[key][ii,1]



        self.waves = interpWaves(prof[:,0],
                                 prof[:,1],
                                 wavec['x'],
                                 wavec['y'],
                                 wavec['Hs'],
                                 wavec['Dir'],
                                 wavec['Tp'],
                                 wavec['surge'],
                                 wavec['tide'],
                                 wavec['depth'])
        
        self.waves['time'] = wavec['time']
        self.waves['x'] = wavec['x']
        self.waves['y'] = wavec['y']

    # def interpWaves(self, waves):

    #     self.waves = interpWaves(waves, self.refline)

class refLine:
    def __init__(self, ref):
        self.xyi = ref['xyi']
        self.nTrs = int(ref['nTrs'])
        self.b = ref['b']
        self.m = ref['m']
        self.mp = ref['mp']
        self.alpha = ref['alpha']
        self.alphap = ref['alphap']
        self.meanPoint = ref['meanPoint']
        self.dx = ref['dx']
        self.mode = ref['mode']
        self.flagSea = ref['flagSea']
        self.flag_f = ref['flag_f']
        if self.mode == 'pol1':
            self.despl = ref['despl']

class transects:
    def __init__(self, refline, length):
        
        n = refline.nTrs
        xyi = refline.xyi
        b = refline.b
        m = refline.m
        mp = refline.mp
        alpha = refline.alpha
        alphap = refline.alphap
        dx = refline.dx
        xyf = find_transects_f(xyi, alphap, length, refline.flagSea, refline.flag_f)
        
        config = {'length': length, 'n': n,
                  'xyi': xyi, 'xyf': xyf,
                  'b': b, 'm': m, 'mp': mp,
                  'alpha': alpha, 'alphap': alphap,
                  'dx': dx, 'mode': refline.mode,
                  'flagSea': refline.flagSea, 'flag_f': refline.flag_f}

        trs = getTrs(config)

        if refline.flag_f == 1:
            if refline.flagSea == 1:
                self.phi = np.rad2deg(trs['phi'])[0::2]
                self.phi05 = np.rad2deg(trs['phi'])[1::2]
            else:
                self.phi = 360 - (90 - np.rad2deg(trs['phi']))[0::2]
                self.phi05 = 360 - (90 - np.rad2deg(trs['phi']))[1::2]
        elif refline.flag_f == 0:
            if refline.flagSea == 1:
                self.phi = np.rad2deg(trs['phi'])[0::2]
                self.phi05 = np.rad2deg(trs['phi'])[1::2]
            else:
                self.phi = 360 - np.rad2deg(trs['phi'])[0::2]
                self.phi05 = 360 - np.rad2deg(trs['phi'])[1::2]            
        
        self.xi = trs['xi'][0::2]
        self.yi = trs['yi'][0::2]
        self.xf = trs['xf'][0::2]
        self.yf = trs['yf'][0::2]
        

        self.xi05 = trs['xi'][1::2]
        self.yi05 = trs['yi'][1::2]
        self.xf05 = trs['xf'][1::2]
        self.yf05 = trs['yf'][1::2]

        self.n = trs['n']

def pol1(shorelines, seaPoint, despl, dx):
    
    auxX = np.array([])
    auxY = np.array([])

    for key in shorelines.keys():
        auxX = np.concatenate((auxX, shorelines[key]['x']))
        auxY = np.concatenate((auxY, shorelines[key]['y']))

    meanPoint = np.vstack((np.mean(auxX), np.mean(auxY)))
  
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
           'meanPoint': meanPoint, 'dx': dx,
           'despl': despl, 'mode': 'pol1',
           'flagSea': flagSea}

    return ref

def draw(seaPoint, dx, refPoints, flag_f):

    if flag_f == 1:
        ii = np.argsort(refPoints[:,0])
        refPoints = refPoints[ii,:]
    elif flag_f == 0:
        ii = np.argsort(refPoints[:,1])
        refPoints = refPoints[ii,:]   
    
    meanPoint = np.vstack((np.mean(refPoints[:,0]), np.mean(refPoints[:,1])))
        
    nm = int(len(refPoints)-1)
    m = np.zeros(nm)
    mp = np.zeros(nm)
    b = np.zeros(nm)
    alpha = np.zeros(nm)
    alphap = np.zeros(nm)
    xyi = np.zeros((nm, 2))
    length = 0

    if flag_f == 1:
        for i in range(1, len(refPoints[:,0])):
            m[i-1] = (refPoints[i,1]-refPoints[i-1,1])/(refPoints[i,0]-refPoints[i-1,0])
            mp[i-1] = -1/m[i-1]
            b[i-1] = refPoints[i,1] - m[i-1] * refPoints[i,0]
            alpha[i-1] = np.arctan(m[i-1])
            alphap[i-1] = np.arctan(mp[i-1])
            xyi[i-1,0] = refPoints[i,0] + np.cos(alphap[i-1]) * 10
            xyi[i-1,1] = refPoints[i,1] + np.sin(alphap[i-1]) * 10
            if cdist(xyi[i-1,:].reshape(1,2), seaPoint.T, 'euclidean') > cdist(refPoints[i-1,:].reshape(1,2), seaPoint.T, 'euclidean'):
                flagSea = -1
            else:
                flagSea = 1
            length += cdist(refPoints[i,:].T, refPoints[i-1,:].T, 'euclidean')
    elif flag_f == 0:
        for i in range(1, len(refPoints[:,1])):
            m[i-1] = (refPoints[i,0]-refPoints[i-1,0])/(refPoints[i,1]-refPoints[i-1,1])
            mp[i-1] = -1/m[i-1]
            b[i-1] = refPoints[i,0] - m[i-1] * refPoints[i,1]
            alpha[i-1] = np.arctan(m[i-1])
            alphap[i-1] = np.arctan(mp[i-1])
            xyi[i-1,0] = refPoints[i,0] + np.cos(alphap[i-1]) * 10
            xyi[i-1,1] = refPoints[i,1] + np.sin(alphap[i-1]) * 10
            if cdist(xyi[i-1,:].reshape(1,2), seaPoint.T, 'euclidean') > cdist(refPoints[i-1,:].reshape(1,2), seaPoint.T, 'euclidean'):
                flagSea = -1
            else:
                flagSea = 1
            length += cdist(refPoints[i,:].reshape(1,2), refPoints[i-1,:].reshape(1,2), 'euclidean')

    nTrs = np.floor(length/dx)

    ddxy = .5 * (dx - (length - nTrs * dx))

    nTrs = int(nTrs + 2)

    xyi, m, b = find_transects_i(refPoints, m, b, int(nTrs  + nTrs -1), ddxy)

    alpha = np.arctan(m)
    mp = -1/m
    alphap = np.arctan(mp)

    ref = {'xyi': xyi,
           'nTrs': nTrs, 'b': b,
           'm': m, 'mp': mp,
           'alpha': alpha, 'alphap': alphap,
           'meanPoint': meanPoint, 'dx': dx,
           'mode': 'draw',
           'flagSea': flagSea, 'flag_f': flag_f}

    return ref

def getTrs(config):
    mode = config['mode']
    length = config['length']
    xyi = config['xyi']
    xyf = config['xyf']
    b = config['b']
    m = config['m']
    mp = config['mp']
    alpha = config['alpha']
    alphap = config['alphap']
    dx = config['dx']
    flagSea = config['flagSea']
    flag_f = config['flag_f']

    nTrs = len(config['xyi'])
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
    elif mode == 'draw':
        for i in range(nTrs):
            n[i] = i+1
            xi[i] = xyi[i,0]
            yi[i] = xyi[i,1]
            xf[i] = xyf[i,0]
            yf[i] = xyf[i,1]
            phi[i] = alpha[i]

    
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

def interpShores(shores, N):

    x_up = -np.inf
    x_down = np.inf
    y_up = -np.inf
    y_down = np.inf

    for key in shores.keys():
        x_up = max(x_up, shores[key]['x'].max())
        x_down = min(x_down, shores[key]['x'].min())
        y_up = max(y_up, shores[key]['y'].max())
        y_down = min(y_down, shores[key]['y'].min())

    x_ang = np.zeros(len(shores))
    y_ang = np.zeros(len(shores))

    for i, key in enumerate(shores.keys()):
        pol = np.polyfit(shores[key]['x'], shores[key]['y'], 1)
        x_ang[i] = np.arctan(pol[0])
        y_ang[i] = np.arctan(1/pol[0])

    mean_ang_x = np.mean(x_ang)
    mean_ang_y = np.mean(y_ang)
    

    if mean_ang_x <= mean_ang_y:
        for key in shores.keys():
            xi = shores[key]['x'][0]
            xf = shores[key]['x'][-1]
            if xf > xi:
                aux = np.linspace(xi, xf, N)
                # ii = np.argsort(shores[key]['x'])
                # sply = spi.make_interp_spline(shores[key]['x'][ii], shores[key]['y'][ii], bc_type='natural')
                shores[key]['y'] = np.interp(aux, shores[key]['x'], shores[key]['y'])
                # shores[key]['y'] = sply(aux)
                shores[key]['x'] = aux
            else:
                aux = np.linspace(xf, xi, N)
                # ii = np.argsort(shores[key]['x'])
                # sply = spi.make_interp_spline(shores[key]['x'][ii], shores[key]['y'][ii], bc_type='natural')
                shores[key]['y'] = np.interp(aux, shores[key]['x'], shores[key]['y'])
                # shores[key]['y'] = sply(aux)
                shores[key]['x'] = aux
        flag_f = 1
    else:
        for key in shores.keys():
            yi = shores[key]['y'][0]
            yf = shores[key]['y'][-1]
            if yf > yi:
                aux = np.linspace(yi, yf, N)
                # ii = np.argsort(shores[key]['y'])
                # splx = spi.make_interp_spline(shores[key]['y'][ii], shores[key]['x'][ii], bc_type='natural')
                shores[key]['x'] = np.interp(aux, shores[key]['y'], shores[key]['x'])
                # shores[key]['x'] = splx(aux)
                shores[key]['y'] = aux
            else:
                aux = np.linspace(yf, yi, N)
                # ii = np.argsort(shores[key]['y'])
                # splx = spi.make_interp_spline(shores[key]['y'][ii], shores[key]['x'][ii], bc_type='natural')
                shores[key]['x'] = np.interp(aux, shores[key]['y'], shores[key]['x'])
                # shores[key]['x'] = splx(aux)
                shores[key]['y'] = aux
        flag_f = 0

    return shores, flag_f

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

def interpWaves(x, y, xw, yw, Hs, Dir, Tp, surge, tide, depth):

    # mkii = np.vectorize(lambda xww, yww: np.argmin(np.sqrt((x-xww) ** 2 + (y-yww) ** 2 )))
    
    # ii = mkii(xw, yw)

    d = np.sqrt((x-x[0]) ** 2 + (y-y[0]) ** 2)

    dd = np.sqrt((x[0]-xw) ** 2 + (y[0]-yw) ** 2)

    # wavec = {
    #     'depth': np.interp(d, d[ii], depth),
    #     'Hs': np.zeros((Hs.shape[0], len(x))),
    #     'Dir': np.zeros((Hs.shape[0], len(x))),
    #     'Tp': np.zeros((Hs.shape[0], len(x))),
    #     'surge': np.zeros((Hs.shape[0], len(x))),
    #     'tide': np.zeros((Hs.shape[0], len(x))),
    # }

    # for i in range(Hs.shape[0]):
    #     wavec['Hs'][i,:] =  np.interp(d, d[ii], Hs[i,:])
    #     wavec['Dir'][i,:] = np.interp(d, d[ii], Dir[i,:])
    #     wavec['Tp'][i,:] = np.interp(d, d[ii], Tp[i,:])
    #     wavec['surge'][i,:] = np.interp(d, d[ii], surge[i,:])
    #     wavec['tide'][i,:] = np.interp(d, d[ii], tide[i,:])

    wavec = {
        'depth': np.interp(d, dd, depth),
        # 'depth': np.zeros((len(x), Hs.shape[1])),
        'Hs': np.zeros((Hs.shape[1], len(x))),
        'doc': np.zeros((Hs.shape[1], len(x))),
        'Dir': np.zeros((Hs.shape[1], len(x))),
        'Tp': np.zeros((Hs.shape[1], len(x))),
        'surge': np.zeros((Hs.shape[1], len(x))),
        'tide': np.zeros((Hs.shape[1], len(x)))
    }

    for i in range(Hs.shape[1]):
        wavec['Hs'][i,:] =  np.interp(d, dd, Hs[:,i])
        wavec['Dir'][i,:] = np.interp(d, dd, Dir[:,i])
        wavec['Tp'][i,:] = np.interp(d, dd, Tp[:,i])
        wavec['surge'][i,:] = np.interp(d, dd, surge[:,i])
        wavec['tide'][i,:] = np.interp(d, dd, tide[:,i])
        Hs12, Ts12 = Hs12Calc(Hs[:,i], Tp[:,i])
        wavec['doc'][i,:] = depthOfClosure(Hs12, Ts12)
        # wavec['depth'][:,i] = np.interp(d, dd, depth[:,i])
    

    return wavec

def find_transects_i(pontos, m, b, num_transectos, ddxy):

    distancias = np.cumsum(np.sqrt(np.sum(np.diff(pontos, axis=0)**2, axis=1)))
    distancias = np.hstack((0, distancias))
    m = np.hstack((m[0], m))
    b = np.hstack((b[0], b))

    t_interp = np.linspace(-ddxy, distancias[-1]+ddxy, num_transectos).squeeze()

    interx = spi.interp1d(distancias, pontos[:,0], kind='linear', fill_value = 'extrapolate')
    intery = spi.interp1d(distancias, pontos[:,1], kind='linear', fill_value = 'extrapolate')
    interm = spi.interp1d(distancias, m, kind='linear', fill_value = 'extrapolate')
    interb = spi.interp1d(distancias, b, kind='linear', fill_value = 'extrapolate')

    interp_pontos = np.array([interx(t_interp), intery(t_interp)]).T
    inter_m = interm(t_interp)
    inter_b = interb(t_interp)

    # interp_pontos = np.array([np.interp(t_interp, distancias, pontos[:,0]).squeeze(),
    #                            np.interp(t_interp, distancias, pontos[:,1]).squeeze()]).T
    # inter_m = np.interp(t_interp, distancias, m).squeeze()
    # inter_b = np.interp(t_interp, distancias, b).squeeze()

    return interp_pontos, inter_m, inter_b

def find_transects_f(pontos, alphap, length, flagSea, flag_f):

    xyf = np.zeros_like(pontos)

    if flag_f == 1:
        for i in range(0, len(pontos[:,0])):
            xyf[i,0] = pontos[i,0] + flagSea * np.cos(alphap[i]) * length
            xyf[i,1] = pontos[i,1] + flagSea * np.sin(alphap[i]) * length
    elif flag_f == 0:
        for i in range(0, len(pontos[:,0])):
            xyf[i,0] = pontos[i,0] + flagSea * -1 * np.sin(alphap[i]) * length
            xyf[i,1] = pontos[i,1] + flagSea * -1 * np.cos(alphap[i]) * length

    return xyf