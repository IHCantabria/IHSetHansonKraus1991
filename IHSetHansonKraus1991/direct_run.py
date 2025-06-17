import numpy as np
import xarray as xr
import pandas as pd
import fast_optimization as fo
from .HansonKraus1991 import hansonKraus1991
from IHSetUtils import Hs12Calc, depthOfClosure
import json

class HansonKraus1991_run(object):
    """
    Yates09_run
    
    Configuration to calibrate and run the Yates et al. (2009) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
        self.name = 'Hanson and Kraus (1991)'
        self.mode = 'standalone'
        self.type = 'OL'
     
        data = xr.open_dataset(path)
        
        cfg = json.loads(data.attrs['run_HansonKraus'])
        self.cfg = cfg

        self.depth = cfg['depth']
        self.switch_Kal = cfg['switch_Kal']
        self.breakType = cfg['break_type']
        self.bctype = cfg['bctype']
        self.doc_formula = cfg['doc_formula']
        self.fomulation = cfg['formulation']

        self.start_date = pd.to_datetime(cfg['start_date'])
        self.end_date = pd.to_datetime(cfg['end_date'])
        
        if self.breakType == 'Spectral':
            self.Bcoef = 0.45
        elif self.breakType == 'Monochromatic':
            self.Bcoef = 0.78

        if self.fomulation == 'CERC (1984)':
            self.fomulation = 1
        elif self.fomulation == 'Komar (1998)':
            self.fomulation = 2
        elif self.fomulation == 'Kamphhuis (2002)':
            self.fomulation = 3
            self.mb = cfg['mb']
            self.D50 = cfg['D50']
        elif self.fomulation == 'Van Rijn (2014)':
            self.fomulation = 4

        self.Y0 = data.yi.values
        self.X0 = data.xi.values
        self.Xf = data.xf.values
        self.Yf = data.yf.values
        self.phi = data.phi.values
        
        self.hs = data.hs.values
        self.tp= data.tp.values
        self.dir = data.dir.values
        self.time = pd.to_datetime(data.time.values)

        self.Obs = data.obs.values
        self.time_obs = pd.to_datetime(data.time_obs.values)

        self.ntrs = len(self.X0)
        self.dx = ((self.Y0[1:]- self.Y0[:-1])**2 + (self.X0[1:]- self.X0[:-1])**2)**0.5
        self.dx = np.hstack((self.dx[0], ((self.Yf[1:]- self.Yf[:-1])**2 + (self.Xf[1:]- self.Xf[:-1])**2)**0.5))
        
        data.close()

        self.interp_forcing()
        self.split_data()
        
        self.yi = self.Obs[0,:]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        self.idx_obs = mkIdx(self.time_obs)

        # Now we calculate the dt from the time variable
        mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
        self.dt = mkDT(np.arange(0, len(self.time)-1))

        
        self.doc = np.zeros_like(self.hs_)
        self.depth = np.zeros_like(self.hs_) + self.depth
        for k in range(self.ntrs):
            hs12, ts12 = Hs12Calc(self.hs_, self.tp_)
            self.doc[:,k] = depthOfClosure(hs12, ts12, self.doc_formula)
        

        def run_model(par):
            K = par
            Ymd, _ = hansonKraus1991(self.yi,
                                        self.dt,
                                        self.dx,
                                        self.hs_,
                                        self.tp_,
                                        self.dir_,
                                        self.depth,
                                        self.doc,
                                        K,
                                        self.X0,
                                        self.Y0,
                                        self.phi,
                                        self.bctype,
                                        self.Bcoef,
                                         self.fomulation,
                                         self.mb,
                                         self.D50)
            return Ymd

        self.run_model = run_model
    
    def run(self, par):
        self.full_run = self.run_model(par)
        if self.switch_Kal == 1:
            self.par_names = []
            for i in range(len(par)):
                self.par_names.append(rf'K_{i}')
            self.par_values = par
        elif self.switch_Kal == 0:
            self.par_names = [r'K']
            self.par_values = par

        self.calculate_metrics()

    def calculate_metrics(self):
        self.metrics_names = fo.backtot()[0]
        self.indexes = fo.multi_obj_indexes(self.metrics_names)
        self.metrics = fo.multi_obj_func(self.Obs.flatten(), self.full_run[self.idx_obs].flatten(), self.indexes)

    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """
        ii = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0][0]
        self.time = self.time[ii:]
        self.hs_ = self.hs_[ii:, :]
        self.tp_ = self.tp_[ii:, :]
        self.dir_ = self.dir_[ii:, :]

        ii = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]
        self.Obs = self.Obs[ii,:]
        self.time_obs = self.time_obs[ii]


    def interp_forcing(self):
        """
        Interpolate the forcing data to the half way of the transects.
        hs(time, trs) -> hs(time, trs+0.5)
        tp(time, trs) -> tp(time, trs+0.5)
        dir(time, trs) -> dir(time, trs+0.5)
        depth(time, trs) -> depth(time, trs+0.5)
        doc(time, trs) -> doc(time, trs+0.5)
        """

        dist = np.cumsum(self.dx)
        dist_ = dist[1:] - self.dx[1:]/2

        
        self.hs_ = np.zeros((len(self.time), self.ntrs+1))
        self.tp_ = np.zeros((len(self.time), self.ntrs+1))
        self.dir_ = np.zeros((len(self.time), self.ntrs+1))

        self.hs_[:, 0], self.hs_[:, -1] = self.hs[:, 0], self.hs[:, -1]
        self.tp_[:, 0], self.tp_[:, -1] = self.tp[:, 0], self.tp[:, -1]
        self.dir_[:, 0], self.dir_[:, -1] = self.dir[:, 0], self.dir[:, -1]

        for i in range(len(self.time)):
            self.hs_[i, 1:-1] = np.interp(dist_, dist, self.hs[i, :])
            self.tp_[i, 1:-1] = np.interp(dist_, dist, self.tp[i, :])
            self.dir_[i, 1:-1] = np.interp(dist_, dist, self.dir[i, :])



