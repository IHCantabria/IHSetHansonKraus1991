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

        self.start_date = pd.to_datetime(cfg['start_date'])
        self.end_date = pd.to_datetime(cfg['end_date'])
        
        if self.breakType == 'Spectral':
            self.Bcoef = 0.45
        elif self.breakType == 'Monochromatic':
            self.Bcoef = 0.78

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
        
        data.close()

        self.split_data()

        self.dx = ((self.Y0[1:]- self.Y0[:-1])**2 + (self.X0[1:]- self.X0[:-1])**2)**0.5
        
        self.yi = self.Obs[0,:]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        self.idx_obs = mkIdx(self.time_obs)

        # Now we calculate the dt from the time variable
        mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
        self.dt = mkDT(np.arange(0, len(self.time)-1))
        mkDTsplited = np.vectorize(lambda i: (self.time_splited[i+1] - self.time_splited[i]).total_seconds()/3600)
        self.dt_splited = mkDTsplited(np.arange(0, len(self.time_splited)-1))

        self.ntrs = len(self.X0)
        
        self.doc = np.zeros_like(self.hs)
        self.depth = np.zeros_like(self.hs) + self.depth
        for k in range(self.ntrs):
            hs12, ts12 = Hs12Calc(self.hs, self.tp)
            self.doc[:,k] = depthOfClosure(hs12, ts12, self.doc_formula)
        

        def run_model(par):
            K = par[0]
            Ymd, _ = hansonKraus1991(self.yi,
                                        self.dt,
                                        self.dx,
                                        self.hs,
                                        self.tp,
                                        self.dir,
                                        self.depth,
                                        self.doc,
                                        K,
                                        self.X0,
                                        self.Y0,
                                        self.phi,
                                        self.bctype,
                                        self.Bcoef)
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
        self.hs = self.hs[ii:, :]
        self.tp= self.tp[ii:, :]
        self.dir = self.dir[ii:, :]

        ii = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]
        self.Obs = self.Obs[ii,:]
        self.time_obs = self.time_obs[ii]



