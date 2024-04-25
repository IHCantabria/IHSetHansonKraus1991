import numpy as np
import xarray as xr
from datetime import datetime
from spotpy.parameter import Uniform
from .HansonKraus1991 import hansonKraus1991
from IHSetCalibration import objective_functions

class cal_HansonKraus1991(object):
    """
    cal_HansonKraus1991
    
    Configuration to calibrate and run the Hanson and Kraus (1991) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
        
        
        mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))

        cfg = xr.open_dataset(path+'config.nc')
        wav = xr.open_dataset(path+'wav.nc')
        ens = xr.open_dataset(path+'ens.nc')
        trs = xr.open_dataset(path+'trs.nc')

        self.cal_alg = cfg['cal_alg'].values
        self.metrics = cfg['metrics'].values
        self.dt = float(cfg['dt'].values)
        self.dx = float(cfg['dx'].values)
        self.bctype = str(cfg['bctype'].values)
        self.switch_Kal = cfg['switch_Kal'].values
        self.breakType = cfg['breakType'].values

        if self.breakType == 'Spectral':
            self.Bcoef = 0.45
        elif self.breakType == 'Monochromatic':
            self.Bcoef = 0.78

        if self.cal_alg == 'NSGAII':
            self.n_pop = cfg['n_pop'].values
            self.generations = cfg['generations'].values
            self.n_obj = cfg['n_obj'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, n_pop=self.n_pop, generations=self.generations, n_obj=self.n_obj)
        else:
            self.repetitions = cfg['repetitions'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, repetitions=self.repetitions)


        self.Y0 = np.array(trs['Y0'].values).squeeze()
        self.X0 = np.array(trs['X0'].values).squeeze()
        self.Xf = np.array(trs['Xf'].values).squeeze()
        self.Yf = np.array(trs['Yf'].values).squeeze()
        self.phi = np.array(trs['phi'].values).squeeze()
        self.yi = np.array(trs['yi'].values).squeeze()

        self.Hs = np.array(wav['Hs'].values).squeeze()
        self.Tp = np.array(wav['Tp'].values).squeeze()
        self.Dir = np.array(wav['Dir'].values).squeeze()
        self.depth = np.array(wav['depth'].values).squeeze()
        self.doc = np.array(wav['doc'].values).squeeze()
        self.time = mkTime(wav['Y'].values, wav['M'].values, wav['D'].values, wav['h'].values)

        self.Obs = np.array(ens['Obs'].values).squeeze()
        self.time_obs = mkTime(ens['Y'].values, ens['M'].values, ens['D'].values, ens['h'].values)

        self.start_date = datetime(int(cfg['Ysi'].values), int(cfg['Msi'].values), int(cfg['Dsi'].values))
        self.end_date = datetime(int(cfg['Ysf'].values), int(cfg['Msf'].values), int(cfg['Dsf'].values))

        self.split_data()

        cfg.close()
        wav.close()
        ens.close()
        trs.close()
        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        self.idx_obs = mkIdx(self.time_obs)

        if self.switch_Kal == 0:
            def model_simulation(par):
                K = par['K']
                Ymd, _ = hansonKraus1991(self.yi,
                                         self.dt,
                                         self.dx,
                                         self.Hs_splited,
                                         self.Tp_splited,
                                         self.Dir_splited,
                                         self.depth,
                                         self.doc,
                                         K,
                                         self.X0,
                                         self.Y0,
                                         self.phi,
                                         self.bctype,
                                         self.Bcoef)
                return Ymd[self.idx_obs_splited, :].flatten()
            
            self.params = [
                Uniform('K', 1e-4, 2)
            ]
            self.model_sim = model_simulation
        elif self.switch_Kal == 1:
            def model_simulation(par):
                for i in range(len(par)):
                    K = list()
                    K = K.append(par['K'+str(i)])
                Ymd, _ = hansonKraus1991(self.yi,
                                         self.dt,
                                         self.dx,
                                         self.Hs_splited,
                                         self.Tp_splited,
                                         self.Dir_splited,
                                         self.depth,
                                         self.doc,
                                         K,
                                         self.X0,
                                         self.Y0,
                                         self.phi,
                                         self.bctype,
                                         self.Bcoef)
                return Ymd[self.idx_obs_splited, :].flatten()
            
            self.params = list()
            for i in range(len(self.X0)):
                self.params.append(Uniform('K'+str(i), 1e-4, 2))
            self.model_sim = model_simulation


    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """
        ii = np.where(self.time>=self.start_date)[0][0]
        self.time = self.time[ii:]
        self.Hs = self.Hs[ii:]
        self.Tp = self.Tp[ii:]
        self.Dir = self.Dir[ii:]

        idx = np.where((self.time < self.start_date) | (self.time > self.end_date))
        self.idx_validation = idx

        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))
        self.idx_calibration = idx
        self.Hs_splited = self.Hs[idx]
        self.Tp_splited = self.Tp[idx]
        self.Dir_splited = self.Dir[idx]
        self.time_splited = self.time[idx]

        idx = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))
        self.Obs_splited = self.Obs[idx,:]
        self.time_obs_splited = self.time_obs[idx]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_splited - t)))
        self.idx_obs_splited = mkIdx(self.time_obs_splited)
        self.observations = self.Obs_splited.flatten()

        # Validation
        idx = np.where((self.time_obs < self.start_date) | (self.time_obs > self.end_date))
        self.idx_validation_obs = idx[0]
        if len(self.idx_validation)>0:
            mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
            if len(self.idx_validation_obs)>0:
                self.idx_validation_for_obs = mkIdx(self.time_obs[idx])
            else:
                self.idx_validation_for_obs = []
        else:
            self.idx_validation_for_obs = []