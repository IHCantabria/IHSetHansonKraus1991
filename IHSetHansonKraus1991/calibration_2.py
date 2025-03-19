import numpy as np
import xarray as xr
import fast_optimization as fo
from .HansonKraus1991 import hansonKraus1991
from IHSetUtils import Hs12Calc, depthOfClosure
import pandas as pd
import json

class cal_HansonKraus1991_2(object):

    """
    cal_HansonKraus1991_2
    
    Configuration to calibrate and run the Hanson and Kraus (1991) One-line Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
        self.name = 'Hanson and Kraus (1991)'
        self.mode = 'calibration'
        self.type = 'OL'

        data = xr.open_dataset(path)

        cfg = json.loads(data.attrs['HansonKraus'])
        self.cfg = cfg

        self.cal_alg = cfg['cal_alg']
        self.metrics = cfg['metrics']
        self.depth = cfg['depth']
        self.switch_Kal = cfg['switch_Kal']
        self.breakType = cfg['break_type']
        self.bctype = cfg['bctype']
        self.doc_formula = cfg['doc_formula']
        self.lb = cfg['lb']
        self.ub = cfg['ub']

        self.calibr_cfg = fo.config_cal(cfg)

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

        self.Obs_ = data.obs.values
        self.Obs = data.obs.values.flatten()
        self.time_obs = pd.to_datetime(data.time_obs.values)
        
        data.close()

        self.split_data()

        self.dx = ((self.Y0[1:]- self.Y0[:-1])**2 + (self.X0[1:]- self.X0[:-1])**2)**0.5
        
        self.yi = self.Obs_splited_[0,:]

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
        

        if self.switch_Kal == 0:
            def model_simulation(par):
                K = np.exp(par[0])
                Ymd, _ = hansonKraus1991(self.yi,
                                         self.dt,
                                         self.dx,
                                         self.hs_splited,
                                         self.tp_splited,
                                         self.dir_splited,
                                         self.depth,
                                         self.doc,
                                         K,
                                         self.X0,
                                         self.Y0,
                                         self.phi,
                                         self.bctype,
                                         self.Bcoef)
                
                return Ymd[self.idx_obs_splited, :].flatten()

            self.model_sim = model_simulation

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

            def init_par(population_size):
                log_lower_bounds = np.array([self.lb[0]])
                log_upper_bounds = np.array([self.ub[0]])
                population = np.zeros((population_size, 1))
                for i in range(1):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

        elif self.switch_Kal == 1:

            self.lb = np.repeat(self.lb, self.ntrs)
            self.ub = np.repeat(self.ub, self.ntrs)


            def model_simulation(par):
                K = []
                for i in range(len(par)):
                    K.append(np.exp(par[i]))
                K = np.array(K)
                Ymd, _ = hansonKraus1991(self.yi,
                                         self.dt,
                                         self.dx,
                                         self.hs_splited,
                                         self.tp_splited,
                                         self.dir_splited,
                                         self.depth,
                                         self.doc,
                                         K,
                                         self.X0,
                                         self.Y0,
                                         self.phi,
                                         self.bctype,
                                         self.Bcoef)
                return Ymd[self.idx_obs_splited, :].flatten()

            self.model_sim = model_simulation

            def run_model(par):
                K = []
                for i in range(len(par)):
                    K.append(par[i])
                K = np.array(K)
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

            def init_par(population_size):
                log_lower_bounds = np.array(self.lb)
                log_upper_bounds = np.array(self.ub)
                population = np.zeros((population_size, self.ntrs))
                for i in range(self.ntrs):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par


    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """
        ii = np.where(self.time>=self.start_date)[0][0]
        self.time = self.time[ii:]
        self.hs = self.hs[ii:, :]
        self.tp= self.tp[ii:, :]
        self.dir = self.dir[ii:, :]

        idx = np.where((self.time < self.start_date) | (self.time > self.end_date))[0]
        self.idx_validation = idx

        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
        self.idx_calibration = idx
        self.hs_splited = self.hs[idx, :]
        self.tp_splited = self.tp[idx, :]
        self.dir_splited = self.dir[idx, :]
        self.time_splited = self.time[idx]

        idx = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]
        self.Obs_splited_ = self.Obs_[idx,:]
        self.Obs_splited = self.Obs_splited_.flatten()
        self.time_obs_splited = self.time_obs[idx]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_splited - t)))
        self.idx_obs_splited = mkIdx(self.time_obs_splited)

        # Validation
        idx = np.where((self.time_obs < self.start_date) | (self.time_obs > self.end_date))[0]
        self.idx_validation_obs = idx
        if len(self.idx_validation)>0:
            mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
            if len(self.idx_validation_obs)>0:
                self.idx_validation_for_obs = mkIdx(self.time_obs[idx])
            else:
                self.idx_validation_for_obs = []
        else:
            self.idx_validation_for_obs = []

    def calibrate(self):
        """
        Calibrate the model.
        """
        self.solution, self.objectives, self.hist = self.calibr_cfg.calibrate(self)
        self.solution = np.exp(self.solution)

        if self.switch_Kal == 0:
            self.par_names = []
            for i in range(len(self.solution)):
                self.par_names.append(rf'K_{i}')
            self.par_values = self.solution.copy()
        elif self.switch_Kal == 1:
            self.par_names = ['K']
            self.par_values = self.solution.copy()
