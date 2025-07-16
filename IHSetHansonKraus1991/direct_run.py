import numpy as np
import xarray as xr
import pandas as pd
import fast_optimization as fo
from IHSetUtils import Hs12Calc, depthOfClosure, nauticalDir2cartesianDir
from .HansonKraus1991 import hansonKraus1991 as hk1991
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

        self.mb = 1/100 # Default value for mb in Kamphuis (2002)
        self.D50 = 0.3e-3  # Default value for D50 in Kamphuis (2002)

        
        self.switch_Kal = cfg['switch_Kal']
        self.breakType = cfg['break_type']
        self.bctype = cfg['bctype']
        self.doc_formula = cfg['doc_formula']
        self.formulation = cfg['formulation']


        self.start_date = pd.to_datetime(cfg['start_date'])
        self.end_date = pd.to_datetime(cfg['end_date'])
        
        if self.breakType == 'Spectral':
            self.Bcoef = 0.45
        elif self.breakType == 'Monochromatic':
            self.Bcoef = 0.78


        if self.formulation == 'CERC (1984)':
            print('Using CERC (1984) formulation')
            from IHSetUtils.libjit.morfology import CERC_ALST as lst_f
        elif self.formulation == 'Komar (1998)':
            print('Using Komar (1998) formulation')
            from IHSetUtils.libjit.morfology import Komar_ALST as lst_f
        elif self.formulation == 'Kamphuis (2002)':
            print('Using Kamphuis (2002) formulation')
            from IHSetUtils.libjit.morfology import Kamphuis_ALST as lst_f
            self.mb = cfg['mb']
            self.D50 = cfg['D50']
        elif self.formulation == 'Van Rijn (2014)':
            print('Using Van Rijn (2014) formulation')
            from IHSetUtils.libjit.morfology import VanRijn_ALST as lst_f
            self.mb = cfg['mb']
            self.D50 = cfg['D50']
        

        bc_conv = [0,0]
        if self.bctype[0] == 'Dirichlet':
            bc_conv[0] = 0
        elif self.bctype[0] == 'Neumann':
            bc_conv[0] = 1
        if self.bctype[1] == 'Dirichlet':
            bc_conv[1] = 0
        elif self.bctype[1] == 'Neumann':
            bc_conv[1] = 1
        
        self.bctype = np.array(bc_conv)

        self.Y0 = data.yi.values
        self.X0 = data.xi.values
        self.Xf = data.xf.values
        self.Yf = data.yf.values
        self.phi = data.phi.values
        self.depth = data.waves_depth.values
        
        self.hs = data.hs.values
        self.tp= data.tp.values
        self.dir = data.dir.values
        self.dir = nauticalDir2cartesianDir(self.dir)
        self.time = pd.to_datetime(data.time.values)

        self.Obs = data.obs.values
        self.time_obs = pd.to_datetime(data.time_obs.values)

        self.ntrs = len(self.X0)
        
        data.close()

        self.interp_forcing()
        self.split_data()
        
        self.yi = np.zeros_like(self.Obs[0,:])
        for i in range(self.ntrs):
            self.yi[i] = np.nanmean(self.Obs[:, i])

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        self.idx_obs = mkIdx(self.time_obs)

        # Now we calculate the dt from the time variable
        mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds())
        self.dt = mkDT(np.arange(0, len(self.time)-1))

        
        self.doc = np.zeros_like(self.hs_)
        for k in range(self.doc.shape[1]):
            hs12, ts12 = Hs12Calc(self.hs_[:,k], self.tp_[:,k])
            self.doc[:,k] = depthOfClosure(hs12, ts12, self.doc_formula)
                
        def run_model(par):
            K = par
            Ymd, _ = hk1991(self.yi,
                            self.dt,
                            # self.dx,
                            self.hs_,
                            self.tp_,
                            self.dir_,
                            self.depth_,
                            self.doc,
                            K,
                            self.X0,
                            self.Y0,
                            self.phi,
                            self.bctype,
                            self.Bcoef,
                            self.mb,
                            self.D50,
                            lst_f)
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

        # self.calculate_metrics()

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
        doc(time, trs) -> doc(time, trs+0.5)
        depth(trs) -> depth(time, trs+0.5)
        """

        dist = np.hstack((0,np.cumsum(np.sqrt(np.diff(self.Xf)**2 + np.diff(self.Yf)**2))))
        dist_ = dist[1:] - (dist[1:]-dist[:-1])/2

        
        self.hs_ = np.zeros((len(self.time), self.ntrs+1))
        self.tp_ = np.zeros((len(self.time), self.ntrs+1))
        self.dir_ = np.zeros((len(self.time), self.ntrs+1))
        self.depth_ = np.zeros((self.ntrs+1))

        self.hs_[:, 0], self.hs_[:, -1] = self.hs[:, 0], self.hs[:, -1]
        self.tp_[:, 0], self.tp_[:, -1] = self.tp[:, 0], self.tp[:, -1]
        self.dir_[:, 0], self.dir_[:, -1] = self.dir[:, 0], self.dir[:, -1]
        self.depth_[0], self.depth_[-1] = self.depth[0], self.depth[-1]

        # self.hs_[:, 1], self.hs_[:, -2] = self.hs[:, 0], self.hs[:, -1]
        # self.tp_[:, 1], self.tp_[:, -2] = self.tp[:, 0], self.tp[:, -1]
        # self.dir_[:, 1], self.dir_[:, -2] = self.dir[:, 0], self.dir[:, -1]
        # self.depth_[1], self.depth_[-2] = self.depth[0], self.depth[-1]

        for i in range(len(self.time)):
            # self.hs_[i, 2:-2] = np.interp(dist_, dist, self.hs[i, :])
            # self.tp_[i, 2:-2] = np.interp(dist_, dist, self.tp[i, :])
            # self.dir_[i, 2:-2] = np.interp(dist_, dist, self.dir[i, :])
            self.hs_[i, 1:-1] = np.interp(dist_, dist, self.hs[i, :])
            self.tp_[i, 1:-1] = np.interp(dist_, dist, self.tp[i, :])
            self.dir_[i, 1:-1] = np.interp(dist_, dist, self.dir[i, :])


        # self.depth_[2:-2] = np.interp(dist_, dist, self.depth)
        self.depth_[1:-1] = np.interp(dist_, dist, self.depth)


        # self.hs_ = self.hs
        # self.tp_ = self.tp
        # self.dir_ = self.dir
        # self.depth_ = self.depth
        



