import numpy as np
from .HansonKraus1991 import hansonKraus1991 as hk1991
from IHSetUtils.CoastlineModel import CoastlineModel


class cal_HansonKraus1991_2(CoastlineModel):

    """
    cal_HansonKraus1991_2
    
    Configuration to calibrate and run the Hanson and Kraus (1991) One-line Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Hanson and Kraus (1991)',
            mode='calibration',
            model_type='OL',
            model_key='HansonKraus'
        )

        self.setup_forcing()


    def setup_forcing(self):
        self.switch_Kal = self.cfg['switch_Kal']
        self.y_ini = np.zeros_like(self.Obs_splited_[0,:])
        for i in range(self.ntrs):
            self.y_ini[i] = np.nanmean(self.Obs_splited_[:, i])
    
    def init_par(self, population_size: int):
        if self.switch_Kal == 0:
            if self.is_exp:
                lowers = np.log(np.array(self.lb))
                uppers = np.log(np.array(self.ub))
            else:
                lowers = np.array(self.lb)
                uppers = np.array(self.ub)
            pop = np.zeros((population_size, 1))
        else:
            if self.is_exp:
                lowers = np.log(np.array(self.lb))
                uppers = np.log(np.array(self.ub))
            else:
                lowers = np.array(self.lb)
                uppers = np.array(self.ub)
            pop = np.zeros((population_size, self.ntrs))
        for i in range(pop.shape[1]):
            pop[:, i] = np.random.uniform(lowers, uppers, population_size)
        return pop, lowers, uppers

    def model_sim(self, par: np.ndarray) -> np.ndarray:

        if self.switch_Kal == 0:
            if self.is_exp:
                K = np.exp(par)
            else:
                K = par
        else:
            K = []
            for i in range(len(par)):
                if self.is_exp:
                    K.append(np.exp(par[i]))
                else:
                    K.append(par[i])
            K = np.array(K)

        Ymd, _ = hk1991(self.y_ini,
                        self.dt,
                        self.hs_s,
                        self.tp_s,
                        self.dir_s,
                        self.depth,
                        self.doc,
                        K,
                        self.X0,
                        self.Y0,
                        self.phi,
                        self.bctype,
                        self.Bcoef,
                        self.mb,
                        self.D50,
                        self.lst_f)


        return Ymd[self.idx_obs_splited].flatten()

    def run_model(self, par: np.ndarray) -> np.ndarray:
        K = par
        Ymd, _ = hk1991(self.y_ini, #y_ini,        #
                        self.dt,
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
                        self.Bcoef,
                        self.mb,
                        self.D50,
                        self.lst_f)
        return Ymd

    def _set_parameter_names(self):
        if self.switch_Kal == 0:
            self.par_names = ['K']
        elif self.switch_Kal == 1:
            self.par_names = []
            for i in range(len(self.solution)):
                self.par_names.append(rf'K_trs_{i+1}')
        if self.is_exp:
            self.par_values = np.exp(self.par_values)
