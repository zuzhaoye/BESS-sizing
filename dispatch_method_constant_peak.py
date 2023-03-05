import numpy as np
from settings import *
from pricing import months_summer, months_winter, time_phase_lookup

class Control_CP():
    def __init__(self, battery, pct_shaved):
        self.battery = battery
        self.pct_shaved = pct_shaved
        
    def simulate(self, load_org, costE_lookup, costD_lookup):
        Ps = np.zeros((12, T))
        Es = np.zeros((12, T))
        cs = np.zeros((12, T))
        ds = np.zeros((12, T))

        for m in range(12):
            # find maximum and mean load
            peak_load = np.max(load_org[m])
            mean_load = np.mean(load_org[m])
            diff_load = peak_load - mean_load

            p_thres_u = mean_load + (1-self.pct_shaved)*diff_load # reduce 95% of the (peak-average) load
            p_thres_l = p_thres_u

            for t in range(T):
                p = load_org[m][t]
                p_new = p

                # if p greater than upper, discharge
                if p > p_thres_u:
                    dp_tgt = p - p_thres_u # target discharging power
                    dp = self.battery.discharge(dp_tgt) # actual discharging power
                    p_new = p - dp # new load point after shaving
                    ds[m][t] = dp

                # if p less than lower, charge
                if p < p_thres_l:
                    dp_tgt = p_thres_l - p
                    dp = self.battery.charge(dp_tgt)
                    p_new = p + dp
                    cs[m][t] = dp

                Ps[m][t] = p_new
                Es[m][t] = self.battery.E

        load_new = Ps.copy()     
        return load_new, Es, cs, ds