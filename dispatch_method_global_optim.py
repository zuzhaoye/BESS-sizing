import numpy as np
from mip import *
from settings import *
from pricing import months_summer, months_winter, time_phase_lookup

class Control_GO():
    def __init__(self, battery):
        self.battery = battery
        
    def global_optim(self, load_org, costE_lookup, costD_lookup):

        # retrieve battery data
        Emax = self.battery.Emax
        Emin = self.battery.Emin
        Eini = self.battery.Eini
        Pmax = self.battery.Pmax
        Pmin = self.battery.Pmin
        kappa = self.battery.eff

        G = 1e8 # a large number

        ################# initialize model #################
        Mod = Model(sense=MINIMIZE, solver_name=GRB) # use GRB for Gurobi

        ################# create decision variables #################
        e = []
        c = []
        d = []
        load_new = []
        peaks = []
        EC = []
        DC = []
        bill = []

        for m in range(12):
            e.append([Mod.add_var(name='e, t={}, m={}'.format(t, m),\
                                   var_type=CONTINUOUS, lb=Emin, ub=Emax) for t in range(T)])
            c.append([Mod.add_var(name='c, t={}, m={}'.format(t, m),\
                                   var_type=CONTINUOUS, lb=0, ub=-Pmin) for t in range(T)])
            d.append([Mod.add_var(name='d, t={}, m={}'.format(t, m),\
                                   var_type=CONTINUOUS, lb=0, ub=Pmax) for t in range(T)])
            load_new.append([Mod.add_var(name='load_new, t={}, m={}'.format(t, m),\
                                   var_type=CONTINUOUS, lb=Pmin_building, ub=Pmax_building) for t in range(T)])

            if m in months_summer:
                season = 'summer'
            else:
                season = 'winter'
            costD = costD_lookup[season]
            H = len(costD)
            peaks.append([Mod.add_var(name='peak, h={}, m={}'.format(h, m),\
                                   var_type=CONTINUOUS, lb=0, ub=Pmax_building) for h in range(H)])

            EC.append(Mod.add_var(name='energy charge, m={}'.format(m), var_type=CONTINUOUS, lb=-G, ub=G))
            DC.append(Mod.add_var(name='demand charge, m={}'.format(m), var_type=CONTINUOUS, lb=-G, ub=G))
            bill.append(Mod.add_var(name='bill, m={}'.format(m), var_type=CONTINUOUS, lb=-G, ub=G))

        ################# add constraints #################
        # SoC update and range constraints

        for m in range(12):
            if m in months_summer:
                season = 'summer'
            else:
                season = 'winter'
            costE = costE_lookup[season]
            costD = costD_lookup[season]
            H = len(costD)
            time_phase_map = time_phase_lookup[season]

            if m == 0:
                Mod += e[m][0] == Eini
            else:
                Mod += e[m][0] == e[m-1][T-1]

            for t in range(1,T):
                Mod += e[m][t] == e[m][t-1] + (c[m][t-1]-d[m][t-1])*DT\
                                    -(c[m][t-1]+d[m][t-1])*DT*(1-np.sqrt(kappa))

            for t in range(T):
                h = time_phase_map[t]
                Mod += load_new[m][t] == load_org[m][t]-d[m][t]+c[m][t]
                Mod += load_new[m][t] <= peaks[m][h]

            Mod += EC[m] == xsum(load_new[m][t]*costE[t]*DT for t in range(T))
            Mod += DC[m] == xsum(peaks[m][h]*costD[h] for h in range(H))
            Mod += bill[m] == EC[m] + DC[m]

        ################# objective #################

        Mod.objective = xsum(bill[m] for m in range(12))

        return Mod


    def get_global_optim_results(self, Mod):
        res = dict()
        res['objective'] = Mod.objective_value
        res['optimal_status'] = Mod.status
        #res['solver_time'] = t_solve
        res['log'] = Mod.search_progress_log.log.copy()

        res['properties'] = {}
        res['properties']['max_mip_gap'] = Mod.max_mip_gap
        if len(res['log']) > 0:
            item = res['log'][-1]
            res['properties']['final_mip_gap'] = abs((item[1][0] - item[1][1])/item[1][0])
        else:
            res['properties']['final_mip_gap'] = None
        res['properties']['n_threads'] = Mod.threads
        res['properties']['n_cols'] = Mod.num_cols
        res['properties']['n_int'] = Mod.num_int
        res['properties']['n_rows'] = Mod.num_rows
        res['properties']['n_nz'] = Mod.num_nz
        res['properties']['n_sol'] = Mod.num_solutions
        res['dt'] = DT

        res['e'] = np.zeros((12, T))
        res['c'] = np.zeros((12, T))
        res['d'] = np.zeros((12, T))
        res['x'] = np.zeros((12, T))
        res['load_new'] = np.zeros((12, T))
        res['peak'] = np.zeros((12, 4))
        res['EC'] = np.zeros(12)
        res['DC'] = np.zeros(12)
        res['bill'] = np.zeros(12)

        for m in range(12):
            for t in range(T):
                res['e'][m][t] = Mod.vars['e, t={}, m={}'.format(t, m)].x
                res['c'][m][t] = Mod.vars['c, t={}, m={}'.format(t, m)].x
                res['d'][m][t] = Mod.vars['d, t={}, m={}'.format(t, m)].x
                res['x'][m][t] = res['c'][m][t] - res['d'][m][t]
                res['load_new'][m][t] = Mod.vars['load_new, t={}, m={}'.format(t, m)].x

            for h in range(H):
                res['peak'][m][h] = Mod.vars['peak, h={}, m={}'.format(h, m)].x

            res['EC'][m] = Mod.vars['energy charge, m={}'.format(m)].x
            res['DC'][m] = Mod.vars['demand charge, m={}'.format(m)].x
            res['bill'][m] = Mod.vars['bill, m={}'.format(m)].x

        return res

    def simulate(self, load_org, costE_lookup, costD_lookup):
        mod = self.global_optim(load_org, costE_lookup, costD_lookup)
        status = mod.optimize(max_seconds=500)
        res = self.get_global_optim_results(mod)

        load_new = res['load_new'].copy()
        Es = res['e'].copy()
        cs = res['c'].copy()
        ds = res['d'].copy()

        return load_new, Es, cs, ds