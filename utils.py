import numpy as np
from settings import *
from pricing import months_summer, months_winter, time_phase_lookup
from pricing import PriceSetter
from dispatch_method_global_optim import Control_GO
from dispatch_method_constant_peak import Control_CP

def get_bill(loads, costE_lookup, costD_lookup):
    
    total_bill = 0
    total_EC = 0
    total_DC = 0
    
    # Calculate bill for each month of the year
    for m in range(12):
        # Determine if the month is summer or winter
        if m in months_summer:
            season = 'summer'
        else:
            season = 'winter'
        # Determine the energy cost and demand cost for each hour the month
        costE = costE_lookup[season]
        costD = costD_lookup[season]
        H = len(costD)
        time_phase_map = time_phase_lookup[season]
        
        # Calculate eneary charge
        load = loads[m].copy()
        EC = np.sum(load*costE*DT)
        
        # Calculate demand charge
        peaks = [0 for h in range(H)]
        for t in range(T):
            h = time_phase_map[t]
            if load[t] > peaks[h]:
                peaks[h] = load[t]
        DC = np.sum([peaks[h] * costD[h] for h in range(H)])
        
        # Bill of the month
        bill = EC + DC
        
        # Add to total bill
        total_bill += bill
        total_EC += EC
        total_DC += DC
        
    return total_EC, total_DC, total_bill

def eval_annual(load_org, battery, method = "constant_peak", pricing_plan = "flat", vocal = False):
    if method == "constant_peak":
        controller = Control_CP(battery, pct_shaved = 0.5)
    elif method == "global_optim":
        controller = Control_GO(battery)
    elif method == "lyapunov":
        controller = Control_LO(battery)
    else:
        print("Control method not recognized.")
    
    priceSetter = PriceSetter()
    costE_lookup, costD_lookup = priceSetter.generate_pricing(pricing_plan)
    
    load_new, Es, cs, ds = controller.simulate(load_org, costE_lookup, costD_lookup)
    EC, DC, bill = get_bill(load_new, costE_lookup, costD_lookup)
    EC0, DC0, bill0 = get_bill(load_org, costE_lookup, costD_lookup)
    saving = bill0 - bill
    if vocal:
        print("Original/New bill:", bill0, bill)
    return saving, load_new, Es, cs, ds
