import numpy as np
from settings import *

# Seasons and months
seasons = ['summer', 'winter']
months_summer = [6, 7, 8, 9]
months_winter = [1, 2, 3, 4, 5, 10, 11, 12]

# Time of use (ToU) hours
tou_summer = {'supper-off-peak': [],
              'off-peak':list(range(0,16)) + list(range(21,24)),
              'mid-peak': [],
              'on-peak': [16, 17, 18, 19, 20]}
tou_winter = {'supper-off-peak': list(range(8,16)),
              'off-peak':list(range(0,8)) + list(range(21,24)),
              'mid-peak': [16, 17, 18, 19, 20],
              'on-peak': []}
tou_lookup = {'summer': tou_summer, 'winter': tou_winter} # a complete dictionary for ToU hours
time_phase_lookup = {'summer': {}, 'winter': {}} # a dictionary mapping time (hour) and ToU phases
for season in seasons:
    tou = tou_lookup[season]
    phases = list(tou.keys())
    for t in range(T):
        ToD = t % 24 # time of the day
        for phase in tou.keys():
            if ToD in tou[phase]:
                time_phase_lookup[season][t] = phases.index(phase) # use ToU index
            else:
                continue
                
# Flat rate
non_exist_rate = 1e6 # this phase is not applicable for the season

# Energy prices at different ToU phases
costE_summerF = {'supper-off-peak': non_exist_rate,
              'off-peak': 0.11,
              'mid-peak': 0.11,
              'on-peak': 0.11}
costE_winterF = {'supper-off-peak': 0.11,
              'off-peak': 0.11,
              'mid-peak': 0.11,
              'on-peak': non_exist_rate}
# Demand prices at different ToU phases
costD_summerF = {'supper-off-peak': non_exist_rate,
              'off-peak': 29.2,
              'mid-peak': non_exist_rate,
              'on-peak': 29.2}
costD_winterF = {'supper-off-peak': 29.2,
              'off-peak': 29.2,
              'mid-peak': 29.2,
              'on-peak': non_exist_rate}
costs_flat = {"summer": {"E": costE_summerF, "D": costD_summerF}, "winter": {"E": costE_winterF, "D": costD_winterF}}

# Energy-oriented
costE_summerE = {'supper-off-peak': non_exist_rate,
              'off-peak': 0.16056,
              'mid-peak': 0.22885,
              'on-peak': 0.59779}
costE_winterE = {'supper-off-peak': 0.10782,
              'off-peak': 0.11459,
              'mid-peak': 0.19652,
              'on-peak': non_exist_rate}
costD_summerE = {'supper-off-peak': non_exist_rate,
              'off-peak': 11.84,
              'mid-peak': non_exist_rate,
              'on-peak': 11.84}
costD_winterE = {'supper-off-peak': 11.84,
              'off-peak': 11.84,
              'mid-peak': 11.84,
              'on-peak': non_exist_rate}
costs_energy = {"summer": {"E": costE_summerE, "D": costD_summerE}, "winter": {"E": costE_winterE, "D": costD_winterE}}

# Demand-oriented
costE_summerD = {'supper-off-peak': non_exist_rate,
              'off-peak': 0.10286,
              'mid-peak': 0.13226,
              'on-peak': 0.14165}
costE_winterD = {'supper-off-peak': 0.08683,
              'off-peak': 0.10847,
              'mid-peak': 0.11999,
              'on-peak': non_exist_rate}
costD_summerD = {'supper-off-peak': non_exist_rate,
              'off-peak': 17.04,
              'mid-peak': non_exist_rate,
              'on-peak': 31.87}
costD_winterD = {'supper-off-peak': 17.04,
              'off-peak': 17.04,
              'mid-peak': 22.36,
              'on-peak': non_exist_rate}
costs_demand = {"summer": {"E": costE_summerD, "D": costD_summerD}, "winter": {"E": costE_winterD, "D": costD_winterD}}

costE_dict = {}
costD_dict = {}
pricing_plans = ["flat", "energy", "demand"]
costs = [costs_flat, costs_energy, costs_demand]

for pricing_plan, cost in zip(pricing_plans, costs):
    costE_dict[pricing_plan] = {}
    costD_dict[pricing_plan] = {}
    for season in seasons:
        costE_dict[pricing_plan][season] = cost[season]["E"]
        costD_dict[pricing_plan][season] = cost[season]["D"]

class PriceSetter():
    def __init__(self):
        #self.pricing_plan = pricing_plan
        pass
    def generate_pricing(self, pricing_plan):
        
        costE_source_lookup = {'summer': costE_dict[pricing_plan]["summer"],\
                                'winter': costE_dict[pricing_plan]["winter"]}
        costD_source_lookup = {'summer': costD_dict[pricing_plan]["summer"],\
                               'winter': costD_dict[pricing_plan]["winter"]}
        
        costE_lookup = {}
        for season in seasons:
            time_phase_map = time_phase_lookup[season]
            costE_source = costE_source_lookup[season]
            phases = list(costE_source.keys())
            costE = np.zeros(T) # 30 days per month and 24 hours per day
            for t in range(T):
                phase = phases[time_phase_map[t]]
                costE[t] = costE_source[phase]     
            costE_lookup[season] = costE.copy()
        
        costD_lookup = {}
        for season in seasons:
            costD_source = costD_source_lookup[season]
            phases = list(costD_source.keys())
            costD = np.zeros(len(phases))
            for phase in phases:
                h = phases.index(phase)
                costD[h] = costD_source[phase]
            costD_lookup[season] = costD.copy()
        
        return costE_lookup, costD_lookup

        