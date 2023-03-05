import numpy as np

cost_capacity = 50 # $50/kWh for second-life battery
cost_power_equip = 100 # $100/kW for power equipment
cost_construction = 100 # $100/kW for construction

degCap = 0.02 # default capacity degradation speed per year (-5% capacity per year)
degEff = 0.02 # default energy efficiency degradation speed per year (-2% efficiency per year)

Nyear = 10 # Study time horizon
ir = 0.10 # Interest rate

Pmax_building = 200
Pmin_building = -200

T = 30*24
DT = 1
H = 3
H_adj_factor = 1

# # Seasons and months
# seasons = ['summer', 'winter']
# months_summer = [6, 7, 8, 9]
# months_winter = [1, 2, 3, 4, 5, 10, 11, 12]

# # Time of use (ToU) hours
# tou_summer = {'supper-off-peak': [],
#               'off-peak':list(range(0,16)) + list(range(21,24)),
#               'mid-peak': [],
#               'on-peak': [16, 17, 18, 19, 20]}
# tou_winter = {'supper-off-peak': list(range(8,16)),
#               'off-peak':list(range(0,8)) + list(range(21,24)),
#               'mid-peak': [16, 17, 18, 19, 20],
#               'on-peak': []}
# tou_lookup = {'summer': tou_summer, 'winter': tou_winter} # a complete dictionary for ToU hours

# time_phase_lookup = {'summer': {}, 'winter': {}} # a dictionary mapping time (hour) and ToU phases
# for season in seasons:
#     tou = tou_lookup[season]
#     phases = list(tou.keys())
#     for t in range(T):
#         ToD = t % 24 # time of the day
#         for phase in tou.keys():
#             if ToD in tou[phase]:
#                 time_phase_lookup[season][t] = phases.index(phase) # use ToU index
#             else:
#                 continue
                
# # Energy prices at different ToU phases
# costE_summer = {'supper-off-peak': 0,
#               'off-peak': 0.16056,
#               'mid-peak': 0.22885,
#               'on-peak': 0.59779}
# costE_winter = {'supper-off-peak': 0.10782,
#               'off-peak': 0.11459,
#               'mid-peak': 0.19652,
#               'on-peak': 0}
# costE_source_loopkup = {'summer': costE_summer, 'winter': costE_winter}
# costE_lookup = {}
# for season in seasons:
#     time_phase_map = time_phase_lookup[season]
#     costE_source = costE_source_loopkup[season]
#     phases = list(costE_source.keys())
#     costE = np.zeros(T) # 30 days per month and 24 hours per day
#     for t in range(T):
#         phase = phases[time_phase_map[t]]
#         costE[t] = costE_source[phase]     
#     costE_lookup[season] = costE.copy()
    
# # Demand prices at different ToU phases
# costD_summer = {'supper-off-peak': 0,
#               'off-peak': 11.84,
#               'mid-peak': 0,
#               'on-peak': 11.84}
# costD_winter = {'supper-off-peak': 11.84,
#               'off-peak': 11.84,
#               'mid-peak': 11.84,
#               'on-peak': 0}
# costD_source_lookup = {'summer': costD_summer, 'winter': costD_winter}
# costD_lookup = {}
# for season in seasons:
#     costD_source = costD_source_lookup[season]
#     phases = list(costD_source.keys())
#     costD = np.zeros(len(phases))
#     for phase in phases:
#         h = phases.index(phase)
#         costD[h] = costD_source[phase]
#     costD_lookup[season] = costD.copy()