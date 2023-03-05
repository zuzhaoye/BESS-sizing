# BESS-sizing

This code repo develops a battery energy storage system (BESS) sizing optimization framework for commercial customers considering accurate degradation models. The framework is inspired by [[Ref. 1]](https://ieeexplore.ieee.org/abstract/document/8715850)

Use “Sizing.ipynb” to perform the BESS sizing. The input of the module includes the annual load of a building (in an hourly basis). The negative values indicate surplus of solar generation on the rooftop of the building. Note that the sample data here assumes uniform 30 days of a month. Besides the load information, the users can also specify the desired dispatch algorithm, degradation model, and pricing plan. The algorithm offers three types of pricing plans for selection. Table 1 lists the complete inputs and output.

Table 1: The Inputs and Outputs of the Core Sizing Module
|**Input**               | **Description** |
|:-------------------|:-------------------|
| Annual Building Load | Annual load information of a building in an hourly basis. If there is solar with the building, use negative load when the solar generation exceeds the building load.|
| Dispatch Algorithm   | Select from “global_optim” and “constant_peak”. The “global_optim” algorithm will provide the best possible dispatching results, but it requires a commercial MIP solver to work (e.g. Gurobi). For users without a MIP solver, “constant_peak” is suggested.|
| Degradation Model    | Select from “Xu” [[Ref. 2]](https://ieeexplore.ieee.org/abstract/document/7488267) and “Wang” [[Ref. 3]](https://www.sciencedirect.com/science/article/pii/S0378775310021269). Both models provide reasonable estimation of BESS degradation. Xu’s model is recommended for applications dominated by irregular cycles. Wang’s model is recommended for high C-rate applications.|
| Pricing Plans        | Select from “flat”, “demand”, and “energy”. For the “flat” plan, the energy charge and demand charge remain the same throughout the year, regardless of hour of the day or season. The “demand” and “energy” plans are time-of-use plans, in which “demand” has higher on-peak demand charge while “energy” has higher on-peak energy charge [[Ref. 4]](https://www.sce.com/sites/default/files/custom-files/Web%20files/TOU-GS-2%20Rate%20Fact%20Sheet%200422_WCAG.pdf).|
| Cost of BESS         | The cost of BESS includes battery, power equipment, and construction. Users can tune these parameters in “settings.py”.|
|**Output**               | **Description** |
| NPVs                 | The net-present values of the selected BESS configurations throughout their lifespans. |

## References
1. Zhang, Zhenhai, Jie Shi, Yuanqi Gao, and Nanpeng Yu. "Degradation-aware valuation and sizing of behind-the-meter battery energy storage systems for commercial customers." In 2019 IEEE PES GTD Grand International Conference and Exposition Asia (GTD Asia), pp. 895-900. IEEE, 2019.
2. Xu, Bolun, Alexandre Oudalov, Andreas Ulbig, Göran Andersson, and Daniel S. Kirschen. "Modeling of lithium-ion battery degradation for cell life assessment." IEEE Transactions on Smart Grid 9, no. 2 (2016): 1131-1140. 
3. Wang, John, Ping Liu, Jocelyn Hicks-Garner, Elena Sherman, Souren Soukiazian, Mark Verbrugge, Harshad Tataria, James Musser, and Peter Finamore. "Cycle-life model for graphite-LiFePO4 cells." Journal of power sources 196, no. 8 (2011): 3942-3948.
4. “RATE SCHEDULE TOU-GS-2 for Small- to Medium-Sized Business Customers”, Southern California Edison (SCE), [Online]. Available: https://www.sce.com/sites/default/files/custom-files/Web%20files/TOU-GS-2%20Rate%20Fact%20Sheet%200422_WCAG.pdf
