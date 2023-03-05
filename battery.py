import numpy as np
import rainflow
from settings import DT as dt

class Battery:
    def __init__(self, Emax, Emin, Pmax, Pmin, eff, soc_ini, degModel="constant"):
        self.Emax = Emax
        self.Emin = Emin
        self.Pmax = Pmax
        self.Pmin = Pmin
        self.Eini = soc_ini*self.Emax
        self.E = self.Eini
        self.eff = eff
        self.degModel = degModel
        
    def charge(self, dp_tgt):
        
        assert dp_tgt >= 0
        dp_tgt = min(dp_tgt, abs(self.Pmin))
        dE_pre = dp_tgt*dt - (1-np.sqrt(self.eff))*dp_tgt*dt
        room = self.Emax - self.E
        
        if room >= dE_pre:
            dE = dE_pre
            dp = dp_tgt 
        else:
            dE = room
            dp = dE/(np.sqrt(self.eff)*dt)

        self.E += dE
        
        return dp
    
    def discharge(self, dp_tgt):
        
        assert dp_tgt >= 0
        dp_tgt = min(dp_tgt, self.Pmax)
        dE_pre = dp_tgt*dt + (1-np.sqrt(self.eff))*dp_tgt*dt
        reserve = self.E - self.Emin
        
        if reserve >= dE_pre:
            dE = dE_pre
            dp = dp_tgt
        else:
            dE = reserve
            dp = dE/((2-np.sqrt(self.eff))*dt)
        
        self.E -= dE
        return dp
    
    def degradate(self, degCap=0.05, degEff=0.02, Es=np.zeros(8640,), xs=np.zeros(8640,)):
        
        if self.degModel == "constant":
            self.Emax *= (1-degCap)
            self.eff *= (1-degEff)
        
        elif self.degModel == "Xu":
            kd1 = 1.4e5
            kd2 = -0.501e1
            kd3 = -1.23e5

            ks = 1.04
            SoC_ref = 0.50

            kt = 4.14e-10 # per second

            kT = 6.93e-2
            T_ref = 25

            fd = 0
            socs = Es/self.Emax
            cycles = rainflow.count_cycles(socs)

            # calendar aging
            Stime = kt*3600*8640 # 3600s per hour and 8640 hours per year
            fd += Stime

            # stress aging
            for depth, number in cycles:
                Sdod = (kd1*depth**kd2 + kd3)**-1 * number
                #Ssoc = np.exp(ks*(socs.mean()-SoC_ref))
                Ssoc = 1
                #Stemp = np.exp(kT*(T-T_ref)*(T_ref/T))
                Stemp = 1
                fd += Sdod*Ssoc*Stemp

            degCap = 1-np.exp(-fd)
            self.Emax *= (1-degCap)
            self.eff *= (1-degEff)
            
        elif self.degModel == "Wang":
            b = 31630
            T = 288
            R = 8.3145
            C_rate = 0.5
            if self.Emax > 0:
                Ah = np.abs(xs).sum()*dt/self.Emax
            else:
                Ah = 0

            if self.Emax < 0.7*self.Eini:
                print('Battery should be retired')

            degCap = b * np.exp((-31700 + 370.3*C_rate)/(R*T)) * (Ah**0.55)
            degCap /= 100
            self.Emax *= (1-degCap)
            self.eff *= (1-degEff)
        else:
            print("Degradation model not in the list of [constant, Xu, Wang], please specify.")
    