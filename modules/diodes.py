
import math

diodes={}

_pre_ann={
    5.0e14: 0.5,
    6.5e14: 0.7,
    1.0e15: 1,
    1.5e15: 1.7,
    2.5e15: 3,
    1.0e16: 14
    }

fluence_colours = None

class diode(object):
    def __init__(self,
                 rad,
                 no,
                 thickness
                 ):
        self.rad=rad
        self.no=str(no)
        self.thickness=thickness
        self.ann_offset=_pre_ann[rad]
        self.ann_offset_error=1
        self.area=0.2595 # sensor area in cm2 
        
        if self.no[0] == "6":
            self.area= 0.165 #+- 0.005 cm2
        
        self.const_cap=0
        if self.no[0] == "1":
            self.const_cap = math.sqrt(1/1.2e22)
        elif self.no[0] == "2":
            self.const_cap = math.sqrt(1/5.5e21)
        elif self.no[0] == "3":
            self.const_cap = math.sqrt(1/1.92e21)
        elif self.no[0] == "6":
            self.const_cap = None #unknown
        
    def fluenceCol(self):
        return fluence_colours[self.rad]
        
    def radstr(self):
        return str("{:.1e}".format(self.rad))+" neq/cm$^2$"
        
    def label(self):
        return self.no+', '+self.radstr()
        
    def NEff(self, Vdep, Cend=None):
        
        if Cend is None:
            Cend = self.const_cap

        eps0 = 8.85E-14 # F/cm
        eps = 11.9
        q0 = 1.60E-19 # Coulomb
        
        return 2*eps*eps0 / q0 * Vdep/(self.thickness / 1000. / 10.)**2
        
        return (Cend/self.area)**2 * 2*Vdep/(eps*eps0*q0)
    
    def NEffFromSlope(self,slope):
        eps0 = 8.85E-14 # F/cm
        eps = 11.9
        q0 = 1.60E-19 # Coulomb
        
        return 2./(self.area**2 * eps * eps0 *q0) * 1/slope



D1002=diode(6.5e14,1002,300)
diodes['1002']=D1002

D1003=diode(1.0e15,1003,300)
diodes['1003']=D1003

D1102=diode(1.5e15,1102,300)
diodes['1102']=D1102

D2002=diode(1.0e15,2002,200)
diodes['2002']=D2002

D2003=diode(1.5e15,2003,200)
diodes['2003']=D2003

D2102=diode(2.5e15,2102,200)
diodes['2102']=D2102

D3003=diode(1.0e16,3003,120)
diodes['3003']=D3003

D3007=diode(2.5e15,3007,120)
diodes['3007']=D3007

D3008=diode(1.5e15,3008,120)
diodes['3008']=D3008

D6002=diode(5.0e14,6002,120)
D6002.ann_offset_error = 3
D6002.area=0.2595
diodes['6002']=D6002

#create fluence colours

def _make_colours():
    fluences = [d[1].rad for d in diodes.items()]
    tab_colours=[
        'tab:blue'   ,
        'tab:orange' ,
        'tab:green'  ,
        'tab:red'    ,
        'tab:purple' ,
        'tab:brown'  ,
        'tab:pink'   ,
        'tab:gray'   ,
        'tab:olive'  ,
        'tab:cyan']
    if len(fluences) <= 10: #use tableau
        cols = {}
        for i in range(len(fluences)):
            cols[fluences[i]]=tab_colours[i]
        return cols
    raise ValueError("more than 10 auto fluence colors nor implemented")

fluence_colours = _make_colours()







