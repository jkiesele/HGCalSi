
diodes={}

_pre_ann={
    6.5e14: 0.7,
    1.0e15: 1,
    1.5e15: 1.7,
    2.5e15: 3,
    1.0e16: 14
    }

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
        self.area=0.2595 # sensor area in cm2 
        
    def toNEff(self, Vdep, Cend):

        eps0 = 8.85E-14 # F/cm
        eps = 11.9
        q0 = 1.60E-19 # Coulomb
        
        return (Cend/SArea)*2 * 2*Vdep/(eps*eps0*q0)



D1002=diode(6.5e14,1002,300)
diodes['1002']=D1002

D1003=diode(1.0e15,1003,300)
diodes['1003']=D1003

D1102=diode(1.5e15,1102,300)
diodes['1102']=D1102

D2002=diode(1.0e15,2002,200)
diodes['2002']=D2002

D2003=diode(1.5e15,2002,200)
diodes['2003']=D2003

D2102=diode(2.5e15,2102,200)
diodes['2102']=D2102

D3003=diode(1.0e16,3003,120)
diodes['3003']=D3003

D3007=diode(2.5e15,3007,120)
diodes['3007']=D3007

D3008=diode(1.5e15,3008,120)
diodes['3008']=D3008


