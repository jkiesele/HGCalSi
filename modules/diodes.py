
import math
import numpy as np


# data here
_pre_ann={
    9.9e14: 12.4,#keep this
    1.4e15: 7.8,#keep this
    1.8e15: 20,#keep
    2.1e15: 20,#keep
    }

#from JSI
_rad_times={
    5e14: 5.7,
    6.5e14: 7.4,
    1e15: 10.8,
    1.5e15: 16.2,
    2.5e15: 27.5,
    1e16: 108
    }

_rad_temp_low=45
_rad_temp=50
_rad_temp_hi=55



def radiation_annealing_Ljubljana(radtime, plot=False, maxtemp=45):
    
    import alpha_calc #this cpython lib import might not work on all machine sin the same way

#now calculate average annealing time assuming linearly increasing radiation damage per second (because why not)

    rising_annealing=np.array([i for i in range(int(radtime*60))],dtype='float32') #again in minutes
    falling_annealing=np.array([i+rising_annealing[-1] for i in range(30*60)],dtype='float32')#30 minutes cool off
    
    rising_annealing = rising_annealing/60
    falling_annealing= falling_annealing/60
    
    #back to minutes
    
    
    def rising(t_in_minutes):
        return maxtemp - 25.*math.exp( - (t_in_minutes)/3.5 ) 
    
    def falling(t_in_minutes,last_risen):
        return ((maxtemp-25)+(last_risen-maxtemp))*math.exp( - (t_in_minutes - radtime)/8 ) + 25
    
    xr = [i for i in rising_annealing]
    xf = [i for i in falling_annealing]
    
    rise = np.array([rising(i) for i in xr], dtype='float32')
    
    fall = np.array([falling(i,rise[-1]) for i in xf], dtype='float32')
    
    
    x=xr+xf
    y=np.concatenate([rise,fall],axis=0)
    
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(x,y)
        plt.xlabel("minutes")
        plt.xlabel("temperature in reactor")
        plt.show()
    
    
    #get full equiv annealing time for all in rise
    time_bins = []
    for T in y:
        time_bins.append(alpha_calc.equiv_annealing_time(T, 1/60., 60.))#per second
        
    fraction_irradiated = xr/np.max(xr)
    fraction_irradiated = np.concatenate([fraction_irradiated, falling_annealing*0.+1],axis=0)#to have full range
    
    totaltime = 0
    for t,f in zip(time_bins,fraction_irradiated):
        totaltime +=  t*f
    
    return totaltime
 


diodes={}


#for k in _pre_ann.keys():
#    _pre_ann[k]*=10.

fluence_colours = None

class diode(object):
    def __init__(self,
                 rad,
                 no,
                 thickness,
                 material
                 ):
        self.rad=rad
        self.no=str(no)
        self.thickness=thickness
        self.radtime = -1.
        self.ann_offset = 0.
        self.ann_offset_up = 0.
        self.ann_offset_down = 0.
        ## ## ##
        if rad in _pre_ann.keys():
            self.ann_offset=_pre_ann[rad]
            self.ann_offset_up = _pre_ann[rad]
            self.ann_offset_down = _pre_ann[rad]
        else:
            if not rad in _rad_times.keys():
                raise ValueError("diode: fluence value not associated with irradiation time")
            self.radtime = _rad_times[rad]
            self.ann_offset = radiation_annealing_Ljubljana(self.radtime,False,_rad_temp)  
            self.ann_offset_up = radiation_annealing_Ljubljana(self.radtime,False,_rad_temp_hi)  
            self.ann_offset_down = radiation_annealing_Ljubljana(self.radtime,False,_rad_temp_low)  
        
        self.ann_offset_error = np.max([np.abs(self.ann_offset-self.ann_offset_up) ,
                                        np.abs(self.ann_offset-self.ann_offset_down)])
        self.ann_offset_error=np.sqrt(1.+self.ann_offset_error**2)
        
        
        self.area=0.2595 # sensor area in cm2 
        self.material=material
        self.udep_norad=1
        
        if self.no[0] == "6":
            self.area= 0.165 #+- 0.005 cm2
        
        self.const_cap=0
        if self.no[0] == "1":
            self.const_cap = math.sqrt(1/1.2e22)
            self.udep_norad = 264
        elif self.no[0] == "2":
            self.const_cap = math.sqrt(1/5.5e21)
            self.udep_norad = 126
        elif self.no[0] == "3":
            self.const_cap = math.sqrt(1/1.92e21)
            self.udep_norad = 44
        elif self.no[0] == "6":
            self.const_cap = None #unknown
        
    def fluenceCol(self):
        return fluence_colours[self.rad]
        
    def radstr(self):
        return str("{:.1e}".format(self.rad))+" neq/cm$^2$"
        
    def rad_str(self):#consistency
        return self.radstr()    
        
    def label(self):
        return self.no+', '+self.radstr()
    
    def paperlabel(self, 
                   mode='',
                   addfluence=True):
        
        if mode == 'fluence':
            return self.rad_str()
        if mode == 'material,fluence':
            return self.material_str()+' '+self.rad_str()
        
        if addfluence:
            return self.thickness_str()+' '+self.material_str()+' '+self.rad_str()
        else:
            return self.thickness_str()+' '+self.material_str()
    
    def label_str(self): #just for consistency
        return self.label()
    
    def thickness_str(self):
        return str(self.thickness)+'$\,\mu$m'
    
    def material_str(self):
        return self.material
    
    def __str__(self):
        roffs = round(self.ann_offset,1)
        roffserrup = round(self.ann_offset_up-self.ann_offset,1)
        roffserrdown = round(self.ann_offset-self.ann_offset_down,1)
        return self.label() + ", ann_offset " + str(roffs) +" $\\pm^{"+ str(roffserrup)+"}_{"+str(roffserrdown)+"}$"
    
    @property
    def thickness_cm(self):
        return self.thickness / (1000. * 10.)
    
    def Cideal(self):
        eps = 11.9
        eps0 = 8.85E-14
        return eps0*eps * self.area / self.thickness_cm 
        
        
    def NEff(self, Vdep, Cend=None):
        
        if Cend is None:
            Cend = self.const_cap

        eps0 = 8.85E-14 # F/cm
        eps = 11.9
        q0 = 1.60E-19 # Coulomb
        
        #return Vdep * 2.*eps*eps0 / q0 * 1./(self.thickness_cm)**2
        
        return (Cend/self.area)**2 * 2*Vdep/(eps*eps0*q0)
    
    def VDep(self, Neff, Cend=None):
        
        if Cend is None:
            Cend = self.const_cap

        eps0 = 8.85E-14 # F/cm
        eps = 11.9
        q0 = 1.60E-19 # Coulomb
        
        return Neff / (2.*eps*eps0 / q0 * 1./(self.thickness_cm)**2)
        
    def NEff_norad(self):
        return self.NEff(self.udep_norad)
    
    def NEffFromSlope(self,slope):
        eps0 = 8.85E-14 # F/cm
        eps = 11.9
        q0 = 1.60E-19 # Coulomb
        
        return 2./(self.area**2 * eps * eps0 *q0) * 1/slope



D1002=diode(6.5e14,1002,300,"FZ")
diodes['1002']=D1002

D1003=diode(1.0e15,1003,300,"FZ")
diodes['1003']=D1003

D1102=diode(1.5e15,1102,300,"FZ")
diodes['1102']=D1102

D1113=diode(9.9e+14,1113,300,"FZ")
diodes['1113']=D1113

D1114=diode(1.4e+15,1114,300,"FZ")
diodes['1114']=D1114

D2002=diode(1.0e15,2002,200,"FZ")
diodes['2002']=D2002

D2003=diode(1.5e15,2003,200,"FZ")
diodes['2003']=D2003

D2102=diode(2.5e15,2102,200,"FZ")
diodes['2102']=D2102

D2114=diode(2.1e+15,2114,200,"FZ")
diodes['2114']=D2114

D3003=diode(1.0e16,3003,120,"EPI")
diodes['3003']=D3003

D3007=diode(2.5e15,3007,120,"EPI")
diodes['3007']=D3007

D3008=diode(1.5e15,3008,120,"EPI")
diodes['3008']=D3008

D3015=diode(1.8e15,3015,120,"EPI")
diodes['3015']=D3015

D6002=diode(5.0e14,6002,120,"DD")
D6002.ann_offset_error = 3
D6002.area=0.2595 #check this
diodes['6002']=D6002

#create fluence colours

def _make_colours():
    fluences = [d[1].rad for d in diodes.items()]
    fluences = list(set(fluences))
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

def print_all_diodes():
    for k in diodes.keys():
        print(k, diodes[k])





