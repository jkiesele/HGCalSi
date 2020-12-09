
from fileIO import fileReader
from plotting import curvePlotter
import os 
from fitting import DepletionFitter
from matplotlib import pyplot as plt
import numpy as np
from tools import getDepletionVoltage
from fitting import fittedIV

class IvSet(object):
    def __init__(self):
        self.ivs=[]
        
    def append(self, iv):
        self.ivs.append(iv)
        
    def eval(self, x):
        if not len(x) ==  len(self.ivs):
            raise ValueError("valtage array must have same size as IV array")
        return np.array([self.ivs[i].eval(x[i])[0] for i in range(len(x))], dtype='float')

def getFullSet(diodestr, debugall=False, time_offset=10, debugdir=None):

    globalpath="/Users/jkiesele/cern_afs/eos_hgsensor_testres/Results_SSD/CVIV/Diode_TS/"
    
    x=[]
    y=[]
    ivs=IvSet()
    
    const_cap=None
    
    if diodestr=="1002":
        const_cap=1.1975328065907755e+22
    elif diodestr=="1003":
        const_cap=1.1944930287989327e+22
    elif diodestr=="1102":
        const_cap=1.1859109408633843e+22
    elif diodestr=="2003" or diodestr=="2002" or diodestr=="2102":
        const_cap=5.500145558851301e+21
    
    time=0
    v = getDepletionVoltage(globalpath+diodestr+"_UL_diode_big_no_ann/*.cv",min_x=-850,
                            const_cap=const_cap,
                             rising=1.1,
                             debug=debugall,
                             debugfile=debugdir+diodestr+'_'+str(time))
    x.append(time+time_offset)
    y.append(v)
    ivfit = fittedIV(globalpath,diodestr+"_UL_diode_big_no_ann/*.iv",debug=debugall,
                     debugfile=debugdir+diodestr+'_'+str(time)+'iv',min_x=-850)
    ivs.append(ivfit)
    
    ########
    
    for time in [73,103,153,247,378,645,1352]:
        thiscap = None
        if time >= 378:
            thiscap = const_cap
        if time < 100:
            thiscap = const_cap
            
            
        v = getDepletionVoltage(globalpath+diodestr+"_UL_diode_big_ann_"+str(time)+"min/*.cv",min_x=-900,
                            const_cap=thiscap,
                             rising=1.1,
                             debug=debugall,debugfile=debugdir+diodestr+'_'+str(time))
        x.append(time+time_offset)
        y.append(v)
    
        print('Depletion voltage at',time, diodestr, 'is',v)
        
        ivfit = fittedIV(globalpath,diodestr+"_UL_diode_big_ann_"+str(time)+"min/*.iv",
                         debug=debugall,debugfile=debugdir+diodestr+'_'+str(time)+'iv',min_x=-900)
        
        ifix = ivfit.eval([-600])
        print("leakage at -600V",ifix,'\n')
        
        ivs.append(ivfit)
    
        

    
    
    return np.array(x),np.array(y), ivs
   