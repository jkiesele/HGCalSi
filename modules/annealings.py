
from fileIO import fileReader
from plotting import curvePlotter
import os 
from fitting import DepletionFitter
from matplotlib import pyplot as plt
import numpy as np
from tools import getDepletionVoltage
from fitting import fittedIV
from diodes import diodes

#dict of annealing steps








class IvSet(object):
    def __init__(self):
        self.ivs=[]
        
    def append(self, iv):
        self.ivs.append(iv)
        
    def eval(self, x):
        if not len(x) ==  len(self.ivs):
            raise ValueError("valtage array must have same size as IV array")
        return np.array([self.ivs[i].eval(x[i])[0] for i in range(len(x))], dtype='float')

def getFullSet(diodestr, debugall=False,  debugdir=None, useIVGP=False, times = [10, 30, 73,103,153,247,378,645,1352,2919]):

    globalpath=os.getenv("DATAPATH")+'/'
    
    x=[]
    y=[]
    yerr=[]
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
    
    time_offset=diodes[diodestr].ann_offset
    time=0
    
    v = getDepletionVoltage(globalpath+diodestr+"_UL_diode_big_no_ann/*.cv",min_x=-850,
                            const_cap=const_cap,
                             rising=1.1,
                             debug=debugall,
                             debugfile=debugdir+diodestr+'_'+str(time))
    x.append(time+time_offset)
    y.append(v)
    yerr.append(0.*v*0.05)
    ivfit = fittedIV(globalpath,diodestr+"_UL_diode_big_no_ann/*.iv",debug=debugall,
                     debugfile=debugdir+diodestr+'_'+str(time)+'iv',min_x=-850,
                     useGP=useIVGP)
    ivs.append(ivfit)
    
    ########
    
    for time in times:
        
        identifier="_UL_diode_big_ann_"
        if time < 73 or time==74:
            identifier="_UR_2D_diode_big_ann_"
        if diodestr[0] == '3':
            identifier="_UL_3D_diode_big_ann_"
        
        thiscap = None
        if time >= 378:
            thiscap = const_cap
        if time < 74 and diodestr[0] == '1':
            thiscap = const_cap
            if diodestr=='1102' and (time == 30):
                thiscap=1.0359109408633843e+22
            
        rising=1.1
        if time==2919 and (diodestr=="2003" or diodestr=="2102" or diodestr=="1003"):
            rising=0.0
        if time<=646 and (diodestr=="2002" or diodestr=="2002"):
            rising=1.5
            
            
        v = getDepletionVoltage(globalpath+diodestr+identifier+str(time)+"min/*.cv",min_x=-900,
                            const_cap=thiscap,
                             rising=rising,
                             debug=debugall,debugfile=debugdir+diodestr+'_'+str(time))
        x.append(time+time_offset)
        y.append(v)
        if thiscap is not None:
            yerr.append(0.*v*0.05)
        else:
            yerr.append(0.*v*0.025)
    
        print('Depletion voltage at',time, diodestr, 'is',v)
        
        ivfit = fittedIV(globalpath,diodestr+identifier+str(time)+"min/*.iv",
                         debug=debugall,debugfile=debugdir+diodestr+'_'+str(time)+'iv',min_x=-900,
                         useGP=useIVGP)
        
        ifix = ivfit.eval([-600])
        print("leakage at -600V",ifix,'\n')
        
        ivs.append(ivfit)
    
        
    x = np.array(x)
    xerr = x * 0.025 + 1 #1 minute unc from irradiation
    yerr = np.array(yerr)
    return np.array(x),np.array(y), ivs, xerr, yerr
   