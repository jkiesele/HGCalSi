#!/usr/bin/python3

import os
import matplotlib.pyplot as plt
import styles
import numpy as np
from tools import convert60Cto21C

from pointset import pointSetsContainer, pointSet, loadAnnealings

datadir=os.getenv("DATAOUTPATH")
outdir=os.getenv("DATAOUTPATH")+'/hamburg_plots/'
os.system('mkdir -p '+outdir)


'''
'diode': ps.diode,
                't': x,
                'terr':xerr,
                'y':y,
                'yerr':yerr
                '''

def symmetrize(errup,errdown):
    
    return max(abs(errup),abs(errdown))
    

def findClosestTime(dataarr, anntime):
    outI=[]
    outIerr=[]
    outD=[]
    for d in dataarr:
        xnp = np.array(d['t'])
        idx, good = pointSet.get_closest_point(xnp, anntime, allowed_difference=anntime*0.25)
        idx = int(idx)
        if good:
            outI.append(d['y'][idx])
            outIerr.append(d['yerr'][idx])
            outD.append(d['diode'])
            
    return outI,outD,outIerr
    
#pointSet.get_closest_point(xnp, pointat, allowed_difference)
    
#get all data
pointsets, kneepointset= loadAnnealings()

for U in (-100,-600, -800, "Udep"):
    voltstring = str(U)
    if not hasattr(U, "split"):
        voltstring+=" V"
    else:
        voltstring="U$_{dep}$" 
        
    data120 = pointsets.getXYsDiodes("IperVolume", ["3003_UL","3007_UL","3008_UL"]                              , current_at=U)
    
    data200 = pointsets.getXYsDiodes("IperVolume", ["2002_UL","2003_UL","2102_UL","2002_UR","2003_UR","2102_UR"], current_at=U)
    
    data300 = pointsets.getXYsDiodes("IperVolume", ["1002_UL","1003_UL","1102_UL","1002_UR","1003_UR","1102_UR"], current_at=U)
    
        
    for t in [73, 100, 150]: 
        minx=1e19
        maxx=0   
        for thicknessdata, desc in zip([data120,data200,data300],["120µm EPI", "200µm FZ", "300µm FZ"]):
        
            print(t)
            outI,outD,outIerr = findClosestTime(thicknessdata,t)
            xs = np.array([d.rad for d in outD])
            ys = np.squeeze(np.concatenate(outI,axis=0)) * 1e-2/1e-6 #from µm to cm
            
            yerrs = np.squeeze(np.concatenate(outIerr,axis=0))* 1e-2/1e-6 
            #print(t,desc,yerrs)
            plt.errorbar(xs, ys,  yerr=yerrs, xerr=0.1*xs,label=desc,
                         marker='x',linewidth=0,elinewidth=1.,markersize=2.)
            
            if np.min(xs)<minx:
                minx=np.min(xs)
            if maxx<np.max(xs):
                maxx=np.max(xs)
            #print(minx)
            #plt.plot(xs,ys, label=str(outD[0].thickness)+"µm",marker='x')
        
        def line(minx,maxx):
            return 7.675e-19*minx, 7.675e-19*maxx
        
        plt.legend()
        plt.xlabel("Fluence [neq/cm$^2$]", fontsize=12)
        plt.ylabel("I("+voltstring+")/Volume [A/cm$^3$] @ -20˚C", fontsize=12)
        
        print(minx)
        plt.plot([minx,maxx],line(minx,maxx))
        #plt.xscale('log')
        #plt.yscale('log')
        plt.tight_layout()
        plt.savefig(outdir+"ann_"+str(t)+"_"+str(U)+".pdf")
        plt.close()
        
        
    
    
    
    