#!/usr/bin/python3

import os
import matplotlib.pyplot as plt
import styles
import numpy as np
from tools import convert60CTo0C
import scipy.odr as odr

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
    

def findClosestTime(dataarr, anntime, allowederr):
    outI=[]
    outIerr=[]
    outD=[]
    for d in dataarr:
        xnp = np.array(d['t'])
        idx, good=-1, False
        try:
            idx, good = pointSet.get_closest_point(xnp, anntime, allowed_difference=allowederr*anntime)
        except:
            continue
        idx = int(idx)
        if good:
            outI.append(d['y'][idx])
            outIerr.append(d['yerr'][idx])
            outD.append(d['diode'])
        else:
            print('failed by', (d['t'][idx]-anntime)/anntime,'vs',allowederr,'at',anntime)
            
    return outI,outD,outIerr
    
#pointSet.get_closest_point(xnp, pointat, allowed_difference)
    
#get all data
pointsets, kneepointset= loadAnnealings()

fitmode='onlyFZ'

for fitmode in ( 'onlyFZ', 'all'):
    
    U="Udep"
    allalphas=[]
    allalphaerrs=[]
    allts=[]
    allterr=[]
    
    voltstring = str(U)
    if not hasattr(U, "split"):
        voltstring+=" V"
    else:
        voltstring="U$_{dep}$" 
        
    data120 = pointsets.getXYsDiodes("IperVolume", ["3003_UL","3007_UL","3008_UL"]                              , current_at=U)
    
    data200 = pointsets.getXYsDiodes("IperVolume", ["2002_UL","2003_UL","2102_UL","2002_UR","2003_UR","2102_UR"], current_at=U)
    
    data300 = pointsets.getXYsDiodes("IperVolume", ["1002_UL","1003_UL","1102_UL","1002_UR","1003_UR","1102_UR"], current_at=U)
    
    allterr=[]    
    for t, terr in zip([74, 103, 151, 245, 380, 640],#, 1444],
                       [0.2,0.15,0.1, 0.05, 0.06, 0.05]#, 0.7]
                       ): 
        
        minx=1e19
        maxx=0
        xys=[]   
        for thicknessdata, desc in zip([data120,data200,data300],["120µm EPI", "200µm FZ", "300µm FZ"]):
        
            #print(t)
            try:
                outI,outD,outIerr = findClosestTime(thicknessdata,t,terr)
            except Exception as e:
                print('time ',t , 'failed', desc, )
                print(thicknessdata)
                raise e
            if len(outI) < 1:
                continue
            xs = np.array([d.rad for d in outD])
            ys = np.squeeze(np.concatenate(outI,axis=0)) * 1e-2/1e-6 #from µm to cm
            
            yerrs = np.squeeze(np.concatenate(outIerr,axis=0))* 1e-2/1e-6 
            #print(t,desc,yerrs)
            plt.errorbar(xs, ys,  yerr=yerrs, xerr=0.1*xs,label=desc,
                         marker='x',linewidth=0,elinewidth=1.,markersize=2.)
            
            xys.append([xs,ys,0.1*xs,yerrs])
            if np.min(xs)<minx:
                minx=np.min(xs)
            if maxx<np.max(xs):
                maxx=np.max(xs)
            print('success', t, desc, voltstring)
            #print(minx)
            #plt.plot(xs,ys, label=str(outD[0].thickness)+"µm",marker='x')
        
        if not len(xys):
            print('full set at time',t,'failed for ',voltstring)
            continue
        
        
        def EstebansLine(minx,maxx):
            return 7.675e-19*minx, 7.675e-19*maxx
        
        def linear(a,x):
            return a*1e-19*x
        
        plt.xlabel("Fluence [neq/cm$^2$]", fontsize=12)
        plt.ylabel("I("+voltstring+")/Volume [A/cm$^3$] @ -20˚C", fontsize=12)
        
        #print(minx)
        plt.plot([minx,maxx],EstebansLine(minx,maxx),label='TDR DD, 80 min @ 60˚C',linestyle='--')
        
        #create fit data
        fitdata=None
        
        if U == "Udep":
            for i in range(len(xys)):
                if fitmode == 'onlyFZ' and not i:
                    continue
                #print(xys[i])
                if fitdata is None:
                    fitdata = [np.expand_dims(a,axis=1) for a in xys[i]]
                else:
                    fitdata = [np.concatenate([fitdata[j],
                                           np.expand_dims(xys[i][j],axis=1)],axis=0) 
                                           for j in range(len(xys[i]))]
            
            #print(fitdata)
            fitdata = [np.squeeze(a,axis=1) for a in fitdata]
            #fit linear
            m = odr.Model(linear)
            mydata = odr.RealData(fitdata[0], fitdata[1], sx=fitdata[2], sy=fitdata[3])
            myodr = odr.ODR(mydata, m, beta0=[7.])
            out=myodr.run()
            print(out.beta)
            print(out.sd_beta)
            
            alpha = round(out.beta[0],2)
            alphaerr = round(out.sd_beta[0],2)
            
            if alphaerr:
                allalphas.append(alpha)
                allalphaerrs.append(alphaerr)
                allts.append(t)
                allterr.append(t*terr)
            
            
        
            plt.plot([minx,maxx],[linear(out.beta,minx),linear(out.beta,maxx)],
                 label=r"$\alpha$(fit)="+str(alpha)+"$\pm$"+str(alphaerr)+"$10^{-19}$A/cm")

        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.title(r''+str(t)+'$\pm$'+str(round(t*terr,0))+' min annealing')
        plt.tight_layout()
        plt.savefig(outdir+"ann_"+str(t)+"_"+str(U)+"_"+fitmode+".pdf")
        plt.close()
    
    print(allalphas)    
    if len(allalphas):
        print(U)
        plt.errorbar(allts, allalphas,  yerr=allalphaerrs, xerr=allterr,
                         marker='o',linewidth=0,elinewidth=2.)#,markersize=2.)
        
        plt.xlabel('t @ 60˚C [min]')
        plt.ylabel(r'$\alpha$ [$10^{-19}$ A/cm]')
        def minat60ToDaysAt0(t):
            return convert60CTo0C(t/(60*24.))
        styles.addModifiedXAxis(allts, minat60ToDaysAt0, "time (0˚C) [d]")
        plt.ylim([5,7.5])
        #plt.xscale('log')
    
        plt.tight_layout()
        
        plt.savefig(outdir+"ann_alpha_"+str(U)+"_"+fitmode+".pdf")
        plt.close()
    
    
    
    