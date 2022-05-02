#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import styles
import numpy as np
from pointset import loadAnnealings
from base_tools import VarSetSet, VarSet

import scipy.odr as odr
import pickle
import styles
styles.setstyles()

datadir=os.getenv("DATAOUTPATH")
outdir=os.getenv("DATAOUTPATH")+'/hamburg_model/'
os.system('mkdir -p '+outdir)

pointsets, _= loadAnnealings()

from iminuit import Minuit

samplesperfluence = ["1002_UL","1002_UR"] +\
 ["2002_UL","2002_UR","1003_UL","1003_UR"] +\
   ["2003_UL","2003_UR","1102_UL", "1102_UR"] +\
    ["2102_UL","2102_UR"] +\
    ["3008_UL"] +\
    ["3007_UL"] +\
    ["3003_UL"] 

samples  = [
            ["2002_UL","2002_UR","1003_UL","1003_UR"],
            ["1002_UL","1002_UR"],
            ["2003_UL","2003_UR","1102_UL", "1102_UR"],
            ["2102_UL","2102_UR"],
            ["3008_UL"],
            ["3007_UL"],
            ["3003_UL"]
            ]

#dataFZ = pointsets.getXYsDiodes("NEff", 
#                                ["1102_UR","3007_UL","1002_UR","1003_UR",]#+
#                                #["1002_UL","1003_UL","1102_UL",]
#                              #["2003_UR","2002_UR","2102_UR","2002_UL","2003_UL","2102_UL"]
#                              )
#
stdvars = {
            'g_a': 1e-2,
            'tau_a': 40.,
            'g_c': 1e-3,
            'g_y': 1e-1,
            'tau_y': 2000.
        }
def mergePoints(pset):
    ref = pset[0]
    for i in range(1,len(pset)):
        for k in ['t','y','terr','yerr','t_oup','t_odown']:
            ref[k] = np.concatenate([ref[k],pset[i][k]],axis=0)
    return [ref]


def makeData():

    allres=[]
    for s in samples:
    
        globalvars = []#['tau_a', 'tau_y']
        
        
        print(s)
        
        
        dataFZ = pointsets.getXYsDiodes("NEff", s)
        NC0s=[]
        for d in dataFZ:
            print(str(d['diode']))
            print('NC0:',d['diode'].NEff_norad())
            NC0s.append(d['diode'].NEff_norad())
            
        dataFZ = mergePoints(dataFZ)
        
        def makefit(indicator = "",debugplot=False):
            vvs = VarSetSet(globalvars)
            
            for d in dataFZ:
                print(str(d['diode']))
                print('NC0:',d['diode'].NEff_norad())
                
                if np.any(d['yerr']==0):
                    print('at least one y error is zero')
                
                x = d['t']
                if indicator == "oup":
                    x = d['t_oup']
                elif indicator == "odown":
                    x = d['t_odown']
                
                vvs.addVarSet( VarSet(x, d['y'], d['yerr'], None, stdvars, {'phi': d['diode'].rad,
                                                                                 'NC0' : d['diode'].NEff_norad()
                                                                                 }) )
                #vvs.addVarSet( VarSet(d['t'], d['y'], d['yerr'], d['terr'], stdvars) )
            
            
        
            import jax
            jax.config.update("jax_enable_x64", True)
            vvs.createStartScale()#important
            m = Minuit(jax.jit(vvs.allSetChi2), [1. for _ in vvs.getVarList()], grad=jax.grad(vvs.allSetChi2))
            m.limits = [(0,None) for _ in vvs.getVarList()]
            #m.parameters=vvs.getVarList(True).keys()
            m.strategy=1
            o = m.migrad(iterate=10)
            print(o)
            print(vvs.allSetYErrs())
            print(m.minos())
            vvs.applyVarList(vvs.scaleFittedVars(m.values))
            cov = np.array(m.covariance)
            merrs = np.array(m.errors)
            merrst = np.expand_dims(merrs,axis=0)
            merrstt = np.expand_dims(merrs,axis=1)
            merrstt = merrstt*merrst
            corr = cov/merrstt
            
            #print(o)
            resdict = vvs.getVarList(True)
            print(vvs.getVarList(True))
            print(vvs.getVarList())
            print(vvs.scaleFittedVars(m.errors))
            
            if debugplot:
                for v in vvs.varsets:
                    print(indicator)
                    v.debugPlot()
                    v.debugPlotEval()
                    xs = np.arange(np.min(v.xs),np.max(v.xs),2)
                    plt.plot(xs,v.NC(xs))
                    plt.plot(xs,v.NA(xs))
                    plt.plot(xs,v.NY(xs))
                    for n in NC0s:
                        plt.plot(xs,xs*0.+n,linestyle='-.')
                    plt.xscale('log')
                    plt.show()
                    plt.close()
                  
            from scipy.optimize import fmin  
            mint_std = []
            for i in range(-1,len(m.values)):
            #get minimum
                if i<0:
                    v=vvs.varsets[0]#same by definition
                    minx = fmin(v.eval,np.array([120.]))[0]
                    resdict['mint']=minx
                else:
                    mask = np.zeros_like(m.values)
                    mask[i]=1.
                    vvss = vvs
                    vvss.applyVarList(vvs.scaleFittedVars(m.values)+vvs.scaleFittedVars(m.errors)*mask)
                    v=vvss.varsets[0]
                    minx = fmin(v.eval,np.array([120.]))[0]
                    mint_std.append(minx-resdict['mint'])
                #create uncertainties
            mint_std = np.array(mint_std)
            
                
                
            resdict['mint_std']=np.sqrt(np.sum( mint_std[np.newaxis,...]*corr*mint_std[...,np.newaxis] ))
            print('min_t',resdict['mint'] , '+-',resdict['mint_std'])
            resdict['diode']=dataFZ[0]['diode']
            if len(indicator):
                return resdict
            
            for k,e in zip(vvs.getVarList(True).keys(),vvs.scaleFittedVars(m.errors)):
                resdict[k+'_std']=e
                
            
            return resdict
        
        resdict = makefit()
        roup = makefit("oup")
        for k in roup.keys():
            resdict[k+"_oup"]=roup[k]
        rodown = makefit("odown")
        for k in rodown.keys():
            resdict[k+"_odown"]=rodown[k]
            
        #print(resdict.keys())
        #exit()
        
        allres.append(resdict)
        
        #for v in vvs.varsets:
        #    v.debugPlot()
        #    v.debugPlotEval()
        #    xs = np.arange(np.min(v.xs),np.ma#x(v.xs),2)
        #    plt.plot(xs,v.NC(xs))
        #    plt.plot(xs,v.NA(xs))
        #    plt.plot(xs,v.NY(xs))
        #    for n in NC0s:
        #        plt.plot(xs,xs*0.+n,linestyle='-.')
        #    plt.xscale('log')
        #    #plt.show()
        #    plt.close()
        #exit()
        
    with open(outdir+'/data.pkl','wb') as f:
        pickle.dump(allres,f)
    print('saved to',outdir+'/data.pkl')
   

#makeData()
#exit()

#read dump

with open(outdir+'/data.pkl','rb') as f:
    allres=pickle.load(f)
        
varkeys = [k for k in stdvars.keys()] + ['mint']
#print(allres)

labeldict={
    'g_a':r'$g_a$ [$\Phi_{eq}^{-1} cm^{-1}$]',
    'tau_a': r'$\tau_a$ [min]',
    #'N_C+N_{Eff,0}': r'[$cm^{-3}$]',
    #'N_C': r'[$cm^{-3}$]',
    'g_c': r'$g_c$ [$\Phi_{eq}^{-1} cm^{-1}$]',
    'g_y': r'$g_y$ [$\Phi_{eq}^{-1} cm^{-1}$]',
    'tau_y': r'$\tau_y$ [min]',
    'mint': r'$t(U_{dep,min})$ [min]',
    #'U_{dep}^{min}': r'[V]',
    }

#exit()
#make plot

def getYErrs(a,b):
    #a, b, are already nominal subtracted
    a = np.expand_dims(a,axis=0)
    b = np.expand_dims(b,axis=0)
    ab = np.concatenate([a,b],axis=0)
    down = np.min(ab,axis=0,keepdims=True)
    up = np.max(ab,axis=0,keepdims=True)
    return np.abs(np.concatenate([down,up],axis=0))# 2 x points

for var in varkeys:
    
    fig, ax = plt.subplots()
    #ax.set_title(var)
    
    for sel in ["FZ","EPI"]:
        
        col = 'tab:blue'
        mult=0.98
        if sel == "EPI":
            col = 'tab:orange'
            mult=1.02
    
        x = mult*np.array([r['diode'].rad  for r in allres if r['diode'].material_str()==sel])
        y = [r[var] for r in allres if r['diode'].material_str()==sel]
        yerr = [r[var+'_std'] for r in allres if r['diode'].material_str()==sel]
        youp = [r[var+'_oup'] for r in allres if r['diode'].material_str()==sel]
        yodown = [r[var+'_odown'] for r in allres if r['diode'].material_str()==sel]
    
        ax.errorbar(x,y,yerr,linewidth=0,marker='o',elinewidth=1,label=sel,capsize=2)
        
        downvar = np.array(y)-np.array(youp)
        
        upvar = np.array(y)-np.array(yodown)
        updown = getYErrs(upvar,downvar)
        #print(np.array(y))
        #print(updown/np.array(y)*100.)
        
        yerr = np.sqrt(np.array(yerr)[np.newaxis,...]**2 + (updown)**2)
        
        ax.errorbar(x,y,yerr,linewidth=0,marker='o',elinewidth=1,label=None,capsize=0,
                    color = col)
        
        #for xi,yi, txt in zip(x,y,[r['diode'] for r in allres]):
        #    ax.annotate(txt, (xi,yi))
        
    ax.set_ylabel(labeldict[var])
    ax.set_xlabel(r'Fluence [$\Phi_{eq} / cm^{-2}$]')
    ax.set_xscale('log')
    
    
        
    plt.legend() 
    plt.tight_layout()
    plt.savefig(outdir+'/'+var+'.pdf')
    print('saved',outdir+'/'+var+'.pdf')
    plt.close()







