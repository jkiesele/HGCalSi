#!/usr/bin/env python3

import os

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import styles
import numpy as np
from pointset import loadAnnealings
from base_tools import VarSetSet, VarSet, DNeff_calc, fitPoints

from scipy.optimize import fmin 
import scipy.odr as odr
import pickle
import styles
styles.setstyles()

from jax import numpy as jnp 

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

allFZ = ["2002_UL","2002_UR","1003_UL","1003_UR"]+\
            ["1002_UL","1002_UR"]+\
            ["2003_UL","2003_UR","1102_UL", "1102_UR"]+\
            ["2102_UL","2102_UR"]

allEPI = ["3008_UL"] +["3007_UL"]+["3003_UL"]
            

#samples = [allFZ]

yscaling = 1.

#dataFZ = pointsets.getXYsDiodes("NEff", 
#                                ["1102_UR","3007_UL","1002_UR","1003_UR",]#+
#                                #["1002_UL","1003_UL","1102_UL",]
#                              #["2003_UR","2002_UR","2102_UR","2002_UL","2003_UL","2102_UL"]
#                              )
#
stdvars = {
            'g_a': yscaling*1e-2/4.,
            'tau_a': 40.,
            'g_c': yscaling*1e-3/4.,
            'g_y': yscaling* 5e-2/4.,
            'tau_y': 5000.
        }

def mergePoints(pset):
    ref = pset[0]
    for i in range(1,len(pset)):
        for k in ['t','y','terr','yerr','t_oup','t_odown']:
            ref[k] = np.concatenate([ref[k],pset[i][k]],axis=0)
    return [ref]


def make_single_fit(x,y,yerr,globalvars,localvars,plotstr=None,plottitle=None):
    
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)
    
    #x *= 1.8
    #y *= 1.8
    #yerr *= 1.8
    
    fitfunc = DNeff_calc([ (str(k), stdvars[k]) for k in stdvars.keys() ], localvars)
    fp = fitPoints(x, y, yerr, fitfunc, direct_neighbour_corr=0.5)
    fplargey = fitPoints(x, y, yerr, fitfunc)
    fplargey.enlargeYErrs()#for first fit
    
    if plotstr is not None and False:
        fp.plot(fitfunc.start_vals())
        plt.xscale('log')
        plt.savefig(plotstr+'_pre.pdf')
        plt.close()
    
    m = Minuit(jax.jit(fplargey.chi2), fitfunc.start_vals(), grad=jax.grad(fplargey.chi2))
    m.limits = [(0,None) for _ in fitfunc.start_vals()]
    m.strategy=1
    m.migrad(iterate=10)
    startvals=np.array(m.values)
    #print(startvals)
    
    m = Minuit(jax.jit(fp.chi2), startvals, grad=jax.grad(fp.chi2))
    m.strategy=2
    o = m.migrad(iterate=10)
    o = m.minos()

    cov = np.array(m.covariance)
    merrs = np.array(m.errors)
    merrst = np.expand_dims(merrs,axis=0)
    merrstt = np.expand_dims(merrs,axis=1)
    merrstt = merrstt*merrst
    corr = cov/merrstt
    
    fvals=np.array(m.values)
    
    resdict = fitfunc.list_to_dict(m.values)  ##vvs.getVarList(True)
    errdict = fitfunc.list_to_dict(m.errors)
    for k in errdict.keys():
        resdict[k+'_std']=errdict[k]
        
    class fittedfunc(object):
        def __init__(self,infittedvals):
            self.fvals=infittedvals
        def eval(self,x):
            return fitfunc.eval(x, self.fvals)
        
    mint_std = []
    for i in range(-1,len(m.values)):
    #get minimum
        if i<0:
            minff=fittedfunc(fvals)
            minx = fmin(minff.eval,np.array([120.]))[0]
            resdict['mint']=minx
        else:
            mask = np.zeros_like(m.values)
            mask[i]=1.
            minff=fittedfunc(fvals+np.array(m.errors)*mask)
            minx = fmin(minff.eval,np.array([120.]))[0]
            mint_std.append(minx-resdict['mint'])
        #create uncertainties
    mint_std = np.array(mint_std)
    resdict['mint_std']=np.sqrt(np.sum( mint_std[np.newaxis,...]*corr*mint_std[...,np.newaxis] ))
    
    if plotstr is not None:
        fp.plot(fvals)
        plt.xscale('log')
        plt.legend()
        plt.title(plottitle)
        plt.savefig(plotstr+'.pdf')
        plt.close()
    
    return resdict, [fp, fvals]

    

def makeData():

    allres=[]
    for ip,s in enumerate(samples):
    
        globalvars = ['g_a', 'tau_a', 'g_c', 'g_y', 'tau_y']#['tau_a', 'tau_y']
        
        
        print(s)
        
        
        dataFZ = pointsets.getXYsDiodes("NEff", s)
        NC0s=[]
        for d in dataFZ:
            print(str(d['diode']))
            print('NC0:',d['diode'].NEff_norad())
            NC0s.append(d['diode'].NEff_norad())
            
        d = mergePoints(dataFZ)[0]
        
        #print(d)
        
        yerr = d['yerr']#remove additional dim
        if len(yerr.shape):
            yerr = yerr[:,0]
            
        print('data shapes',d['t'].shape, d['y'].shape, yerr.shape)
        
        localvars = {'phi': d['diode'].rad,'NC0' : d['diode'].NEff_norad() }
        resdict, v = make_single_fit(d['t'],d['y'],yerr,globalvars,localvars,
                                     plotstr=outdir+'/'+str(ip),plottitle = d['diode'].radstr())
        
        roup, _ = make_single_fit(d['t_oup'],d['y'],yerr,globalvars,localvars,
                                  plotstr=outdir+'/'+str(ip)+'_tup',plottitle = d['diode'].radstr())
        rodown, _ = make_single_fit(d['t_odown'],d['y'],yerr,globalvars,localvars,
                                     plotstr=outdir+'/'+str(ip)+'_tdown',plottitle = d['diode'].radstr())
        
        for k in roup.keys():
            resdict[k+"_oup"]=roup[k]
        for k in rodown.keys():
            resdict[k+"_odown"]=rodown[k]
        
        resdict['diode']=d['diode']
        
        allres.append(resdict)
        
        # debug plot
        
        
        
        
    
    with open(outdir+'/data.pkl','wb') as f:
        pickle.dump(allres,f)
    print('saved to',outdir+'/data.pkl')
    
   
makeData()
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
    ymax=0
    ymin=0
    
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
        
        yerrnew = np.where(np.abs(yerr)/3.>y,0.,yerr)
        if not np.all(yerrnew == yerr):
            print('\n\nWARNING SET SOME ERRORS TO ZERO!!!\n\n')
            yerr = yerrnew
        
        ax.errorbar(x,y,yerr,linewidth=0,marker='o',elinewidth=1,label=None,capsize=0,
                    color = col)
        
        
        #for xi,yi, txt in zip(x,y,[r['diode'] for r in allres]):
        #    ax.annotate(txt, (xi,yi))
        
    ax.set_ylabel(labeldict[var])
    ax.set_xlabel(r'Fluence [$\Phi_{eq} / cm^{-2}$]')
    ax.set_xscale('log')
    
    
        
    plt.legend() 
    #plt.ylim([0,None])
    plt.tight_layout()
    plt.savefig(outdir+'/'+var+'.pdf')
    print('saved',outdir+'/'+var+'.pdf')
    plt.close()







