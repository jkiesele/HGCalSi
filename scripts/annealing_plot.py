#!/usr/bin/python3

import os
import styles
import matplotlib.pyplot as plt
from tools import convert60CTo0C, convert60CTom30C, convert60CTo15C, convert60Cto21C
import numpy as np
from pointset import pointSetsContainer, pointSet, loadAnnealings

datadir=os.getenv("DATAOUTPATH")


def minat60ToDaysAt0(t):
    return convert60CTo0C(t/(60*24.))

def minat60ToMonthsAtm30(t):
    return convert60CTom30C(t) /(60.*24.*30.4166)

def minat60ToDaysAt15(t):
    return convert60CTo15C(t)/(60*24.)

def minat60ToDaysAt21(t):
    return convert60Cto21C(t)/(60*24.)

def identity(x):
    return x



addminus30=False


def newplot():
    plt.close()
    height = 5.5
    if addminus30:
        height+=1
    fig,ax = plt.subplots(figsize=(6, height))
    return ax

addout=""
if addminus30:
    addout +="withm30"
#for func, altxlabel, addout in zip(
#    [minat60ToDaysAt21, minat60ToDaysAt15, minat60ToDaysAt0, minat60ToMonthsAtm30],
#    ["time (21˚C) [d]" , "time (15˚C) [d]" , "time (0˚C) [d]", "time (-30˚C) [months]"],
#    ["_21C",             "_15C",             "_0C",            "_-30C"]
#    ):

    

def cosmetics(x, ax, ylabel="-$U_{depl}$ [V]"):
    
    
    plt.legend()
    plt.xscale('log')
    plt.xlabel("time (60˚C) [min]", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    sax = ax.secondary_xaxis('top', functions=(minat60ToDaysAt21, identity))
    sax.set_xlabel("time (21˚C) [d]")
    sax = ax.secondary_xaxis(1.2, functions=(minat60ToDaysAt0, identity))
    sax.set_xlabel("time (0˚C) [d]")
    if addminus30:
        sax = ax.secondary_xaxis(1.4, functions=(minat60ToMonthsAtm30, identity))
        sax.set_xlabel("time (-30˚C) [months]")
    
    plt.xscale('log')
    
    plt.tight_layout()
    

pointsets, kneepointset= loadAnnealings()





'''
["3007_UL","2102_UL","2102_UR",
             "3008_UL","2003_UL","2003_UR","1102_UL", "1102_UR",
             "2002_UL","2002_UR","1003_UL","1003_UR",
             "3003_UL"]:
'''


from fitting import AnnealingFitter




def addfitted(dstr):

    fitter = AnnealingFitter()
    xyerssdatas = pointsets.getXYsDiodes("NEff",dstr)
    '''
    'diode': ps.diode(),
    't': x,
    'terr':xerrs,
    'y':y,
    'yerr':yerrs
    '''
    x = np.concatenate([d['t'] for d in  xyerssdatas],axis=0)
    y = np.concatenate([d['y'] for d in  xyerssdatas],axis=0)
    xerr = np.concatenate([d['terr'] for d in  xyerssdatas],axis=0)
    yerr = np.concatenate([np.squeeze(d['yerr']) for d in  xyerssdatas],axis=0)
    
    yerr = np.sign(yerr)*np.sqrt(yerr**2 + (0.1*y)**2) # add 10% fluence uncertainty
    
    print(dstr)
    fitter.fit(x,y,xerr,yerr)
    
    min = np.min(x)
    max = np.max(x)
    newx = np.logspace(np.log(min)/4.,np.log(max)/np.log(10))
    
    plt.plot(newx, fitter.DNeff(newx), label="fit")
    return fitter



allfits=[]


ax = newplot()
xs = pointsets.addToPlot("NEff", ["2002_UL","2002_UR","1003_UL","1003_UR"],["UL, 200µm","UR, 200µm","UL, 300µm","UR, 300µm"],
                         add_rel_y_unc=0.1)
allfits.append( (addfitted(["2002_UL","2002_UR","1003_UL","1003_UR"]), r'1.0e15 neq/$cm^2$', 1.0) )
cosmetics(xs,ax,"NEff $[1/cm^{3}]$")
plt.ylim([0.5e13,4.5e13])
plt.xlim([0.5,5000])
plt.savefig(datadir+"/annealing_plots/allNEff_f1.0_"+addout+".pdf")


ax = newplot()
xs = pointsets.addToPlot("NEff", ["2003_UL","2003_UR","1102_UL", "1102_UR"],["UL, 200µm","UR, 200µm","UL, 300µm","UR, 300µm"],
                         add_rel_y_unc=0.1)
allfits.append( ( addfitted(["2003_UL","2003_UR","1102_UL", "1102_UR"]), r'1.5e15 neq/$cm^2$', 1.5) )
cosmetics(xs,ax,"NEff $[1/cm^{3}]$")
plt.ylim([0.5e13,4.5e13])
plt.xlim([0.5,5000])
plt.savefig(datadir+"/annealing_plots/allNEff_f1.5_"+addout+".pdf")


ax = newplot()
xs = pointsets.addToPlot("NEff", ["2102_UL","2102_UR"],["UL, 200µm","UR, 200µm"],
                         add_rel_y_unc=0.1)
allfits.append( ( addfitted(["2102_UL","2102_UR"]), r'2.5e15 neq/$cm^2$', 2.5) )
cosmetics(xs,ax,"NEff $[1/cm^{3}]$")
plt.ylim([0.5e13,4.5e13])
plt.xlim([0.5,5000])
plt.savefig(datadir+"/annealing_plots/allNEff_f2.5_"+addout+".pdf")

ax = newplot()
xs = pointsets.addToPlot("NEff", ["6002_6in"],["DD"],
                         add_rel_y_unc=0.1)
addfitted(["6002_6in"])
cosmetics(xs,ax,"NEff $[1/cm^{3}]$")
plt.ylim([0.5e13,4.5e13])
plt.xlim([0.5,5000])
plt.savefig(datadir+"/annealing_plots/allNEff_6inchDD_"+addout+".pdf")



ax = newplot()
for f in allfits:
    
    newx = np.logspace(np.log(1)/np.log(10),np.log(3000)/np.log(10))
    plt.plot(newx, (f[0].DNeff(newx) - f[0].N_c)/(allfits[0][0].DNeff(newx) - allfits[0][0].N_c), label=f[1])
    
cosmetics(xs,ax,"∆NEff (Fluence) / ∆NEff (1.0 neq/$cm^2$) ")
#plt.ylim([0.5e13,4.5e13])
plt.xlim([0.5,5000])
plt.savefig(datadir+"/annealing_plots/allNEff_fits_"+addout+".pdf")

alla = []

allNC = []
allfluence=[]
for f in allfits:
    alla.append(f[0].a)
    allNC.append(f[0].N_c)
    allfluence.append(f[2])

ax = newplot()
plt.plot(allfluence,alla)
plt.xlabel(r"Fluence [$10^{15}$ neq/$cm^2$]")
plt.ylabel(r"$N_a \cdot g_A$")
plt.savefig(datadir+"/annealing_plots/allNEff_fits_NAgA"+addout+".pdf")
#exit()

ax = newplot()
plt.plot(allfluence,allNC)
plt.xlabel(r"Fluence [$10^{15}$ neq/$cm^2$]")
plt.ylabel(r"$N_C$")
plt.savefig(datadir+"/annealing_plots/allNEff_fits_NC"+addout+".pdf")


ax = newplot()
pointsets.addToPlot("Udep",["6002_6in"],
                    colors='fluence',
                    #colors=['tab:red','tab:purple'],
                    marker='x')
kneepointset.addToPlot("Udep",["6002_6in"],[" first kink"],
                       #colors=['tab:red','tab:purple']
                    colors='fluence',
                       )
pointsets.addToPlot("Udep",["1002_UR", "1003_UR", "1102_UR"],[" UR"," UR"," UR"],
                    #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x')


xs = pointsets.addToPlot("Udep",["1002_UL", "1003_UL", "1102_UL"],
                    #colors=['tab:blue','tab:green','tab:orange']
                    colors='fluence',
                    )
cosmetics(xs,ax)
plt.savefig(datadir+"/annealing_plots/300_6inch"+addout+".pdf")


ax = newplot()
pointsets.addToPlot("Udep",["6002_6in"],[" second kink"],
                    #colors=['tab:red','tab:purple'],
                    colors='fluence',
                    marker='x')
xs = kneepointset.addToPlot("Udep",["6002_6in"],[" first kink"],
                       #colors=['tab:red','tab:purple']
                    colors='fluence',
                       )
cosmetics(xs,ax)
plt.savefig(datadir+"/annealing_plots/6inch"+addout+".pdf")


ax = newplot()
pointsets.addToPlot("Udep", ["1002_UL","1003_UL","1102_UL"],
                    #colors=['tab:blue','tab:green','tab:orange']
                    colors='fluence',
                    )
xs = pointsets.addToPlot("Udep", ["1002_UR","1003_UR","1102_UR"],[" UR"," UR"," UR"],
                    #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x')
cosmetics(xs,ax)
plt.savefig(datadir+"/annealing_plots/300"+addout+".pdf")


ax = newplot()
pointsets.addToPlot("Udep", ["2002_UL","2003_UL","2102_UL"],
                    #colors=['tab:blue','tab:green','tab:orange']
                    colors='fluence',
                    )
xs = pointsets.addToPlot("Udep", ["2002_UR","2003_UR","2102_UR"],[" UR"," UR"," UR"],
                    #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x')
cosmetics(xs,ax)
plt.savefig(datadir+"/annealing_plots/200"+addout+".pdf")


ax = newplot()
xs = pointsets.addToPlot("Udep", ["3003_UL","3007_UL","3008_UL"],
                    #colors=['tab:blue','tab:green','tab:orange']
                    colors='fluence',
                    )

cosmetics(xs,ax)
plt.savefig(datadir+"/annealing_plots/120"+addout+".pdf")


newplot()
xs = pointsets.addToPlot("NEff", ["3003_UL","3007_UL","3008_UL"],
                         colors='fluence',
                         marker='o')
xs = pointsets.addToPlot("NEff", ["2002_UL","2003_UL","2102_UL"],
                         colors='fluence',
                         marker='x')
xs = pointsets.addToPlot("NEff", ["1002_UR","1003_UR","1102_UR"],
                         colors='fluence',
                         marker='+')

cosmetics(xs,ax,"NEff $[1/cm^{3}]$")
plt.savefig(datadir+"/annealing_plots/allNEff"+addout+".pdf")



##### with fits






#make the fits:


    





















