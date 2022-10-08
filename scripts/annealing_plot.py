#!/usr/bin/env python3

import os
import styles
import matplotlib.pyplot as plt
from tools import convert60CTo0C, convert60CTom30C, convert60CTo15C, convert60Cto21C
import numpy as np
from pointset import pointSetsContainer, pointSet, loadAnnealings
from diodes import diodes
import pandas
import math
datadir=os.getenv("DATAOUTPATH")


os.system('mkdir -p '+datadir+'/annealing_plots')
outdir = datadir+'/annealing_plots/'

print('output plots in',outdir)

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


def combineDiodeStrings(ds):
    labels = []
    for d in ds:
        l=d.paperlabel(False)
        if not l in labels:
            labels.append(l)
            
    return " ".join([l for l in labels])+' ' +ds[0].radstr()
    
    

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

    

def cosmetics(x, ax, ylabel="-$U_{depl}$ [V]",legtitle=None, add_more_xaxes=False, **kwargs):
    
    
    plt.legend(title=legtitle, **kwargs)
    plt.xscale('log')
    plt.xlabel("time (60˚C) [min]")
    plt.ylabel(ylabel)
    
    if add_more_xaxes:
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


allfits=[]

#styles.setstyles(13)
styles.setstyles(15)

#120µm
###############################["3003_UL","3007_UL","3008_UL"]

ax = newplot()
print('3003_UL, 1e16')
xs = pointsets.addToPlot("NEff", ["3003_UL"],["UL, 120µm"],
                         add_rel_y_unc=0.1)
#allfits.append( ( addfitted(["3003_UL"]), r'1.0e16 neq/$cm^2$', 10.) )
cosmetics(xs,ax,"NEff $[1/cm^{3}]$")
plt.ylim([0.5e13,7.5e13])
plt.xlim([0.5,5000])
plt.savefig(outdir+"/allNEff_120f10_"+addout+".pdf")


ax = newplot()
print('3007_UL, 2.5e15')
xs = pointsets.addToPlot("NEff", ["3007_UL"],["UL, 120µm"],
                         add_rel_y_unc=0.1)
#allfits.append( ( addfitted(["3007_UL"]), r'2.5e15 neq/$cm^2$', 2.5) )
cosmetics(xs,ax,"NEff $[1/cm^{3}]$")
plt.ylim([0.5e13,7.5e13])
plt.xlim([0.5,5000])
plt.savefig(outdir+"/allNEff_120f2.5_"+addout+".pdf")


ax = newplot()
print('3008_UL, 1.5e15')
xs = pointsets.addToPlot("NEff", ["3008_UL"],["UL, 120µm"],
                         add_rel_y_unc=0.1)
#allfits.append( ( addfitted(["3008_UL"]), r'1.5e15 neq/$cm^2$', 10.) )
cosmetics(xs,ax,"NEff $[1/cm^{3}]$")
plt.ylim([0.5e13,7.5e13])
plt.xlim([0.5,5000])
plt.savefig(outdir+"/allNEff_120f1.5_"+addout+".pdf")



#exit()


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
plt.savefig(outdir+"/300_6inch"+addout+".pdf")


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
plt.savefig(outdir+"/6inch"+addout+".pdf")


ax = newplot()
pointsets.addToPlot("Udep", ["1002_UL","1003_UL","1102_UL"],[" UL"," UL"," UL"],
                    #colors=['tab:blue','tab:green','tab:orange']
                    colors='fluence',
                    labelmode='fluence'
                    )
xs = pointsets.addToPlot("Udep", ["1002_UR","1003_UR","1102_UR"],[" UR"," UR"," UR"],
                    #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    labelmode='fluence',
                    marker='x')
cosmetics(xs,ax)
plt.text(.3, 350, "300 µm FZ",fontdict=styles.fontdict())
plt.savefig(outdir+"/300"+addout+".pdf")


ax = newplot()
pointsets.addToPlot("Udep", ["2002_UL","2003_UL","2102_UL"],[" UL"," UL"," UL"],
                    #colors=['tab:blue','tab:green','tab:orange']
                    colors='fluence',
                    labelmode='fluence',
                    )
xs = pointsets.addToPlot("Udep", ["2002_UR","2003_UR","2102_UR"],[" UR"," UR"," UR"],
                    #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    labelmode='fluence',
                    marker='x')
cosmetics(xs,ax)
plt.text(.3, 190, "200 µm FZ",fontdict=styles.fontdict())
plt.savefig(outdir+"/200"+addout+".pdf")


ax = newplot()
xs = pointsets.addToPlot("Udep", ["3003_UL","3007_UL","3008_UL"],
                    #colors=['tab:blue','tab:green','tab:orange']
                    colors='fluence',
                    labelmode='fluence',
                    )

cosmetics(xs,ax)
plt.text(.3, 70, "120 µm EPI",fontdict=styles.fontdict())
plt.savefig(outdir+"/120"+addout+".pdf")


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
plt.savefig(outdir+"/allNEff"+addout+".pdf")



##### with fits






#make the fits:


    





















