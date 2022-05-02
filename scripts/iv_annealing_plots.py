#!/usr/bin/python3

import os
import matplotlib.pyplot as plt
import styles

from tools import convert60CTo0C, convert60CTom30C, convert60CTo15C, convert60Cto21C

from pointset import pointSetsContainer, pointSet, loadAnnealings

datadir=os.getenv("DATAOUTPATH")
outdir=os.getenv("DATAOUTPATH")+'/annealing_plots/'
os.system('mkdir -p '+outdir)

addminus30=False

styles.setstyles(13)

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

def newplot():
    plt.close()
    height = 5.5
    if addminus30:
        height+=1
    fig,ax = plt.subplots(figsize=(6, height))
    return ax

print(outdir)

def setAxisIV(ax,ylabel="-$I$ [A]",**legkwargs):
    
    plt.legend(**legkwargs)
    plt.legend()
    plt.xlim(0.2,2500)
    plt.xscale('log')
    plt.xlabel("time (60˚C) [min]")
    plt.ylabel(ylabel)
    
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

itimesthicknessylim=[0,0.009]

for U in (-600, -800, "Udep"):
    
    voltstring = str(U)
    if not hasattr(U, "split"):
        voltstring+=" V"
    else:
        voltstring="U$_{dep}$"                  
    
    ax = newplot()
    xs = pointsets.addToPlot("I", ["3003_UL","3007_UL","3008_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    labelmode='fluence',
                        current_at=U)
    setAxisIV(ax,"I("+voltstring+")[A]")
    plt.ylim([0,6e-5])
    plt.text(.3, 5.5e-5, "120 µm EPI",fontdict=styles.fontdict())
    plt.tight_layout()
    plt.savefig(outdir+'120_'+str(U)+'.pdf')
    
    ax = newplot()
    xs = pointsets.addToPlot("IperFluence", ["3003_UL","3007_UL","3008_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    setAxisIV(ax, "I("+voltstring+")[A] / Fluence [neq/cm$^2$]")
    plt.ylim([0,0.8e-20])
    plt.tight_layout()
    plt.savefig(outdir+'120_IperFluence_'+str(U)+'.pdf')
    
    
    ax = newplot()
    xs = pointsets.addToPlot("IperThickness", ["3003_UL","3007_UL","3008_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    setAxisIV(ax, "I("+voltstring+")[A] / thickness [µm]")
    plt.ylim([0,6e-5/120.])
    plt.savefig(outdir+'120_IperThickness_'+str(U)+'.pdf')
    
    ax = newplot()
    xs = pointsets.addToPlot("ItimesThickness", ["3003_UL","3007_UL","3008_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    setAxisIV(ax, "I("+voltstring+")[A] * thickness [µm]")
    plt.ylim(itimesthicknessylim)
    plt.tight_layout()
    plt.savefig(outdir+'120_ItimesThickness_'+str(U)+'.pdf')
    
    ###########
    
    ax = newplot()
    pointsets.addToPlot("I", ["2002_UL","2003_UL","2102_UL"],[" UL"," UL"," UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    labelmode='fluence',
                        current_at=U)
    xs = pointsets.addToPlot("I", ["2002_UR","2003_UR","2102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        marker='x',
                    labelmode='fluence',
                        current_at=U)
    
    setAxisIV(ax,"I("+voltstring+")[A]")
    plt.ylim([0,3.5e-5])
    plt.text(.3, 0.2e-5, "200 µm FZ",fontdict=styles.fontdict())
    plt.tight_layout()
    plt.savefig(outdir+'200_'+str(U)+'.pdf')
    
    
    ax = newplot()
    pointsets.addToPlot("IperFluence", ["2002_UL","2003_UL","2102_UL"],[" UL"," UL"," UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    
    xs = pointsets.addToPlot("IperFluence", ["2002_UR","2003_UR","2102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        marker='x',
                        current_at=U)
    
    setAxisIV(ax, "I("+voltstring+")[A] / Fluence [neq/cm$^2$]")
    plt.ylim([0,0.75e-20])
    plt.tight_layout()
    plt.savefig(outdir+'200_IperFluence_'+str(U)+'.pdf')
    
    
    ax = newplot()
    pointsets.addToPlot("IperThickness", ["2002_UL","2003_UL","2102_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    
    xs = pointsets.addToPlot("IperThickness", ["2002_UR","2003_UR","2102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        marker='x',
                        current_at=U)
    
    setAxisIV(ax, "I("+voltstring+")[A] / thickness [µm]")
    plt.ylim([0,3.5e-5/200.])
    plt.tight_layout()
    plt.savefig(outdir+'200_IperThickness_'+str(U)+'.pdf')
    
    
    ax = newplot()
    pointsets.addToPlot("ItimesThickness", ["2002_UL","2003_UL","2102_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    
    xs = pointsets.addToPlot("ItimesThickness", ["2002_UR","2003_UR","2102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        marker='x',
                        current_at=U)
    
    setAxisIV(ax, "I("+voltstring+")[A] * thickness [µm]")
    plt.ylim(itimesthicknessylim)
    plt.tight_layout()
    plt.savefig(outdir+'200_ItimesThickness_'+str(U)+'.pdf')
    
    ###########
    ###########
    
    
    ax = newplot()
    pointsets.addToPlot("I", ["1002_UL","1003_UL","1102_UL"],[" UL"," UL"," UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    labelmode='fluence',
                        current_at=U)
    xs = pointsets.addToPlot("I", ["1002_UR","1003_UR","1102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x',
                    labelmode='fluence',
                        current_at=U)
    setAxisIV(ax,"I("+voltstring+")[A]")
    plt.ylim([0,2.1e-5])
    plt.text(.3, 0.1e-5, "300 µm FZ",fontdict=styles.fontdict())
    plt.tight_layout()
    plt.savefig(outdir+'300_'+str(U)+'.pdf')
    
    
    ax = newplot()
    pointsets.addToPlot("IperFluence", ["1002_UL","1003_UL","1102_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    xs = pointsets.addToPlot("IperFluence", ["1002_UR","1003_UR","1102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x',
                        current_at=U)
    
    setAxisIV(ax, "I("+voltstring+")[A] / Fluence [neq/cm$^2$]")
    plt.ylim([0,1e-20])
    plt.tight_layout()
    plt.savefig(outdir+'300_IperFluence_'+str(U)+'.pdf')
    
    ax = newplot()
    pointsets.addToPlot("IperThickness", ["1002_UL","1003_UL","1102_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    xs = pointsets.addToPlot("IperThickness", ["1002_UR","1003_UR","1102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x',
                        current_at=U)
    
    setAxisIV(ax, "I("+voltstring+")[A] / thickness [µm]")
    plt.ylim([0,2.1e-5/300])
    plt.tight_layout()
    plt.savefig(outdir+'300_IperThickness_'+str(U)+'.pdf')
    
    ax = newplot()
    pointsets.addToPlot("ItimesThickness", ["1002_UL","1003_UL","1102_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    xs = pointsets.addToPlot("ItimesThickness", ["1002_UR","1003_UR","1102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x',
                        current_at=U)
    
    setAxisIV(ax, "I("+voltstring+")[A] * thickness [µm]")
    plt.ylim(itimesthicknessylim)
    plt.tight_layout()
    plt.savefig(outdir+'300_ItimesThickness_'+str(U)+'.pdf')
    
    
### alpha plot

U = "Udep"

plt.close()
f = plt.figure()
f.set_figwidth(10)
f.set_figheight(6)

pointsets.addToPlot("IperThickness", ["1002_UL","1003_UL","1102_UL"],
                    #colors=['tab:blue','tab:green','tab:orange'],
                colors='fluence',
                    current_at=U)
xs = pointsets.addToPlot("IperThickness", ["1002_UR","1003_UR","1102_UR"],[" UR"," UR"," UR"],
                    #colors=['tab:blue','tab:green','tab:orange'],
                colors='fluence',
                marker='x',
                    current_at=U)

###
pointsets.addToPlot("IperThickness", ["2002_UL","2003_UL","2102_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='d',
                        current_at=U)
    
xs = pointsets.addToPlot("IperThickness", ["2002_UR","2003_UR","2102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        marker='s',
                        current_at=U)

###
xs = pointsets.addToPlot("IperThickness", ["3003_UL","3007_UL","3008_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='v',
                        current_at=U)

setAxisIV(ax, "I[A]($U_{dep}$) / thickness [µm]", bbox_to_anchor=(1,1), loc="upper left")
plt.yscale('log')
plt.ylim([7e-9,4e-7])
plt.savefig(outdir+'all_alphaslope_'+str(U)+'.pdf')

    