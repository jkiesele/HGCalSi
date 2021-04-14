#!/usr/bin/python3

import os
import matplotlib.pyplot as plt
import styles

from tools import convert60Cto21C

from pointset import pointSetsContainer, pointSet, loadAnnealings

datadir=os.getenv("DATAOUTPATH")
outdir=os.getenv("DATAOUTPATH")+'/annealing_plots/'
os.system('mkdir -p '+outdir)

print(outdir)

def setAxisIV(x,ylabel="-$I$ [A]",**legkwargs):
    
    plt.legend(**legkwargs)
    plt.xscale('log')
    plt.xlabel("time (60˚C) [min]", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    def minat60ToDaysAt21(t):
        return convert60Cto21C(t/(60*24.))
    styles.addModifiedXAxis(x, minat60ToDaysAt21, "time (21˚C) [d]")
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
    
    plt.close()
    xs = pointsets.addToPlot("I", ["3003_UL","3007_UL","3008_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    setAxisIV(xs,"I("+voltstring+")[A]")
    plt.ylim([0,6e-5])
    plt.tight_layout()
    plt.savefig(outdir+'120_'+str(U)+'.pdf')
    
    plt.close()
    xs = pointsets.addToPlot("IperFluence", ["3003_UL","3007_UL","3008_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    setAxisIV(xs, "I("+voltstring+")[A] / Fluence [neq/cm$^2$]")
    plt.ylim([0,0.8e-20])
    plt.tight_layout()
    plt.savefig(outdir+'120_IperFluence_'+str(U)+'.pdf')
    
    
    plt.close()
    xs = pointsets.addToPlot("IperThickness", ["3003_UL","3007_UL","3008_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    setAxisIV(xs, "I("+voltstring+")[A] / thickness [µm]")
    plt.ylim([0,6e-5/120.])
    plt.savefig(outdir+'120_IperThickness_'+str(U)+'.pdf')
    
    plt.close()
    xs = pointsets.addToPlot("ItimesThickness", ["3003_UL","3007_UL","3008_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    setAxisIV(xs, "I("+voltstring+")[A] * thickness [µm]")
    plt.ylim(itimesthicknessylim)
    plt.tight_layout()
    plt.savefig(outdir+'120_ItimesThickness_'+str(U)+'.pdf')
    
    ###########
    
    plt.close()
    pointsets.addToPlot("I", ["2002_UL","2003_UL","2102_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    xs = pointsets.addToPlot("I", ["2002_UR","2003_UR","2102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        marker='x',
                        current_at=U)
    
    setAxisIV(xs,"I("+voltstring+")[A]")
    plt.ylim([0,3.5e-5])
    plt.tight_layout()
    plt.savefig(outdir+'200_'+str(U)+'.pdf')
    
    
    plt.close()
    pointsets.addToPlot("IperFluence", ["2002_UL","2003_UL","2102_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    
    xs = pointsets.addToPlot("IperFluence", ["2002_UR","2003_UR","2102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        marker='x',
                        current_at=U)
    
    setAxisIV(xs, "I("+voltstring+")[A] / Fluence [neq/cm$^2$]")
    plt.ylim([0,0.75e-20])
    plt.tight_layout()
    plt.savefig(outdir+'200_IperFluence_'+str(U)+'.pdf')
    
    
    plt.close()
    pointsets.addToPlot("IperThickness", ["2002_UL","2003_UL","2102_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    
    xs = pointsets.addToPlot("IperThickness", ["2002_UR","2003_UR","2102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        marker='x',
                        current_at=U)
    
    setAxisIV(xs, "I("+voltstring+")[A] / thickness [µm]")
    plt.ylim([0,3.5e-5/200.])
    plt.tight_layout()
    plt.savefig(outdir+'200_IperThickness_'+str(U)+'.pdf')
    
    
    plt.close()
    pointsets.addToPlot("ItimesThickness", ["2002_UL","2003_UL","2102_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    
    xs = pointsets.addToPlot("ItimesThickness", ["2002_UR","2003_UR","2102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        marker='x',
                        current_at=U)
    
    setAxisIV(xs, "I("+voltstring+")[A] * thickness [µm]")
    plt.ylim(itimesthicknessylim)
    plt.tight_layout()
    plt.savefig(outdir+'200_ItimesThickness_'+str(U)+'.pdf')
    
    ###########
    ###########
    
    
    plt.close()
    pointsets.addToPlot("I", ["1002_UL","1003_UL","1102_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    xs = pointsets.addToPlot("I", ["1002_UR","1003_UR","1102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x',
                        current_at=U)
    setAxisIV(xs,"I("+voltstring+")[A]")
    plt.ylim([0,2.1e-5])
    plt.tight_layout()
    plt.savefig(outdir+'300_'+str(U)+'.pdf')
    
    
    plt.close()
    pointsets.addToPlot("IperFluence", ["1002_UL","1003_UL","1102_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    xs = pointsets.addToPlot("IperFluence", ["1002_UR","1003_UR","1102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x',
                        current_at=U)
    
    setAxisIV(xs, "I("+voltstring+")[A] / Fluence [neq/cm$^2$]")
    plt.ylim([0,1e-20])
    plt.tight_layout()
    plt.savefig(outdir+'300_IperFluence_'+str(U)+'.pdf')
    
    plt.close()
    pointsets.addToPlot("IperThickness", ["1002_UL","1003_UL","1102_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    xs = pointsets.addToPlot("IperThickness", ["1002_UR","1003_UR","1102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x',
                        current_at=U)
    
    setAxisIV(xs, "I("+voltstring+")[A] / thickness [µm]")
    plt.ylim([0,2.1e-5/300])
    plt.tight_layout()
    plt.savefig(outdir+'300_IperThickness_'+str(U)+'.pdf')
    
    plt.close()
    pointsets.addToPlot("ItimesThickness", ["1002_UL","1003_UL","1102_UL"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                        current_at=U)
    xs = pointsets.addToPlot("ItimesThickness", ["1002_UR","1003_UR","1102_UR"],[" UR"," UR"," UR"],
                        #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x',
                        current_at=U)
    
    setAxisIV(xs, "I("+voltstring+")[A] * thickness [µm]")
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

setAxisIV(xs, "I[A]($U_{dep}$) / thickness [µm]", bbox_to_anchor=(1,1), loc="upper left")
plt.yscale('log')
plt.ylim([7e-9,4e-7])
plt.savefig(outdir+'all_alphaslope_'+str(U)+'.pdf')

    