#!/usr/bin/python3

import os
import styles
import matplotlib.pyplot as plt
from tools import convert60Cto21C

from pointset import pointSetsContainer, pointSet, loadAnnealings

datadir=os.getenv("DATAOUTPATH")

def cosmetics(x,ylabel="-$U_{depl}$ [V]"):
    
    plt.legend()
    plt.xscale('log')
    plt.xlabel("time (60˚C) [min]", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    def minat60ToDaysAt21(t):
        return convert60Cto21C(t/(60*24.))
    styles.addModifiedXAxis(x, minat60ToDaysAt21, "time (21˚C) [d]")
    plt.xscale('log')
    
    plt.tight_layout()
    

pointsets, kneepointset= loadAnnealings()


plt.close()
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
cosmetics(xs)
plt.savefig(datadir+"/annealing_plots/300_6inch.pdf")


plt.close()
pointsets.addToPlot("Udep",["6002_6in"],[" second kink"],
                    #colors=['tab:red','tab:purple'],
                    colors='fluence',
                    marker='x')
xs = kneepointset.addToPlot("Udep",["6002_6in"],[" first kink"],
                       #colors=['tab:red','tab:purple']
                    colors='fluence',
                       )
cosmetics(xs)
plt.savefig(datadir+"/annealing_plots/6inch.pdf")


plt.close()
pointsets.addToPlot("Udep", ["1002_UL","1003_UL","1102_UL"],
                    #colors=['tab:blue','tab:green','tab:orange']
                    colors='fluence',
                    )
xs = pointsets.addToPlot("Udep", ["1002_UR","1003_UR","1102_UR"],[" UR"," UR"," UR"],
                    #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x')
cosmetics(xs)
plt.savefig(datadir+"/annealing_plots/300.pdf")


plt.close()
pointsets.addToPlot("Udep", ["2002_UL","2003_UL","2102_UL"],
                    #colors=['tab:blue','tab:green','tab:orange']
                    colors='fluence',
                    )
xs = pointsets.addToPlot("Udep", ["2002_UR","2003_UR","2102_UR"],[" UR"," UR"," UR"],
                    #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x')
cosmetics(xs)
plt.savefig(datadir+"/annealing_plots/200.pdf")


plt.close()
xs = pointsets.addToPlot("Udep", ["3003_UL","3007_UL","3008_UL"],
                    #colors=['tab:blue','tab:green','tab:orange']
                    colors='fluence',
                    )

cosmetics(xs)
plt.savefig(datadir+"/annealing_plots/120.pdf")


plt.close()
xs = pointsets.addToPlot("NEff", ["3003_UL","3007_UL","3008_UL"]+["2002_UL","2003_UL","2102_UL"]+["2002_UR","2003_UR","2102_UR"])

cosmetics(xs,"NEff $[1/cm^{3}]$")
plt.show()



