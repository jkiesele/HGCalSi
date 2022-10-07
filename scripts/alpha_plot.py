#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import styles
import numpy as np
from pointset import pointSetsContainer, pointSet, loadAnnealings
from pointset import interpolatedPointSet
from fitting import alphaExtractor


styles.setstyles(16)

datadir=os.getenv("DATAOUTPATH")
outdir=os.getenv("DATAOUTPATH")+'/alpha_plots/'
os.system('mkdir -p '+outdir)

pointsets, kneepointset= loadAnnealings()

dataAll = pointsets.getInterpolatedXYsDiodes("IperVolume", 
                                          ["3003_UL","3007_UL","3008_UL"]+
                                          ["2002_UL","2003_UL","2102_UL","2002_UR","2003_UR","2102_UR"]+
                                          ["1002_UL","1003_UL","1102_UL","1002_UR","1003_UR","1102_UR"], 
                                          current_at='Udep',
                                          debugplots = {'outfile' : outdir+'EPI_FZ_interpol','xlabel': 'Annealing time [min]', 'ylabel': 'I(U$_{dep}$) [A]'}
                                          )

dataFZ = pointsets.getInterpolatedXYsDiodes("IperVolume", 
                                          ["2002_UL","2003_UL","2102_UL","2002_UR","2003_UR","2102_UR"]+
                                          ["1002_UL","1003_UL","1102_UL","1002_UR","1003_UR","1102_UR"], 
                                          current_at='Udep',
                                          debugplots = {'outfile' : outdir+'FZ_interpol', 'xlabel': 'Annealing time [min]','ylabel': 'I(U$_{dep}$) [A]'}
                                          )


dataEPI = pointsets.getInterpolatedXYsDiodes("IperVolume", 
                                          ["3003_UL","3007_UL","3008_UL"],
                                          current_at='Udep',
                                          debugplots = {'outfile' : outdir+'EPI_interpol', 'xlabel': 'Annealing time [min]','ylabel': 'I(U$_{dep}$) [A]'}
                                          )


alldata=[]
i=0
for savestring, data, c in zip(['FZ','EPI'],[dataFZ,dataEPI],['tab:orange','tab:green']):
    extractor = alphaExtractor(interpolated_pointset_dict=data,
                                rel_fluence_uncertainty=1e-1)
    
    
    #set some offsets so the plot can be read better
    times =  [10+i/3,30+3*i/3,80+8*i/3, 100+10*i/3, 150+15*i/3, 250+25*i/3, 380+38*i/3, 640+64*i/3]
    
    alphas, alphaerrs,times = extractor.extractAlpha(times,plotstring=outdir+'alphafit_'+savestring)
    
    alldata.append((times, alphas, alphaerrs, savestring, c))
    i+=1.
    
for t,a,ae,sstr,color in alldata:    
    
    plt.close()
    plt.errorbar(t,a,yerr=np.sqrt((0.1*a)**2+ ae**2), #add the 10% fluence uncertainty in quadrature a posteriori
                 linestyle=None,linewidth=0,elinewidth=2.,label=None,alpha=0.2,color=color)
    
    plt.errorbar(t,a,yerr=ae,marker='o',
                 linestyle=None,linewidth=0,elinewidth=2.,label=sstr,color=color)
    
    if False:#estebans point
        plt.errorbar([10], [7.675], yerr=[0.2], xerr=[0.], marker='x',linewidth=0,elinewidth=2.,
                         label='6" DD (TDR)')
    
        plt.legend()
        

    plt.xlabel('Annealing time @ 60˚C [min]')
    plt.ylabel(r'$\alpha$ [$10^{-19}$ A/cm]')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(outdir+"annealing_alpha_"+sstr+".pdf")



plt.close()

import styles

for t,a,ae,sstr,color in alldata:
    plt.errorbar(t,a,yerr=np.sqrt((0.1*a)**2+ ae**2) ,
                 linestyle=None,linewidth=0,elinewidth=2.,label=None,alpha=0.2,color=color)
    
    plt.errorbar(t,a,yerr=ae,marker='o',
                 linestyle=None,linewidth=0,elinewidth=2.,label=sstr,color=color)
    
    
    
#add MM reference 0.016947933384961075 from 20˚C to -20˚C
plt.errorbar([81], [0.016947933384961075 * 4e-17 * 1e19], 
             yerr=[0.016947933384961075 * 0.22e-17 * 1e19],
              linestyle=None,
              marker='x',
              linewidth=0.,
              elinewidth=2.,
              alpha=0.9,
              color='k',
              label= "M'99" )
    
plt.legend()
plt.xlabel('Annealing time @ 60˚C [min]')
plt.ylabel(r'$\alpha$ [$10^{-19}$ A/cm], -20˚C')
plt.xscale('log')
plt.tight_layout()
plt.savefig(outdir+"annealing_alpha.pdf")

plt.errorbar([10], [7.675], yerr=[0.2], xerr=[0.], marker='x',linewidth=0,elinewidth=2.,
                         label='6" DD (TDR)')

plt.legend()
plt.tight_layout()
plt.tight_layout()
plt.savefig(outdir+"annealing_alpha_TDR.pdf")





