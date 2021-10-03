

import os
import matplotlib.pyplot as plt
import styles
import numpy as np
from pointset import pointSetsContainer, pointSet, loadAnnealings
from pointset import interpolatedPointSet
from fitting import alphaExtractor

pointsets, kneepointset= loadAnnealings()

data = pointsets.getInterpolatedXYsDiodes("IperVolume", 
                                          ["3003_UL","3007_UL","3008_UL"]+
                                          ["2002_UL","2003_UL","2102_UL","2002_UR","2003_UR","2102_UR"]+
                                          ["1002_UL","1003_UL","1102_UL","1002_UR","1003_UR","1102_UR"], 
                                          current_at='Udep')


extractor = alphaExtractor(interpolated_pointset_dict=data,
                            rel_fluence_uncertainty=1e-1)

#the correlation between the fluence uncertainteis is unclear.. better set it to zero

times = [10,30,74, 103, 151, 245, 380, 640]

alphas, alphaerrs = extractor.extractAlpha(times,plotstring='testalpha')

plt.errorbar(times,alphas,yerr=alphaerrs,marker='o',linestyle=None,linewidth=0,elinewidth=2.,)

plt.xlabel('Annealing time @ 60˚C [min]')
plt.ylabel(r'$\alpha$ [$10^{-19}$ A/cm]')
plt.xscale('log')
plt.tight_layout()
plt.savefig("annealing_alpha.pdf")
exit()
