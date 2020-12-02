import math
import numpy as np

from fileIO import fileReader
from plotting import curvePlotter
from matplotlib import pyplot as plt
import os



def plotFullSet(minstr, outdir, globalpath):
    os.system("mkdir -p "+outdir)
    
    
    cv_plotter = curvePlotter(mode="CVs",path=globalpath)
    iv_plotter = curvePlotter(mode="IV",path=globalpath)
    
    
    linestyle='-'
    
    for m in ['cv','iv']:
        
        plter=cv_plotter
        if m=='iv':
            plter=iv_plotter
    
        
        plter.addPlotFromFile("1002_UL-diode_big_no_ann/1002_UL-diode_big_no_ann_2020-11-12_1."+m,
                                   label="1002 (300µm), 6.5 E14 $neq/cm^2$, no ann", 
                                   min_x=-950)
        
        
        plter.addPlotFromFile("1003_UL-diode_big_no_ann/1003_UL-diode_big_no_ann_2020-11-12_1."+m,
                                   label="1003 (300µm), 1.0 E15 $neq/cm^2$, no ann", 
                                   min_x=-950)
      
        
        plter.addPlotFromFile("1102_UL_diode_big_no_ann/1102_UL_diode_big_no_ann_2020-11-10_2."+m,
                                   label="1102 (300µm), 1.5 E15 $neq/cm^2$, no ann", 
                                   min_x=-950)
        
        #plt.legend(loc='lower left')
        #plter.labelAxes()
        #plt.twinx()
        
        
        
        plter.addPlotFromFile("1002_UL_diode_big_ann_"+minstr+"/*."+m,
                                   label="1002 (300µm), 6.5 E14 $neq/cm^2$, ann. "+minstr+"@60C", 
                                   linestyle='dashed'
                                   )
        
    
        plter.addPlotFromFile("1003_UL_diode_big_ann_"+minstr+"/*."+m,
                                   label="1003 (300µm), 1.0 E15 $neq/cm^2$, ann. "+minstr+"@60C",  
                                   linestyle='dashed'
                                   )
        
        plter.addPlotFromFile("1102_UL_diode_big_ann_"+minstr+"/*."+m,
                                   label="1102 (300µm), 1.5 E15 $neq/cm^2$, ann. "+minstr+"@60C",  
                                   linestyle='dashed'
                                   )
        
        
        plt.legend()
        plter.savePlot(outdir+m+"_1001_1002_1003_1102.pdf",nolegend=True)
        
        
        
        plter.addPlotFromFile("2002_UL-diode_big_no_ann/2002_UL-diode_big_no_ann_2020-11-12_1."+m,
                                   label="2002 (200µm), 1.0 E15 $neq/cm^2$", 
                                   min_x=-950)
        
        
        plter.addPlotFromFile("2003_UL-diode_big_no_ann/2003_UL-diode_big_no_ann_2020-11-12_1."+m,
                                   label="2003 (200µm), 1.5 E15 $neq/cm^2$", 
                                   min_x=-950)
        
        
        plter.addPlotFromFile("2102_UL-diode_big_no_ann/2102_UL-diode_big_no_ann_2020-11-12_1."+m,
                                   label="2102 (200µm), 2.5 E15 $neq/cm^2$", 
                                   min_x=-950)
        
        #plt.legend(loc='lower left')
        ##plt.subplots_adjust(left=0.0, bottom=0.0, right=0.0)
        #plter.labelAxes()
        #plt.twinx()
        
        
        
        plter.addPlotFromFile("2002_UL_diode_big_ann_"+minstr+"/*."+m,
                                   label="2002 (200µm), 1E15 $neq/cm^2$, ann. "+minstr+"@60C",  
                                   linestyle='dashed'
                                   )
        
        
        
        plter.addPlotFromFile("2003_UL_diode_big_ann_"+minstr+"/*."+m,
                                   label="2003 (200µm), 1.5E15 $neq/cm^2$, ann. "+minstr+"@60C",  
                                   linestyle='dashed'
                                   )
        
       
        
        plter.addPlotFromFile("2102_UL_diode_big_ann_"+minstr+"/*."+m,
                                   label="2102 (200µm), 2.5E15 $neq/cm^2$, ann. "+minstr+"@60C",  
                                   linestyle='dashed'
                                   )
        
        
        plt.legend()
        
        plter.savePlot(outdir+m+"_2001_2002_2003_2102.pdf",nolegend=True)
 


'''

'''

'''
no_anneal
V_detector [V]  Capacitance [F] Conductivity [S]        Bias [V]        Curre
BEGIN
-8.727273E+2    1.342080E-11    1.823960E-8     -8.727273E+2    -2.461609E-5
-8.818182E+2    1.341960E-11    1.775680E-8     -8.818182E+2    -2.478013E-5
-8.909091E+2    1.341810E-11    1.750070E-8     -8.909091E+2    -2.486393E-5
-9.000000E+2    1.341720E-11    1.734480E-8     -9.000000E+2    -2.507583E-5
'''


'''
7min
V_detector [V]  Capacitance [F] Conductivity [S]        Bias [V]        Curren

-8.698305E+2    -3.279593E-12   -7.811066E-6    -8.698305E+2    -2.146823E-5
-8.849153E+2    -3.280993E-12   -7.810917E-6    -8.849153E+2    -2.170022E-5
-9.000000E+2    -3.281693E-12   -7.810734E-6    -9.000000E+2    -2.202053E-5
'''