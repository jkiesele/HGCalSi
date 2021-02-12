import math
import numpy as np

from fileIO import fileReader
from plotting import curvePlotter
from matplotlib import pyplot as plt
import os
from fitting import DepletionFitter


def getDepletionVoltage(filename, debug=True,
                        const_cap=None,
                        rising=1, constant=8,
                        min_x=None,max_x=None,
                        debugfile=None,
                        low_start  =None,
                        low_end = None,
                        high_start=None,
                        high_end =None,
                        savedatapath=None
                        ):
    pl = curvePlotter(mode="CVs")
    pl.addPlotFromFile(filename, min_x=min_x,max_x=max_x, noplot=True)
    
    df = DepletionFitter(x=pl.x, y=pl.y, 
                 const_cap=const_cap,
                 rising=rising, constant=constant,
                 debugfile=debugfile,
                 low_start  =low_start,
                        low_end = low_end,
                        high_start=high_start,
                        high_end =high_end)
    v = df.getDepletionVoltage(debugplot=debug,savedatapath=savedatapath)
    return v


def plotFullSet(minstr, outdir, identifier = "UL_diode_big", globalpath=None):
    os.system("mkdir -p "+outdir)
    if globalpath is None:
        globalpath=os.getenv("DATAPATH")+'/'
    
    
    
    cv_plotter = curvePlotter(mode="CVs",path=globalpath)
    iv_plotter = curvePlotter(mode="IV",path=globalpath)
    
    
    
    
    for m in ['cv','iv']:
        
        plter=cv_plotter
        if m=='iv':
            plter=iv_plotter
    
        
        plter.addPlotFromFile("1002_UL*diode_big_no_ann/1002_UL*diode_big_no_ann_2020-11-12_1."+m,
                                   label="6.5 E14 $neq/cm^2$, no ann.", 
                                   min_x=-950)
        
        
        plter.addPlotFromFile("1003_UL*diode_big_no_ann/1003_UL*diode_big_no_ann_2020-11-12_1."+m,
                                   label="1.0 E15 $neq/cm^2$, no ann.", 
                                   min_x=-950)
      
        
        plter.addPlotFromFile("1102_UL*diode_big_no_ann/1102_UL*diode_big_no_ann_2020-11-10_2."+m,
                                   label="1.5 E15 $neq/cm^2$, no ann.", 
                                   min_x=-950)
        
        #plt.legend(loc='lower left')
        #plter.labelAxes()
        #plt.twinx()
        
        
        
        plter.addPlotFromFile("1002_"+identifier+"_ann_"+minstr+"/*."+m,
                                   label="6.5 E14 $neq/cm^2$, "+minstr+"@60C", 
                                   linestyle='dashed'
                                   )
        
    
        plter.addPlotFromFile("1003_"+identifier+"_ann_"+minstr+"/*."+m,
                                   label="1.0 E15 $neq/cm^2$, "+minstr+"@60C",  
                                   linestyle='dashed'
                                   )
        
        plter.addPlotFromFile("1102_"+identifier+"_ann_"+minstr+"/*."+m,
                                   label="1.5 E15 $neq/cm^2$, "+minstr+"@60C",  
                                   linestyle='dashed'
                                   )
        
        plt.title("300µm")
        plt.legend()
        if m=='cv':
            plt.ylim([0.,1.25e22])
        else:
            plt.ylim([0.,1.5e-5])
        plter.savePlot(outdir+m+"_1001_1002_1003_1102.pdf",nolegend=True)
        
        
        
        plter.addPlotFromFile("2002_UL*diode_big_no_ann/2002_UL*diode_big_no_ann_2020-11-12_1."+m,
                                   label="1.0 E15 $neq/cm^2$ ,no ann.", 
                                   min_x=-950)
        
        
        plter.addPlotFromFile("2003_UL*diode_big_no_ann/2003_UL*diode_big_no_ann_2020-11-12_1."+m,
                                   label="1.5 E15 $neq/cm^2$, no ann.", 
                                   min_x=-950)
        
        
        plter.addPlotFromFile("2102_UL*diode_big_no_ann/2102_UL*diode_big_no_ann_2020-11-12_1."+m,
                                   label="2.5 E15 $neq/cm^2$, no ann.", 
                                   min_x=-950)
        
        #plt.legend(loc='lower left')
        ##plt.subplots_adjust(left=0.0, bottom=0.0, right=0.0)
        #plter.labelAxes()
        #plt.twinx()
        
        
        
        plter.addPlotFromFile("2002_"+identifier+"_ann_"+minstr+"/*."+m,
                                   label="1E15 $neq/cm^2$, "+minstr+"@60C",  
                                   linestyle='dashed'
                                   )
        
        
        
        plter.addPlotFromFile("2003_"+identifier+"_ann_"+minstr+"/*."+m,
                                   label="1.5E15 $neq/cm^2$, "+minstr+"@60C",  
                                   linestyle='dashed'
                                   )
        
       
        
        plter.addPlotFromFile("2102_"+identifier+"_ann_"+minstr+"/*."+m,
                                   label="2.5E15 $neq/cm^2$, "+minstr+"@60C",  
                                   linestyle='dashed'
                                   )
        
        
        plt.title("200µm")
        plt.legend()
        if m=='cv':
            plt.ylim([0.,6e21])
        else:
            plt.ylim([-1.e-6,2.1e-5])
        
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
