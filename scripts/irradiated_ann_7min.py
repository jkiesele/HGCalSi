#!/usr/bin/python3

from fileIO import fileReader
from plotting import curvePlotter



from matplotlib import pyplot as plt

globalpath="/Users/jkiesele/cern_afs/eos_hgsensor_testres/Results_SSD/CVIV/Diode_TS/"

outdir="irradiated_ann_7min/"

cv_plotter = curvePlotter(mode="CVs",path=globalpath)
iv_plotter = curvePlotter(mode="IV",path=globalpath)
ivgr_plotter = curvePlotter(mode="IVGR",path=globalpath)


linestyle='-'

for m in ['cv','iv']:
    
    plter=cv_plotter
    if m=='iv':
        plter=iv_plotter

    
    plter.addPlotFromFile("1002_UL-diode_big_no_ann/1002_UL-diode_big_no_ann_2020-11-12_1."+m,
                               label="1002 (300µm), 6.5 E14 $neq/cm^2$", 
                               min_x=-950)
    
    
    plter.addPlotFromFile("1003_UL-diode_big_no_ann/1003_UL-diode_big_no_ann_2020-11-12_1."+m,
                               label="1003 (300µm), 1.0 E15 $neq/cm^2$", 
                               min_x=-950)
    
    
    plter.addPlotFromFile("1102_UL_diode_big_no_ann/1102_UL_diode_big_no_ann_2020-11-10_2."+m,
                               label="1102 (300µm), 1.5 E15 $neq/cm^2$", 
                               min_x=-950)
    
    #plt.legend(loc='lower left')
    #plter.labelAxes()
    #plt.twinx()
    
    if m=='iv':
        plter=ivgr_plotter
    
    
    plter.addPlotFromFile("1002_UL_diode_big_ann_7min/1002_UL_diode_big_ann_7min_2020-11-1*."+m,
                               label="1002 (300µm), 6.5 E14 $neq/cm^2$, ann. 7min@60C", 
                               linestyle='dashed'
                               )
    
    
    plter.addPlotFromFile("1003_UL_diode_big_ann_7min/1003_UL_diode_big_ann_7min_2020-11-19_1."+m,
                               label="1003 (300µm), 1.0 E15 $neq/cm^2$, ann. 7min@60C",  
                               linestyle='dashed'
                               )
    
    
    plter.addPlotFromFile("1102_UL_diode_big_ann_7min/1102_UL_diode_big_ann_7min_2020-11-19_1."+m,
                               label="1102 (300µm), 1.5 E15 $neq/cm^2$, ann. 7min@60C",  
                               linestyle='dashed'
                               )
    
    
    plt.legend(loc='upper right')
    plter.savePlot(outdir+m+"_1001_1002_1003_1102.pdf",nolegend=True)
    
    
    if m=='iv':
        plter=iv_plotter
    
    
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
    
    
    if m=='iv':
        plter=ivgr_plotter
    
    plter.addPlotFromFile("2002_UL_diode_big_ann_7min/2002_UL_diode_big_ann_7min_2020-11-19_1."+m,
                               label="2002 (200µm), 1E15 $neq/cm^2$, ann. 7min@60C",  
                               linestyle='dashed'
                               )
    
    
    plter.addPlotFromFile("2003_UL_diode_big_ann_7min/2003_UL_diode_big_ann_7min_2020-11-19_1."+m,
                               label="2003 (200µm), 1.5E15 $neq/cm^2$, ann. 7min@60C",  
                               linestyle='dashed'
                               )
    
    
    plter.addPlotFromFile("2102_UL_diode_big_ann_7min/2102_UL_diode_big_ann_7min_2020-11-19_1."+m,
                               label="2102 (200µm), 2.5E15 $neq/cm^2$, ann. 7min@60C",  
                               linestyle='dashed'
                               )
    
    
    plt.legend(loc='upper right')
    
    plter.savePlot(outdir+m+"_2001_2002_2003_2102.pdf",nolegend=True)
    

##############






