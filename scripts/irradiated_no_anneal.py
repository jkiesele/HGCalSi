#!/usr/bin/python3

from fileIO import fileReader
from plotting import curvePlotter



from matplotlib import pyplot as plt

globalpath="/Users/jkiesele/cern_afs/eos_hgsensor_testres/Results_SSD/CVIV/Diode_TS/"

outdir="irradiated_no_anneal/"

cv_plotter = curvePlotter(mode="CVs",path=globalpath)
iv_plotter = curvePlotter(mode="IV",path=globalpath)


linestyle='-'

for m in ['cv','iv']:
    
    plter=cv_plotter
    if m=='iv':
        plter=iv_plotter

    plter.addPlotFromFile("1001_UL_diode_big_no_ann/1001_UL_diode_big_no_ann_2020-11-10_1."+m,
                               label="1001 (300µm), 6.5 E14 $neq/cm^2$, $V_{fb}=-5V$", 
                               min_x=-910)
    
    
    plter.addPlotFromFile("1002_UL-diode_big_no_ann/1002_UL-diode_big_no_ann_2020-11-12_1."+m,
                               label="1002 (300µm), 6.5 E14 $neq/cm^2$, $V_{fb}=-2V$", 
                               min_x=-950)
    
    
    plter.addPlotFromFile("1003_UL-diode_big_no_ann/1003_UL-diode_big_no_ann_2020-11-12_1."+m,
                               label="1003 (300µm), 1.0 E15 $neq/cm^2$, $V_{fb}=-2V$", 
                               min_x=-950)
    
    
    plter.addPlotFromFile("1102_UL_diode_big_no_ann/1102_UL_diode_big_no_ann_2020-11-10_2."+m,
                               label="1102 (300µm), 1.5 E15 $neq/cm^2$, $V_{fb}=-2V$", 
                               min_x=-950)
    
    plter.savePlot(outdir+m+"_1001_1002_1003_1102.pdf")
    
    
    plter.addPlotFromFile("2001_UL_diode_big_no_ann/2001_UL_diode_big_no_ann_2020-11-10_1."+m,
                               label="2001 (200µm), 1.0 E15 $neq/cm^2$, $V_{fb}=-5V$", 
                               min_x=-950)
    
    
    plter.addPlotFromFile("2002_UL-diode_big_no_ann/2002_UL-diode_big_no_ann_2020-11-12_1."+m,
                               label="2002 (200µm), 1.0 E15 $neq/cm^2$, $V_{fb}=-2V$", 
                               min_x=-950)
    
    
    plter.addPlotFromFile("2003_UL-diode_big_no_ann/2003_UL-diode_big_no_ann_2020-11-12_1."+m,
                               label="2003 (200µm), 1.5 E15 $neq/cm^2$, $V_{fb}=-2V$", 
                               min_x=-950)
    
    
    plter.addPlotFromFile("2102_UL-diode_big_no_ann/2102_UL-diode_big_no_ann_2020-11-12_1."+m,
                               label="2102 (200µm), 2.5 E15 $neq/cm^2$, $V_{fb}=-2V$", 
                               min_x=-950)
    
    
    plter.savePlot(outdir+m+"_2001_2002_2003_2102.pdf")
    

##############






