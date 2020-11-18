#!/usr/bin/python3

from fileIO import fileReader
from plotting import curvePlotter



from matplotlib import pyplot as plt

globalpath="/Users/jkiesele/cern_afs/eos_hgsensor_testres/Results_SSD/CVIV/Diode_TS/"

outdir="irradiated_no_anneal/"

cv_plotter = curvePlotter(mode="CVs",path=globalpath)
iv_plotter = curvePlotter(mode="IV",path=globalpath)


linestyle='-'

cv_plotter.addPlotFromFile("1001_UL_diode_big_no_ann/1001_UL_diode_big_no_ann_2020-11-10_1.cv",
                           label="1001 (300µm), 6.5E14 $neq/cm^2$", 
                           min_x=-930)


cv_plotter.addPlotFromFile("1102_UL_diode_big_no_ann/1102_UL_diode_big_no_ann_2020-11-10_2.cv",
                           label="1102 (300µm), 1E15 $neq/cm^2$", 
                           min_x=-950)


cv_plotter.savePlot(outdir+"CV_1001_1102.pdf")

cv_plotter.addPlotFromFile("2001_UL_diode_big_no_ann/2001_UL_diode_big_no_ann_2020-11-10_1.cv",
                           label="2001 (200µm), 1E15 $neq/cm^2$", 
                           min_x=-950)


cv_plotter.addPlotFromFile("2002_UL-diode_big_no_ann/2002_UL-diode_big_no_ann_2020-11-12_1.cv",
                           label="2002 (200µm), 1E15 $neq/cm^2$", 
                           min_x=-950)


cv_plotter.addPlotFromFile("2003_UL-diode_big_no_ann/2003_UL-diode_big_no_ann_2020-11-12_1.cv",
                           label="2003 (200µm), 1E15 $neq/cm^2$", 
                           min_x=-950)


cv_plotter.addPlotFromFile("2102_UL-diode_big_no_ann/2102_UL-diode_big_no_ann_2020-11-12_1.cv",
                           label="2102 (200µm), 2.5E15 $neq/cm^2$", 
                           min_x=-950)


cv_plotter.savePlot(outdir+"CV_2001_2002_2003_2102.pdf")


##############


iv_plotter.addPlotFromFile("1001_UL_diode_big_no_ann/1001_UL_diode_big_no_ann_2020-11-10_1.iv",
                           label="1001 (300µm), 6.5E14 $neq/cm^2$", 
                           min_x=-930)


iv_plotter.addPlotFromFile("1102_UL_diode_big_no_ann/1102_UL_diode_big_no_ann_2020-11-10_2.iv",
                           label="1102 (300µm), 1E15 $neq/cm^2$", 
                           min_x=-950)


iv_plotter.savePlot(outdir+"IV_1001_1102.pdf")

iv_plotter.addPlotFromFile("2001_UL_diode_big_no_ann/2001_UL_diode_big_no_ann_2020-11-10_1.iv",
                           label="2001 (200µm), 1E15 $neq/cm^2$", 
                           min_x=-950)


iv_plotter.addPlotFromFile("2002_UL-diode_big_no_ann/2002_UL-diode_big_no_ann_2020-11-12_1.iv",
                           label="2002 (200µm), 1E15 $neq/cm^2$", 
                           min_x=-950)


iv_plotter.addPlotFromFile("2003_UL-diode_big_no_ann/2003_UL-diode_big_no_ann_2020-11-12_1.iv",
                           label="2003 (200µm), 1E15 $neq/cm^2$", 
                           min_x=-950)


iv_plotter.addPlotFromFile("2102_UL-diode_big_no_ann/2102_UL-diode_big_no_ann_2020-11-12_1.iv",
                           label="2102 (200µm), 2.5E15 $neq/cm^2$", 
                           min_x=-950)


iv_plotter.savePlot(outdir+"IV_2001_2002_2003_2102.pdf")


