#!/usr/bin/python3

from fileIO import fileReader
from plotting import curvePlotter

from matplotlib import pyplot as plt

globalpath="/Users/jkiesele/cern_afs/eos_hgsensor_testres/Results_SSD/CVIV/Diode_TS/"

outdir="non_irradiated/"

cv_plotter = curvePlotter(mode="CV",path=globalpath)
iv_plotter = curvePlotter(mode="IV",path=globalpath)


linestyle='-'

cv_plotter.addPlotFromFile("1004_UL_diode_big/1004_UL_diode_big_2020-11-06_1.cv",
                           label="1004 (300 µm), no GR", linestyle='dashed', color="tab:green")

cv_plotter.addPlotFromFile("1013_UL_diode_big/1013_UL_diode_big_2020-11-06_1.cv",
                           label="1013 (300 µm), no GR", linestyle='dashed', color="tab:orange")



cv_plotter.addPlotFromFile("1004_UL_diode_big_GR/1004_UL_diode_big_GR_2020-11-06_1.cv",
                           label="1004 (300 µm), GR",  linestyle=linestyle, color="tab:green")

cv_plotter.addPlotFromFile("1013_UL_diode_big_GR/1013_UL_diode_big_GR_2020-11-06_1.cv",
                           label="1013 (300 µm), GR",  linestyle=linestyle, color="tab:orange")



cv_plotter.savePlot(outdir+"CV_1004_1013.pdf")


cv_plotter.addPlotFromFile("2004_UL_diode_big/2004_UL_diode_big_2020-11-06_1.cv",
                           label="2004 (200 µm), no GR",  linestyle='dashed', color="tab:green")

cv_plotter.addPlotFromFile("2012_UL_diode_big/2012_UL_diode_big_2020-11-06_1.cv",
                           label="2012 (200 µm), no GR",  linestyle='dashed', color="tab:orange")



cv_plotter.addPlotFromFile("2004_UL_diode_big_GR/2004_UL_diode_big_GR_2020-11-06_1.cv",
                           label="2004 (200 µm), GR",  linestyle=linestyle, color="tab:green")

cv_plotter.addPlotFromFile("2012_UL_diode_big_GR/2012_UL_diode_big_GR_2020-11-06_1.cv",
                           label="2012 (200 µm), GR",  linestyle=linestyle, color="tab:orange")


cv_plotter.savePlot(outdir+"CV_2004_2012.pdf")


cv_plotter.addPlotFromFile("3103_UL_diode_big/3103_UL_diode_big_2020-11-06_1.cv",
                           label="3103 (120 µm), no GR",  linestyle='dashed', color="tab:green")

cv_plotter.addPlotFromFile("3104_UL_diode_big/3104_UL_diode_big_2020-11-06_2.cv",
                           label="3104 (120 µm), no GR",  linestyle='dashed', color="tab:orange")


cv_plotter.addPlotFromFile("3103_UL_diode_big_GR/3103_UL_diode_big_GR_2020-11-06_1.cv",
                           label="3103 (120 µm), GR",  linestyle=linestyle, color="tab:green")

cv_plotter.addPlotFromFile("3104_UL_diode_big_GR/3104_UL_diode_big_GR_2020-11-06_2.cv",
                           label="3104 (120 µm), GR",  linestyle=linestyle, color="tab:orange")


cv_plotter.savePlot(outdir+"CV_3103_3104.pdf")


##############



iv_plotter.addPlotFromFile("1004_UL_diode_big/1004_UL_diode_big_2020-11-06_1.iv",
                           label="1004 (300 µm), no GR",  linestyle='dashed', color="tab:green")

iv_plotter.addPlotFromFile("1013_UL_diode_big/1013_UL_diode_big_2020-11-06_1.iv",
                           label="1013 (300 µm), no GR",  linestyle='dashed', color="tab:orange")

iv_plotter.savePlot(outdir+"IV_1004_1013_noGR.pdf")


    

iv_plotter.addPlotFromFile("1004_UL_diode_big_GR/1004_UL_diode_big_GR_2020-11-06_1.iv",
                           label="1004 (300 µm), GR",  linestyle=linestyle, color="tab:green")

iv_plotter.addPlotFromFile("1013_UL_diode_big_GR/1013_UL_diode_big_GR_2020-11-06_1.iv",
                           label="1013 (300 µm), GR",  linestyle=linestyle, color="tab:orange")


iv_plotter.savePlot(outdir+"IV_1004_1013.pdf")


iv_plotter.addPlotFromFile("2004_UL_diode_big/2004_UL_diode_big_2020-11-06_1.iv",
                           label="2004 (200 µm), no GR",  linestyle='dashed', color="tab:green")

iv_plotter.addPlotFromFile("2012_UL_diode_big/2012_UL_diode_big_2020-11-06_1.iv",
                           label="2012 (200 µm), no GR",  linestyle='dashed', color="tab:orange")


iv_plotter.savePlot(outdir+"IV_2004_2012_noGR.pdf")

iv_plotter.addPlotFromFile("2004_UL_diode_big_GR/2004_UL_diode_big_GR_2020-11-06_1.iv",
                           label="2004 (200 µm), GR",  linestyle=linestyle, color="tab:green")

iv_plotter.addPlotFromFile("2012_UL_diode_big_GR/2012_UL_diode_big_GR_2020-11-06_1.iv",
                           label="2012 (200 µm), GR",  linestyle=linestyle, color="tab:orange")


iv_plotter.savePlot(outdir+"IV_2004_2012.pdf")

def selection(x,y):
    return x[x<-50],y[x<-50]

iv_plotter.addPlotFromFile("2012_UL_diode_big_GR/2012_UL_diode_big_GR_2020-11-06_1.iv",
                           selection=selection,
                           label="2012 (200 µm), GR",  linestyle=linestyle, color="tab:orange")



iv_plotter.savePlot(outdir+"IV_2012_select.pdf")



iv_plotter.addPlotFromFile("3103_UL_diode_big/3103_UL_diode_big_2020-11-06_1.iv",
                           label="3103 (120 µm), no GR",  linestyle='dashed', color="tab:green")

iv_plotter.addPlotFromFile("3104_UL_diode_big/3104_UL_diode_big_2020-11-06_1.iv",
                           label="3104 (120 µm), no GR",  linestyle='dashed', color="tab:orange")

iv_plotter.savePlot(outdir+"IV_3103_3104_noGR.pdf")

iv_plotter.addPlotFromFile("3103_UL_diode_big_GR/3103_UL_diode_big_GR_2020-11-06_1.iv",
                           label="3103 (120 µm), GR",  linestyle=linestyle, color="tab:green")

iv_plotter.addPlotFromFile("3104_UL_diode_big_GR/3104_UL_diode_big_GR_2020-11-06_2.iv",
                           label="3104 (120 µm), GR",  linestyle=linestyle, color="tab:orange")


iv_plotter.savePlot(outdir+"IV_3103_3104.pdf")


def selection(x,y):
    return x[x<-40],y[x<-40]


iv_plotter.addPlotFromFile("3103_UL_diode_big_GR/3103_UL_diode_big_GR_2020-11-06_1.iv",
                           selection=selection,
                           label="3103 (120 µm), GR",  linestyle=linestyle, color="tab:green")


def selection(x,y):
    return x[x<-260],y[x<-260]
iv_plotter.addPlotFromFile("3104_UL_diode_big_GR/3104_UL_diode_big_GR_2020-11-06_2.iv",
                           selection=selection,
                           label="3104 (120 µm), GR",  linestyle=linestyle, color="tab:orange")


iv_plotter.savePlot(outdir+"IV_3103_3104_select.pdf")



