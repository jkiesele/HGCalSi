#!/usr/bin/python3

from fileIO import fileReader
from plotting import curvePlotter

from matplotlib import pyplot as plt

globalpath="/Users/jkiesele/cern_afs/eos_hgsensor_testres/Results_SSD/CVIV/Diode_TS/"

outdir="non_irradiated/"

plter = curvePlotter(mode="CV",path=globalpath)
iv_plotter = curvePlotter(mode="IV",path=globalpath)


for m in ['cv','iv']:
    
    plter=plter
    if m=='iv':
        plter=iv_plotter
        
        
    linestyle='-'
    
    plter.addPlotFromFile("1004_UL_diode_big/1004_UL_diode_big_2020-11-06_1."+m,
                               label="1004 (300 µm), no GR, $V_{fb}=-5V$", linestyle='dashed', color="tab:green")
    
    plter.addPlotFromFile("1013_UL_diode_big/1013_UL_diode_big_2020-11-06_1."+m,
                               label="1013 (300 µm), no GR, $V_{fb}=-2V$", linestyle='dashed', color="tab:orange")
    
    
    if m=='iv':
        plter.savePlot(outdir+m+"_1004_1013_noGR.pdf")
    
    plter.addPlotFromFile("1004_UL_diode_big_GR/1004_UL_diode_big_GR_2020-11-06_1."+m,
                               label="1004 (300 µm), GR, $V_{fb}=-5V$",  linestyle=linestyle, color="tab:green")
    
    plter.addPlotFromFile("1013_UL_diode_big_GR/1013_UL_diode_big_GR_2020-11-06_1."+m,
                               label="1013 (300 µm), GR, $V_{fb}=-2V$",  linestyle=linestyle, color="tab:orange")
    
    
    
    plter.savePlot(outdir+m+"_1004_1013.pdf")
    
    
    plter.addPlotFromFile("2004_UL_diode_big/2004_UL_diode_big_2020-11-06_1."+m,
                               label="2004 (200 µm), no GR, $V_{fb}=-5V$",  linestyle='dashed', color="tab:green")
    
    plter.addPlotFromFile("2012_UL_diode_big/2012_UL_diode_big_2020-11-06_1."+m,
                               label="2012 (200 µm), no GR, $V_{fb}=-2V$",  linestyle='dashed', color="tab:orange")
    
    
    if m=='iv':
        plter.savePlot(outdir+m+"_2004_2012_noGR.pdf")
    
    plter.addPlotFromFile("2004_UL_diode_big_GR/2004_UL_diode_big_GR_2020-11-06_1."+m,
                               label="2004 (200 µm), GR, $V_{fb}=-5V$",  linestyle=linestyle, color="tab:green")
    
    plter.addPlotFromFile("2012_UL_diode_big_GR/2012_UL_diode_big_GR_2020-11-06_1."+m,
                               label="2012 (200 µm), GR, $V_{fb}=-2V$",  linestyle=linestyle, color="tab:orange")
    
    
    plter.savePlot(outdir+m+"_2004_2012.pdf")
    
    if m == 'cv':
        plter.addPlotFromFile("3104_UL_diode_big/3104_UL_diode_big_2020-11-06_2."+m,
                               label="3104 (120 µm), no GR, $V_{fb}=-5V$",  linestyle='dashed', color="tab:green")
    else:
        plter.addPlotFromFile("3104_UL_diode_big/3104_UL_diode_big_2020-11-06_1."+m,
                               label="3104 (120 µm), no GR, $V_{fb}=-5V$",  linestyle='dashed', color="tab:green")
    
    
    plter.addPlotFromFile("3103_UL_diode_big/3103_UL_diode_big_2020-11-06_1."+m,
                               label="3103 (120 µm), no GR, $V_{fb}=-2V$",  linestyle='dashed', color="tab:orange")
    
    if m=='iv':
        plter.savePlot(outdir+m+"_3103_3104_noGR.pdf")
    
    
    plter.addPlotFromFile("3104_UL_diode_big_GR/3104_UL_diode_big_GR_2020-11-06_2."+m,
                               label="3104 (120 µm), GR, $V_{fb}=-5V$",  linestyle=linestyle, color="tab:green")
        
    plter.addPlotFromFile("3103_UL_diode_big_GR/3103_UL_diode_big_GR_2020-11-06_1."+m,
                               label="3103 (120 µm), GR, $V_{fb}=-2V$",  linestyle=linestyle, color="tab:orange")
    
    
    plter.savePlot(outdir+m+"_3103_3104.pdf")
    

##############





def selection(x,y):
    return x[x<-50],y[x<-50]

iv_plotter.addPlotFromFile("2012_UL_diode_big_GR/2012_UL_diode_big_GR_2020-11-06_1.iv",
                           selection=selection,
                           label="2012 (200 µm), GR",  linestyle=linestyle, color="tab:orange")



iv_plotter.savePlot(outdir+"IV_2012_select.pdf")

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



