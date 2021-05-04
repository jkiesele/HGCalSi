#!/usr/bin/python3

import os
import matplotlib.pyplot as plt
from plotting import curvePlotter
from pointset import pointSetsContainer, pointSet, loadAnnealings

'''
3008_UL_3D_diode_big_ann_243min_captests

'''

datadir=os.getenv("DATAPATH")+'/'
outdir=os.getenv("DATAOUTPATH")+'/captests/'
os.system('mkdir -p '+outdir)

def setAxisDep(ylabel="-$U_{depl}$ [V]"):
    plt.xlabel("U [V]", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    

def makeplot(diodestr):
    plt.close()
    folder = diodestr+"_captests/"
    sp = curvePlotter(mode="CVs", path=datadir, read_freq=True)
    for i in range(1,5):
    
        readfile = folder+"*_"+str(i)+".cv"
        try:
            f = sp.readFreq(readfile)
            fmtdiodestr = diodestr.replace("*","")
            sp.addPlotFromFile(readfile,
                               label=fmtdiodestr+' '+str(f)+' Hz')
        except:
            pass
    
    sp.labelAxes() 
    plt.legend()   
    diodestr = diodestr.replace("*","_")
    plt.savefig(outdir+diodestr+'_captest.pdf')
    
makeplot("3008_UL_3D_diode_big_ann_243*")
makeplot("3008_UL_3D_diode_big_ann_3128*")
makeplot("3003_UL_3D_diode_big_ann_3128*")
makeplot("2003_UL_diode_big_ann_2919")
makeplot("2002_UL_diode_big_ann_2919*")


