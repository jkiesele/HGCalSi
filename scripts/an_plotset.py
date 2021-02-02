#!/usr/bin/python3


from tools import plotFullSet
import styles
import os




for time in ['10', '30'] :
    outdir="new_irradiated_ann_"+time+"min/"
    plotFullSet(time+"min",outdir,identifier="UR_2D_diode_big")
    
    
for time in ['73', '103', '153', '247', '378', '645', '1352', '2919'] :
    outdir="irradiated_ann_"+time+"min/"
    plotFullSet(time+"min",outdir)

