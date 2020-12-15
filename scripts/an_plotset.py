#!/usr/bin/python3


from tools import plotFullSet
import styles

globalpath="/Users/jkiesele/cern_afs/eos_hgsensor_testres/Results_SSD/CVIV/Diode_TS/"

for time in ['73', '103', '153', '247', '378', '645', '1352', '2919'] :
    outdir="irradiated_ann_"+time+"min/"
    plotFullSet(time+"min",outdir, globalpath)
