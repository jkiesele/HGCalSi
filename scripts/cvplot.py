#!/usr/bin/python3

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('inputFile')
parser.add_argument('--Cp', default=False,  action='store_true')
args = parser.parse_args()

import styles
from matplotlib import pyplot as plt
from fileIO import fileReader
from plotting import curvePlotter
import os

mode = "CVs"
if args.Cp:
    mode = "CV"

cv_plotter = curvePlotter(mode=mode)
cv_plotter.addPlotFromFile(args.inputFile,
                               label="measured")

fileReader=fileReader(mode="CV",return_freq=True)
_,_,_, freq = fileReader.read(args.inputFile)
filename = os.path.basename(args.inputFile)
#read expected end capacitance form diode
diodestr=""
try:
    diodestr = filename[0:4]
    from diodes import diodes
    cideal = diodes[diodestr].Cideal()
    x = cv_plotter.getXY()[0]
    linex = [-x[0],-x[-1]]
    liney = [1/cideal**2,1/cideal**2]
    plt.plot(linex,liney,label=r'$C_{end}$ (ideal)')
    diodestr = diodes[diodestr].label()
except:
    pass
title = diodestr+'  --  '+str(freq)+' Hz'
plt.title(title)
cv_plotter.showPlot()