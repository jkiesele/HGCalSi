#!/usr/bin/python3

from argparse import ArgumentParser
import os
parser = ArgumentParser()
parser.add_argument('inputDir')
parser.add_argument('--const_cap', default=None, type=float)
parser.add_argument('--low_start', default=None, type=float)
parser.add_argument('--low_end', default=None, type=float)
parser.add_argument('--high_start', default=None, type=float)
parser.add_argument('--high_end', default=None, type=float)
args = parser.parse_args()

def invert(a):
    if a is not None:
        return -a
    return a
           

'''
Use two approaches:
- linear fit of two curves
- smoothed inflection point of two skewed curves
'''
low_start=invert(args.low_start)
low_end=invert(args.low_end)
high_start=invert(args.high_start)
high_end=invert(args.high_end)

        
print(low_start, low_end, high_start, high_end)

from matplotlib import pyplot as plt
import os 
from fitting import AnnealingFitter
import numpy as np
import styles
from diodes import diodes
from tools import getDepletionVoltage

from plotting import curvePlotter
from fitting import GPFitter, fittedIV

from matplotlib import pyplot as plt

globalpath=os.getenv("DATAPATH")+'/'

cv_plotter = curvePlotter(mode="CVs",path=globalpath)
cv_plotter.addPlotFromFile(args.inputDir+"/*.cv",
                               label="test")

cv_plotter.showPlot()
print('fit rising edge from to (positive)')
high_end= - float(input())
high_start= - float(input())
print('fit constant from to')
low_end = - float(input())
low_start = - float(input())


v = getDepletionVoltage(globalpath+args.inputDir+"/*.cv",
                        min_x=-900,
                        const_cap=args.const_cap,
                        debug=True,
                        low_start = low_start,
                        low_end = low_end,
                        high_start = high_start,
                        high_end = high_end,
                        savedatapath=globalpath+args.inputDir+"/extracted.dv")

