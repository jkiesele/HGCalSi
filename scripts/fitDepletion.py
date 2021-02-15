#!/usr/bin/python3

from argparse import ArgumentParser
import os
parser = ArgumentParser()
parser.add_argument('inputDir')
parser.add_argument('--rederive', default=False,  action='store_true')
parser.add_argument('--strictcheck', default=False,  action='store_true')
parser.add_argument('--batch', default=False,  action='store_true')
parser.add_argument('--outfile', default="extracted")
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

from fitting import  readDepletionData
from tools import getDepletionVoltage
import matplotlib.pyplot as plt
from plotting import curvePlotter

globalpath=os.getenv("DATAPATH")+'/'
outpath=os.getenv("DATAOUTPATH")+'/'
os.system('mkdir -p '+outpath+args.inputDir)
outpath = outpath+args.inputDir+"/"
outprefix = outpath+args.outfile

high_end= None
high_start= None
low_end = None
low_start = None
const_cap=None

if args.rederive or not os.path.isfile(outpath+args.outfile+".depl"):

    cv_plotter = curvePlotter(mode="CVs",path=globalpath)
    cv_plotter.addPlotFromFile(args.inputDir+"/*.cv",
                                   label="test")
    
    cv_plotter.showPlot()
    print('fit rising edge from to (positive)')
    print('fit constant from to')
    print('constant value (if any)')
    high_end= - float(input())
    high_start= - float(input())
    low_end = - float(input())
    low_start = - float(input())

    const_cap=input()
    if not const_cap == "":
        const_cap=float(const_cap)
    else:
        const_cap=None
else:
    d = readDepletionData(outpath,args.outfile+".depl")
    
    high_end= d['high_end']
    high_start= d['high_start']
    low_end = d['low_end']
    low_start = d['low_start']
    const_cap=d['const_cap']
    

v = getDepletionVoltage(globalpath+args.inputDir+"/*.cv",
                        min_x=-900,
                        const_cap=const_cap,
                        debug=True,
                        low_start = low_start,
                        low_end = low_end,
                        high_start = high_start,
                        high_end = high_end,
                        strictcheck=args.strictcheck,
                        savedatapath=outprefix+".depl",
                        interactive=not args.batch)


#smoothen IV curves and save as dicts for easy processing
plt.close()
ivpl = curvePlotter(mode="IV",path=globalpath)
ivpl.addPlotFromFile(args.inputDir+"/*.iv",label="temp")
xs,ys = ivpl.getXYSmooth()
plt.plot(-xs,-ys, label='smoothened')
plt.legend()
ivpl.savePlot(outprefix+"_iv.pdf",True)


