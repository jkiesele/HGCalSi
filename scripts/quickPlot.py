#!/usr/bin/python3

from argparse import ArgumentParser
import os
parser = ArgumentParser()
parser.add_argument('inputDir')
parser.add_argument('number', default="-1")
args = parser.parse_args()
no=int(args.number)
if no < 0:
    no=""
else:
    no=str(no)


from plotting import curvePlotter
from fitting import GPFitter, fittedIV

from matplotlib import pyplot as plt

globalpath=os.getenv("DATAPATH")+'/'

cv_plotter = curvePlotter(mode="CVs",path=globalpath)
cv_plotter.addPlotFromFile(args.inputDir+"/*"+no+".cv",
                               label="test")

plt.show()
plt.close()

iv_plotter = curvePlotter(mode="IV",path=globalpath)
iv_plotter.addPlotFromFile(args.inputDir+"/*"+no+".iv",
                               label="test")

plt.show()


