#!/usr/bin/python3

from fileIO import readTempLog
from argparse import ArgumentParser

from matplotlib import pyplot as plt

parser = ArgumentParser('convert recorded ')
parser.add_argument('inputFile')
args = parser.parse_args()




x,y = readTempLog(args.inputFile)

plt.plot(x,y)
plt.xlabel("t [s]")
plt.ylabel("T [deg C]")
plt.show()