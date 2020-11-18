#!/usr/bin/python3

from fileIO import readTempLog

from matplotlib import pyplot as plt

x,y = readTempLog("LOG.TXT")

plt.plot(x/60.,y)
plt.show()