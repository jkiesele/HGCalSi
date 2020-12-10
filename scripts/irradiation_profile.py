#!/usr/bin/python3

from argparse import ArgumentParser
import math
parser = ArgumentParser()
parser.add_argument('radtime')
args = parser.parse_args()
radtime=int(args.radtime)

import matplotlib.pyplot as plt


rise = [45 - 25*math.exp( - (i)/3.5 ) for i in range(radtime)]
xr = [i for i in range(radtime)]

fall = [(20.+(rise[-1]-45.))*math.exp( - (i- radtime)/8 ) + 25 for i in range(radtime,radtime+30)]
xf = [i+radtime for i in range(30)]

x=xr+xf
y=rise+fall

plt.plot(xr+xf,rise+fall)
plt.show()

for i in zip(x,y):
    print(i[0]*60, i[1])