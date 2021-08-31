#!/usr/bin/python3

from argparse import ArgumentParser
import numpy as np
import alpha_calc
import math
parser = ArgumentParser()
parser.add_argument('radtime')
args = parser.parse_args()
radtime=float(args.radtime)

import matplotlib.pyplot as plt


#now calculate average annealing time assuming linearly increasing radiation damage per second (because why not)

rising_annealing=np.array([i for i in range(int(radtime*60))],dtype='float32') #again in minutes
falling_annealing=np.array([i+rising_annealing[-1] for i in range(30*60)],dtype='float32')#30 minutes cool off

rising_annealing = rising_annealing/60
falling_annealing= falling_annealing/60

#back to minutes


def rising(t_in_minutes):
    return 45. - 25.*math.exp( - (t_in_minutes)/3.5 ) 

def falling(t_in_minutes,last_risen):
    return (20.+(last_risen-45.))*math.exp( - (t_in_minutes - radtime)/8 ) + 25

xr = [i for i in rising_annealing]
xf = [i for i in falling_annealing]

rise = np.array([rising(i) for i in xr], dtype='float32')

fall = np.array([falling(i,rise[-1]) for i in xf], dtype='float32')


x=xr+xf
y=np.concatenate([rise,fall],axis=0)

plt.plot(x,y)
plt.xlabel("minutes")
plt.xlabel("temperature in reactor")
plt.show()


#get full equiv annealing time for all in rise
time_bins = []
for T in y:
    time_bins.append(alpha_calc.equiv_annealing_time(T, 1/60., 60.))#per second
    
fraction_irradiated = xr/np.max(xr)
fraction_irradiated = np.concatenate([fraction_irradiated, falling_annealing*0.+1],axis=0)#to have full range

totaltime = 0
for t,f in zip(time_bins,fraction_irradiated):
    totaltime +=  t*f

print('total time at 60˚C',totaltime)






