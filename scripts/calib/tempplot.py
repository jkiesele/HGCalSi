#!/usr/bin/python3

from fileIO import readTempLog
from argparse import ArgumentParser

from matplotlib import pyplot as plt
import numpy as np

parser = ArgumentParser('convert recorded ')
parser.add_argument('inputFile')
args = parser.parse_args()

from sympy.solvers import solve
from sympy import Symbol

def R2T_PTX_ITS90(Rlist,R0=1000):
    out=[]
    for R in Rlist:
        t = Symbol('t')
        A = 3.9083E-3
        B = -5.7750E-7
        C = 0.
        if R < R0 : C = -4.183E-12
        T = solve(R-R0*(1+A*t+B*t*t+C*(t-100)*t*t*t),t)
        out.append(T[0])
    return out



def calibADCCounts(counts):
    #return counts
    return 2.7003882937222283e-06* counts**2 + 1.005*0.3770691962172657* counts + 4900.17006022754041 +160

x,y,z = readTempLog(args.inputFile, calib=True)
x=x

#y=y[x<2000]
#z=z[x<2000]
#x=x[x<2000]

#
plt.plot(x,y)
plt.xlabel("t [min]")
plt.ylabel("T [deg C]")

plt.twinx()
plt.plot(x,calibADCCounts(z))
plt.xlabel("t [min]")
plt.ylabel("T [deg C]")
plt.show()

plt.scatter(calibADCCounts(z),y)
plt.show()

#
#

exit()


R=[997.5,
    1022,
    1035,
    1047,
    1055,
    1069.5,
    1080.5,
    1091.5,
    1102.5,
    1113.5,
    1125,
    1142.5,
    1167,
    1187, #taken up to including this one
    1200,
    1215.5,
    1219.5,
    1228.5,
    1233,
    1242.5,
    1256.5,
    1271,
    1281.5,
    1291,
    1304.5,
    1313.5,
    1324.5,
    1334.5,
    1345.75,
    1352.5,
    1363,
    1374.5,
    1386.5,
    1395.5,
    1405.5,
    
    ]

R=np.array(R)
R-=2.3 #cables

def mean_meas(time,adc):
    adcmean=[]
    adctmp=0
    count=0
    for i in range(len(time)):
        if (i and time[i]==10) :
            adcmean.append(adctmp/count)
            count=0
            adctmp=0
        adctmp+=adc[i]
        count+=1

    return np.array(adcmean)


adcm = mean_meas(x, z)
steps = np.arange(len(adcm))


plt.plot(steps,adcm)
plt.xlabel("step")
plt.ylabel("counts")
plt.twinx()
print(len(R), len(steps))
plt.plot(steps,R)

plt.show()



plt.scatter(adcm,R)
plt.show()

temps=R2T_PTX_ITS90(R)
plt.scatter(adcm,temps)
plt.show()




from scipy.optimize import curve_fit


def objective(x, a, b, c):
    return a * x**2 + b*x + c

popt, _ = curve_fit(objective, adcm,temps)
# summarize the parameter values
a, b, c = popt
print((a, b, c))


plt.scatter(adcm,temps)
plt.scatter(adcm,objective(adcm,a,b,c))
plt.show()










