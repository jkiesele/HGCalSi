#!/usr/bin/python3


from argparse import ArgumentParser
from fileIO import readTempLog
from numba import jit
import alpha_calc
import tqdm

parser = ArgumentParser('convert recorded ')
parser.add_argument('inputFile')
parser.add_argument('temperature')
parser.add_argument('-u',help="uncertainty (0.2C) up (u) or down (d)",default="")
args = parser.parse_args()

if not args.u =="" and not args.u == "u" and not args.u == "d":
    parser.print_help()
    exit
    
attemp = float(args.temperature)

@jit(nopython=True)   
def conv(times,temps, unc=""):
    out=[]
    outtemps=[]
    for i in range(len(times)-1):
        out.append(1./60.)
        if unc=="":
            outtemps.append(temps[i])
        elif unc=="u":
            outtemps.append(temps[i]+0.2)
        elif unc=="d":
            outtemps.append(temps[i]-0.2)
    return out, outtemps

logt,logtemp = readTempLog(args.inputFile,readLast=False)#only one allowed

print(logt)

x,y = conv(logt,logtemp,args.u)
totaltime=0.

for t,T in tqdm.tqdm(zip(x,y),total=len(x)):
    totaltime += alpha_calc.equiv_annealing_time(T, t, attemp)
    
    
print(totaltime,'minutes to reach alpha of', alpha_calc.alpha(attemp, totaltime))
