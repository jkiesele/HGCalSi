#!/usr/bin/python3


from argparse import ArgumentParser
from fileIO import readTempLog
from numba import jit

parser = ArgumentParser('convert recorded ')
parser.add_argument('inputFile')
parser.add_argument('outputFile')
parser.add_argument('-u',help="uncertainty (0.2C) up (u) or down (d)",default="")
args = parser.parse_args()

if not args.u =="" and not args.u == "u" and not args.u == "d":
    parser.print_help()
    exit

@jit(nopython=True)   
def conv(times,temps, unc=""):
    out=[]
    outtemps=[]
    for i in range(len(times)-1):
        out.append((times[i+1]-times[i])/60.)
        if unc=="":
            outtemps.append(temps[i])
        elif unc=="u":
            outtemps.append(temps[i]+0.2)
        elif unc=="d":
            outtemps.append(temps[i]-0.2)
    return out, outtemps

logt,logtemp = readTempLog(args.inputFile)

x,y = conv(logt,logtemp,args.u)

with open(args.outputFile, 'w') as f:
    for l in zip(x,y):
        f.write(str(l[0])+' '+str(l[1])+'\n')
        
exit()
        
with open(args.outputFile+"calib.txt", 'w') as f:
    for i in range(3600):
        f.write(str(1./60.)+' 60\n')