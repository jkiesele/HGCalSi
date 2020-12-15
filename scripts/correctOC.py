#!/usr/bin/python3

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('inputFile')
parser.add_argument('oldOCC')
parser.add_argument('newOCC')
parser.add_argument('oldOCG')
parser.add_argument('newOCG')
args = parser.parse_args()

deltaC = float(args.oldOCC) - float(args.newOCC)
deltaS = float(args.oldOCG) - float(args.newOCG)

import os 
os.system('cp '+args.inputFile + ' '+ args.inputFile+'.backup')

out=["CONVERTED FROM DIFFERENT OPEN CORRECTION\n"]
out.append("OLD: "+args.oldOCC+", "+args.oldOCG  +" NEW: "+args.newOCC+", "+args.newOCG+'\n')
start=False
with open(args.inputFile) as f:
    for l in f:
        #just copy
        if l=="CONVERTED FROM DIFFERENT OPEN CORRECTION\n":
            print("ALREADY CONVERTED, DON'T CONVERT TWICE")
            exit
        if l=="BEGIN\n":
            start=True
            out.append(l)
            continue
        if l=="END\n":
            break
        
        if not start:
            out.append(l)
            continue
        
        l=l[:-1]
        vals = l.split('\t')
        #print(vals)
        C = float(vals[1]) + deltaC
        S = float(vals[2]) + deltaS
        out.append(vals[0]+'\t'+str(C)+'\t'+str(S)+'\t'+vals[3]+'\t'+vals[4]+'\n')
        
out.append('END\n')

#print(out)
with open(args.inputFile,'w') as f:
    for l in out:
        f.write(l)
    
    
    
    
    