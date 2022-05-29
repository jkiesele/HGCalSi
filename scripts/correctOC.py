#!/usr/bin/python3

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('inputFile')
#parser.add_argument('oldOCC')
parser.add_argument('--newOCC',default=-1)
#parser.add_argument('oldOCG')
parser.add_argument('--newOCG',default=-1)
parser.add_argument('--overwrite',default=False,action='store_true')
args = parser.parse_args()

newOCC = float(args.newOCC)
newOCG = float(args.newOCG)

if args.newOCC < 0:#attempt defaults
    from fileIO import fileReader
    fr = fileReader(mode="CV",return_freq=True)
    _,_,_, freq = fr.read(args.inputFile)
    assert freq == 63 or freq == 500 or freq == 10000
    if freq == 63:
        newOCC = 4.9923e-11
        newOCG = -0.00000000041296
    if freq == 500:
        newOCC = 4.9838e-11
        newOCG = 0.00000000033576
    if freq == 10000:
        newOCC = 4.9756e-11
        newOCG = 0.0000000020444
        
    print('using frequency',freq)
    #exit()

#read OC
with open(args.inputFile) as f:
    triggered=False
    for l in f:
        if ":LCR open correction" in l:
            triggered=True
            continue
        if triggered:
            l = l[:-1]
            l = l.split(',')
            print()
            oldOCC = float(l[0])
            oldOCG = float(l[1])
            break



deltaC = oldOCC - newOCC
deltaS = oldOCG - newOCG

import os 
if args.overwrite:
    os.system('cp '+args.inputFile + ' '+ args.inputFile+'.backup')

out=["CONVERTED FROM DIFFERENT OPEN CORRECTION\n"]
out.append("OLD: "+str(oldOCC)+", "+str(oldOCG)  +" NEW: "+str(newOCC)+", "+str(newOCG)+'\n')
start=False
with open(args.inputFile) as f:
    for l in f:
        #just copy
        if l=="CONVERTED FROM DIFFERENT OPEN CORRECTION\n":
            print("ALREADY CONVERTED, DON'T CONVERT TWICE")
            exit()
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

outfile = args.inputFile[:-3]+'_c.cv'
if args.overwrite:
    outfile = args.inputFile
print(outfile)
#print(out)
#exit()
with open(outfile,'w') as f:
    for l in out:
        f.write(l)
    
    
    
    
    