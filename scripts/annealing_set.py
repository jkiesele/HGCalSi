#!/usr/bin/python3

from annealings import getFullSet
from matplotlib import pyplot as plt
import os 
from fitting import AnnealingFitter
import numpy as np
import styles
from diodes import diodes

markers='x'
fixedvoltage=600
outdir="full_annealing"
os.system('mkdir -p '+outdir)
outdir+='/'
debugdir=outdir+'debug'
os.system('mkdir -p '+debugdir)
debugdir+='/'

class graph(object):
    def __init__(self, x,y, xerr=None, yerr=None):
        self.x=np.array(x)
        self.y=np.array(y)
        self.xerr=np.array(xerr)
        self.yerr=np.array(yerr)
        if yerr is None:
            self.yerr = np.zeros(self.x.shape) #pass #self.yerr = np.sqrt(10.**2+(self.y*0.025)**2)
        #print(self.xerr.shape)
        #print(self.yerr.shape)
        #exit()
        
    def plot(self,**kwargs):
        plt.errorbar(self.x,self.y, self.yerr, self.xerr,**kwargs)


ivAtDep={}
ivHiDep={}
ivFixed={}

depVolt={}
fittedAn={}

minima={}

alldiodes = ["3003","3007","3008","1102", "1003", "1002", "2102", "2003", "2002"]

for d in alldiodes:
    
    times = [ 30, 73,74,103,153,247,378,645,1352,2919]
    if d[0]=='3':
        times = [ 30, 74]

    x,y,ivs, xerr, yerr = getFullSet(d,debugall=True,debugdir=debugdir,useIVGP=False,times=times)
    an = AnnealingFitter()
    an.fit(x,-y,xerr)
    fittedAn[d]=an

    
    depVolt[d] = graph(x, -y, xerr, yerr)
    depVolt[d].plot()
    
    smoothx=np.arange(np.min(x),np.max(x), 1)
    fitted = an.DNeff(smoothx)
    minima[d]=smoothx[np.argmin(fitted)]
    plt.plot(smoothx,fitted)
    plt.xscale('log')
    plt.savefig(debugdir+'/smoothfit_'+d+'.pdf')
    
    ivAtDepy = ivs.eval(y)
    ivAtDep[d] = graph(x,-ivAtDepy,xerr)
    ivHiDepy = ivs.eval(y*1.1)
    ivHiDep[d] = graph(x,-ivHiDepy,xerr)
    fiv = ivs.eval([- fixedvoltage for _ in range(len(y))])
    ivFixedy = fiv
    ivFixed[d] = graph(x,-ivFixedy,xerr)
    
    #exit()
    
plt.close()
for i in range(len(alldiodes)):
    plt.plot(i, minima[alldiodes[i]], label=alldiodes[i],marker='o')
    
#plt.legend()
plt.ylabel("Annealing minimum [min]")
plt.xticks(range(len(alldiodes)),alldiodes)
plt.xlabel("Sensor number")
plt.savefig(outdir+"annealing_minima.pdf")
plt.close()


######## 120




for d in ["3003","3007","3008"]:
    depVolt[d].plot(label=diodes[d].label(),marker=markers )#,linestyle='None')

plt.title("120µm")
plt.xlabel("t [min]")
plt.ylabel("$-U_{dep} [V]$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"dep_voltage_120.pdf")
plt.close()



####### 300
for d in ["1102", "1003", "1002"]:
    depVolt[d].plot(label=diodes[d].label(),marker=markers )#,linestyle='None')

plt.title("300µm")
plt.xlabel("t [min]")
plt.ylabel("$-U_{dep} [V]$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"dep_voltage_300.pdf")
plt.close()

plt.title("300µm")
for d in ["1102", "1003", "1002"]:
    ivAtDep[d].plot(label=diodes[d].label(),marker=markers)

plt.title("300µm")
plt.xlabel("t [min]")
plt.ylabel("$-I\ (U_{dep}) [A]$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"leakage_at_dep_300.pdf")
plt.close()


for d in ["1102", "1003", "1002"]:
    ivHiDep[d].plot(label=diodes[d].label(),marker=markers)

plt.title("300µm")
plt.xlabel("t [min]")
plt.ylabel("$-I\ (1.1 \cdot U_{dep}) [A]$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"leakage_at_hidep_300.pdf")
plt.close()


for d in ["1102", "1003", "1002"]:
    ivFixed[d].plot(label=diodes[d].label(),marker=markers)


plt.title("300µm")
plt.xlabel("t [min]")
plt.ylabel("$-I\ (-"+str(fixedvoltage)+"V) [A]$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"leakage_at_"+str(fixedvoltage)+"_300.pdf")
plt.close()



####### 200 µm


for d in ["2102", "2003", "2002"]:
    depVolt[d].plot(label=diodes[d].label(),marker=markers)

plt.title("200µm")
plt.xlabel("t [min]")
plt.ylabel("$-U_{dep} [V]$")
plt.xscale('log')
plt.legend()
plt.savefig(outdir+"dep_voltage_200.pdf")
plt.close()



plt.title("200µm")

for d in ["2102", "2003", "2002"]:
    ivAtDep[d].plot(label=diodes[d].label(),marker=markers)

plt.xlabel("t [min]")
plt.ylabel("$-I\ (U_{dep}) [A]$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"leakage_at_dep_200.pdf")
plt.close()



plt.title("200µm")

for d in ["2102", "2003", "2002"]:
    ivFixed[d].plot(label=diodes[d].label(),marker=markers)

plt.xlabel("t [min]")
plt.ylabel("$-I\ (-"+str(fixedvoltage)+"V) [A]$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"leakage_at_"+str(fixedvoltage)+"_200.pdf")
plt.close()

