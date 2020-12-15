#!/usr/bin/python3

from annealings import getFullSet
from matplotlib import pyplot as plt
import os 
from fitting import AnnealingFitter
import numpy as np
import styles

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

alldiodes = ["1102", "1003", "1002", "2102", "2003", "2002"]

for d in alldiodes:
    

    x,y,ivs, xerr, yerr = getFullSet(d,debugall=True,debugdir=debugdir,useIVGP=False)
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

depVolt['1102'].plot(label='1102, 1.5 E15 neq/cm$^2$',marker=markers )#,linestyle='None')
depVolt['1003'].plot(label='1003, 1.0 E15 neq/cm$^2$',marker=markers )#,linestyle='None')
depVolt['1002'].plot(label='1002, 0.65 E15 neq/cm$^2$',marker=markers)#,linestyle='None')

plt.title("300µm")
plt.xlabel("t [min]")
plt.ylabel("$-U_{dep} [V]$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"dep_voltage_300.pdf")
plt.close()

plt.title("300µm")
ivAtDep['1102'].plot(label='1102, 1.5 E15 neq/cm$^2$',marker=markers)
ivAtDep['1003'].plot(label='1003, 1.0 E15 neq/cm$^2$',marker=markers)
ivAtDep['1002'].plot(label='1002, 0.65 E15 neq/cm$^2$',marker=markers)

plt.title("300µm")
plt.xlabel("t [min]")
plt.ylabel("$-I\ (U_{dep}) [A]$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"leakage_at_dep_300.pdf")
plt.close()


ivHiDep['1102'].plot(label='1102, 1.5 E15 neq/cm$^2$',marker=markers)
ivHiDep['1003'].plot(label='1003, 1.0 E15 neq/cm$^2$',marker=markers)
ivHiDep['1002'].plot(label='1002, 0.65 E15 neq/cm$^2$',marker=markers)

plt.title("300µm")
plt.xlabel("t [min]")
plt.ylabel("$-I\ (1.1 \cdot U_{dep}) [A]$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"leakage_at_hidep_300.pdf")
plt.close()


ivFixed['1102'].plot(label='1102, 1.5 E15 neq/cm$^2$',marker=markers)
ivFixed['1003'].plot(label='1003, 1.0 E15 neq/cm$^2$',marker=markers)
ivFixed['1002'].plot(label='1002, 0.65 E15 neq/cm$^2$',marker=markers)


plt.title("300µm")
plt.xlabel("t [min]")
plt.ylabel("$-I\ (-"+str(fixedvoltage)+"V) [A]$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"leakage_at_"+str(fixedvoltage)+"_300.pdf")
plt.close()



####### 200 µm


depVolt['2102'].plot(label='2102, 2.5 E15 neq/cm$^2$',marker=markers)
depVolt['2003'].plot(label='2003, 1.5 E15 neq/cm$^2$',marker=markers)
depVolt['2002'].plot(label='2002, 1.0 E15 neq/cm$^2$',marker=markers)

plt.title("200µm")
plt.xlabel("t [min]")
plt.ylabel("$-U_{dep} [V]$")
plt.xscale('log')
plt.legend()
plt.savefig(outdir+"dep_voltage_200.pdf")
plt.close()



plt.title("200µm")
ivAtDep['2102'].plot(label='2102, 2.5 E15 neq/cm$^2$',marker=markers)
ivAtDep['2003'].plot(label='2003, 1.5 E15 neq/cm$^2$',marker=markers)
ivAtDep['2002'].plot(label='2002, 1.0 E15 neq/cm$^2$',marker=markers)

plt.xlabel("t [min]")
plt.ylabel("$-I\ (U_{dep}) [A]$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"leakage_at_dep_200.pdf")
plt.close()



plt.title("200µm")
ivFixed['2102'].plot(label='2102, 2.5 E15 neq/cm$^2$',marker=markers)
ivFixed['2003'].plot(label='2003, 1.5 E15 neq/cm$^2$',marker=markers)
ivFixed['2002'].plot(label='2002, 1.0 E15 neq/cm$^2$',marker=markers)

plt.xlabel("t [min]")
plt.ylabel("$-I\ (-"+str(fixedvoltage)+"V) [A]$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"leakage_at_"+str(fixedvoltage)+"_200.pdf")
plt.close()

