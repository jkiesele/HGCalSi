#!/usr/bin/python3

from annealings import getFullSet
from matplotlib import pyplot as plt
import os 

markers='x'
fixedvoltage=600
outdir="full_annealing"
os.system('mkdir -p '+outdir)
outdir+='/'
debugdir=outdir+'debug'
os.system('mkdir -p '+debugdir)
debugdir+='/'

x={}
y={}
ivAtDep={}
ivHiDep={}
ivFixed={}

for d in ["1102", "1003", "1002", "2102", "2003", "2002"]:
    

    x[d],y[d],ivs = getFullSet(d,debugall=True,debugdir=debugdir)
    ivAtDep[d] = ivs.eval(y[d])
    ivHiDep[d] = ivs.eval(y[d]*1.1)
    fiv = ivs.eval([- fixedvoltage for _ in range(len(y[d]))])
    print("fixed I for ", d, fiv)
    ivFixed[d] = fiv

plt.close()
plt.plot(x['1102'],-y['1102'],label='1102, 1.5 E15 neq/cm$^2$',marker=markers)
plt.plot(x['1003'],-y['1003'],label='1003, 1.0 E15 neq/cm$^2$',marker=markers)
plt.plot(x['1002'],-y['1002'],label='1002, 0.65 E15 neq/cm$^2$',marker=markers)

plt.xlabel("t [min]")
plt.ylabel("$-U_{dep}$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"dep_voltage_300.pdf")
plt.close()

plt.plot(x['1102'],-ivAtDep['1102'],label='1102, 1.5 E15 neq/cm$^2$',marker=markers)
plt.plot(x['1003'],-ivAtDep['1003'],label='1003, 1.0 E15 neq/cm$^2$',marker=markers)
plt.plot(x['1002'],-ivAtDep['1002'],label='1002, 0.65 E15 neq/cm$^2$',marker=markers)

plt.xlabel("t [min]")
plt.ylabel("$-I\ (U_{dep})$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"leakage_at_dep_300.pdf")
plt.close()


plt.plot(x['1102'],-ivHiDep['1102'],label='1102, 1.5 E15 neq/cm$^2$',marker=markers)
plt.plot(x['1003'],-ivHiDep['1003'],label='1003, 1.0 E15 neq/cm$^2$',marker=markers)
plt.plot(x['1002'],-ivHiDep['1002'],label='1002, 0.65 E15 neq/cm$^2$',marker=markers)

plt.xlabel("t [min]")
plt.ylabel("$-I\ (1.1 \cdot U_{dep})$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"leakage_at_hidep_300.pdf")
plt.close()


plt.plot(x['1102'],-ivFixed['1102'],label='1102, 1.5 E15 neq/cm$^2$',marker=markers)
plt.plot(x['1003'],-ivFixed['1003'],label='1003, 1.0 E15 neq/cm$^2$',marker=markers)
plt.plot(x['1002'],-ivFixed['1002'],label='1002, 0.65 E15 neq/cm$^2$',marker=markers)


plt.xlabel("t [min]")
plt.ylabel("$-I\ (-"+str(fixedvoltage)+"V)$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"leakage_at_"+str(fixedvoltage)+"_300.pdf")
plt.close()



####### 200 µm


plt.plot(x['2102'],-y['2102'],label='2102, 2.5 E15 neq/cm$^2$',marker=markers)
plt.plot(x['2003'],-y['2003'],label='2003, 1.5 E15 neq/cm$^2$',marker=markers)
plt.plot(x['2002'],-y['2002'],label='2002, 1.0 E15 neq/cm$^2$',marker=markers)

plt.xlabel("t [min]")
plt.ylabel("$-U_{dep}$")
plt.xscale('log')
plt.legend()
plt.savefig(outdir+"dep_voltage_200.pdf")
plt.close()



plt.plot(x['2102'],-ivAtDep['2102'],label='2102, 2.5 E15 neq/cm$^2$',marker=markers)
plt.plot(x['2003'],-ivAtDep['2003'],label='2003, 1.5 E15 neq/cm$^2$',marker=markers)
plt.plot(x['2002'],-ivAtDep['2002'],label='2002, 1.0 E15 neq/cm$^2$',marker=markers)

plt.xlabel("t [min]")
plt.ylabel("$-I\ (U_{dep})$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"leakage_at_dep_200.pdf")
plt.close()



plt.plot(x['2102'],-ivFixed['2102'],label='2102, 2.5 E15 neq/cm$^2$',marker=markers)
plt.plot(x['2003'],-ivFixed['2003'],label='2003, 1.5 E15 neq/cm$^2$',marker=markers)
plt.plot(x['2002'],-ivFixed['2002'],label='2002, 1.0 E15 neq/cm$^2$',marker=markers)

plt.xlabel("t [min]")
plt.ylabel("$-I\ (-"+str(fixedvoltage)+"V)$")
plt.xscale('log')

plt.legend()
plt.savefig(outdir+"leakage_at_"+str(fixedvoltage)+"_200.pdf")
plt.close()

