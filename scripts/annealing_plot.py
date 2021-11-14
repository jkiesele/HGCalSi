#!/usr/bin/python3

import os
import styles
import matplotlib.pyplot as plt
from tools import convert60CTo0C, convert60CTom30C, convert60CTo15C, convert60Cto21C
import numpy as np
from pointset import pointSetsContainer, pointSet, loadAnnealings
from diodes import diodes
import pandas
import math
datadir=os.getenv("DATAOUTPATH")

os.system('mkdir -p '+datadir+'/annealing_plots')
outdir = datadir+'/annealing_plots/'

def minat60ToDaysAt0(t):
    return convert60CTo0C(t/(60*24.))

def minat60ToMonthsAtm30(t):
    return convert60CTom30C(t) /(60.*24.*30.4166)

def minat60ToDaysAt15(t):
    return convert60CTo15C(t)/(60*24.)

def minat60ToDaysAt21(t):
    return convert60Cto21C(t)/(60*24.)

def identity(x):
    return x



addminus30=False


def combineDiodeStrings(ds):
    labels = []
    for d in ds:
        l=d.paperlabel(False)
        if not l in labels:
            labels.append(l)
            
    return " ".join([l for l in labels])+' ' +ds[0].radstr()
    
    

def newplot():
    plt.close()
    height = 5.5
    if addminus30:
        height+=1
    fig,ax = plt.subplots(figsize=(6, height))
    return ax

addout=""
if addminus30:
    addout +="withm30"
#for func, altxlabel, addout in zip(
#    [minat60ToDaysAt21, minat60ToDaysAt15, minat60ToDaysAt0, minat60ToMonthsAtm30],
#    ["time (21˚C) [d]" , "time (15˚C) [d]" , "time (0˚C) [d]", "time (-30˚C) [months]"],
#    ["_21C",             "_15C",             "_0C",            "_-30C"]
#    ):

    

def cosmetics(x, ax, ylabel="-$U_{depl}$ [V]"):
    
    
    plt.legend()
    plt.xscale('log')
    plt.xlabel("time (60˚C) [min]", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    sax = ax.secondary_xaxis('top', functions=(minat60ToDaysAt21, identity))
    sax.set_xlabel("time (21˚C) [d]")
    sax = ax.secondary_xaxis(1.2, functions=(minat60ToDaysAt0, identity))
    sax.set_xlabel("time (0˚C) [d]")
    if addminus30:
        sax = ax.secondary_xaxis(1.4, functions=(minat60ToMonthsAtm30, identity))
        sax.set_xlabel("time (-30˚C) [months]")
    
    plt.xscale('log')
    
    plt.tight_layout()
    

pointsets, kneepointset= loadAnnealings()



def XCheckNEff(t, g_a, tau_a, g_y, tau_y, N_C, phi, NEff0):
        
        
        N_A = phi*g_a * np.exp(-t / tau_a)
        N_Y = g_y*phi * (1. - 1./(1 + t/tau_y))
        #print(N_A.shape, N_Y.shape)
        return N_A + N_C + N_Y + NEff0


'''
["3007_UL","2102_UL","2102_UR",
             "3008_UL","2003_UL","2003_UR","1102_UL", "1102_UR",
             "2002_UL","2002_UR","1003_UL","1003_UR",
             "3003_UL"]:
'''


from fitting import AnnealingFitter, SimpleAnnealingFitter




def addfitted(dstr):
    
    
    print(dstr)
    withinSetCorrelation=0.0
    
    from fitting import MeasurementSet, MeasurementWithCovariance, Covariance

    xyerssdatas = pointsets.getXYsDiodes("NEff",dstr)
    
    #make a measurement out of each
    measurements=MeasurementSet()
    for d in xyerssdatas:
        m = MeasurementWithCovariance(
            d, withinSetCorrelation
            )
        measurements.addMeasurement(m)
        
    fitdata = measurements.getCombinedData(addfluence_unc=0.00001,correlatesyst=True)
    t = fitdata['x']
    terr = fitdata['xerr']
    NEff = fitdata['y']
    NEffErr = fitdata['cov']
    
    #print([a.shape for a in [t,terr,NEff,NEffErr]])
    
    diodes = [d['diode'] for d in xyerssdatas]
    #check if they are the same
    for dd in diodes:
        for ddd in diodes:
            if ddd.rad!=dd.rad:
                raise ValueError("not same fluence diodes")
            
    diode = xyerssdatas[0]['diode']
    
    '''
    'diode': ps.diode(),
    't': x,
    'terr':xerrs,
    'y':y,
    'yerr':yerrs
    '''
    splits=None
    #splits = np.array([0]+[+len(d['t']) for d in  xyerssdatas])
    #splits = np.cumsum(splits,axis=0)
    #print('splits',splits)
    #t = np.concatenate([d['t'] for d in  xyerssdatas],axis=0)
    #NEff = np.concatenate([d['y'] for d in  xyerssdatas],axis=0)
    phi = np.ones_like(t)*diode.rad
    #terr = np.concatenate([d['terr'] for d in  xyerssdatas],axis=0)
    #NEffErr = np.concatenate([np.squeeze(d['yerr']) for d in  xyerssdatas],axis=0)
    #
    #
    NEff0 = np.zeros_like(t) #include it in the fit
    #
    #NEffErr = np.sign(NEffErr)*np.sqrt(NEffErr**2 + (0.1*NEff)**2) # add 10% fluence uncertainty
    #
    #
    #
    
    #print('t',t.shape, 'terr',terr.shape, 'NEff',NEff.shape, 'NEffErr',NEffErr.shape, 'phi',phi.shape, 
    #      'NEff0',NEff0.shape)
    
    cv = Covariance()
    cv.set(NEffErr)
    corrs = cv.getCorrelations()
    #for printing
    cv.set(corrs)
    print(cv)
    print(t)
    xpi = np.arange(corrs.shape[0])
    xp = np.concatenate([xpi for _ in np.arange(corrs.shape[0])],axis=0)
    yp = np.concatenate([np.expand_dims(xpi,axis=1)  for _ in np.arange(corrs.shape[0])],axis=1)
    yp = np.reshape(yp,[-1])
    #print(xp.shape)
    #plt.close()
    #plt.scatter(xp, yp, c=np.reshape(corrs,[-1]), marker='o')
    #plt.show()
    #exit()
    fitter = None
    if False:
        fitter = AnnealingFitter(useodr=False)
        fitter.setInputAndFit(t, terr, NEff, NEffErr, phi, NEff0, splits)
    else:
        fitter = SimpleAnnealingFitter(x=t, y=NEff, xerr=terr, ycov=NEffErr)
        fitter.fit(diode.rad,useodr=False)
        
    print('done fitting',dstr)
    return fitter,diodes


##########new fits

#do this for each sample individually
samplesperfluence = []
samplesperfluence += ["1002_UL","1002_UR"] +\
 ["2002_UL","2002_UR","1003_UL","1003_UR"] +\
   ["2003_UL","2003_UR","1102_UL", "1102_UR"] +\
    ["2102_UL","2102_UR"] +\
    ["3008_UL"] +\
    ["3007_UL"] +\
    ["3003_UL"] 
    
samplesperfluence = [[a] for a in samplesperfluence]

samplesperfluence = [["1002_UL","1002_UR"], 
                     ["2002_UL","2002_UR"],
                     ["1003_UL","1003_UR"],
                     ["2003_UL","2003_UR"],
                     ["1102_UL", "1102_UR"],
                     ["2102_UL","2102_UR"],
                     ["3008_UL"],
                     ["3007_UL"],
                     ["3003_UL"],
                     ### combined
                     #["2002_UL","2002_UR","1003_UL","1003_UR"],
                     #["2003_UL","2003_UR", "1102_UL", "1102_UR"]
                     ]

samplesperfluence = [["1002_UL","1002_UR"], 
                     ["2002_UL","2002_UR"],
                     ["1003_UL","1003_UR"],
                     ["2003_UL","2003_UR"],
                     ["1102_UL", "1102_UR"],
                     ["2102_UL","2102_UR"],
                     ["3008_UL"],
                     ["3007_UL"],
                     ["3003_UL"],
                     ]

allparas = []
for i, samples in enumerate(samplesperfluence):
    ax = newplot()
    xs = pointsets.addToPlot("NEff", 
                             samples#+
                            # ["2003_UL"]+#,"2003_UR","1102_UL", "1102_UR"]+
                            # ["2102_UL"]#,"2102_UR"]
                             ,
                             #labels=samples # len(samples)*[""]#+
                            # ["UL, 200µm"]+#,"UR, 200µm","UL, 300µm","UR, 300µm"]+
                            # ["UL, 200µm"]#,"UR, 200µm"]
                             #,
                             add_rel_y_unc=0.)

    fitter,thisdiodes = addfitted(samples)
    
    fparas = fitter.getParameters()
    fparas['ID']=combineDiodeStrings(thisdiodes) #diodes[samples[0][0:4]].paperlabel(False)
    fparas['fluence']=thisdiodes[0].rad
    fparas['NEff,0']=thisdiodes[0].NEff_norad()
    fparas['material']=thisdiodes[0].material_str()
    fparas['thickness']=thisdiodes[0].thickness
    print(samples)
    
    print(fparas)
    
    fitter.collapseNC()
    
    styles.setstyles()
    
    if fparas['N_C+N_{Eff,0}'] is list:
        fparas['N_C+N_{Eff,0}']=fparas['N_C+N_{Eff,0}'][0]
    
    x = np.arange(1.,3000.)
    for j in range(len(samples)):
        neff = fitter.NEff(x, 
                            diodes[samples[j][0:4]].rad, 
                            0.,
                            fparas['N_C+N_{Eff,0}'])
        plt.plot(x, neff)
        neffxceck = XCheckNEff(x,  fparas['g_a'], fparas['tau_a'], 
                               fparas['g_y'], fparas['tau_y'], 
                               fparas['N_C+N_{Eff,0}'], thisdiodes[0].rad, 
                               0.)
        plt.plot(x,neffxceck,linestyle='-.')
        fparas['U_{dep}^{min}'] = x[np.argmin(neff)]
        
        #allvars = fitter.dNEff(x, diodes[samples[0][0:4]].rad,diodes[samples[0][0:4]].NEff_norad())
        ##print(fitter.cov)
        #allerr = 0.
        #for var in allvars:
        #    up,down = var
        #    maxvar = max(abs(x[np.argmin(up)]-x[np.argmin(neff)]),abs(x[np.argmin(down)]-x[np.argmin(neff)]))
        #    allerr += maxvar**2
        
        fparas['sd_U_{dep}^{min}'] = math.sqrt(0.) #needs to be properly evaluated
        fparas['g_axFluence'] = fparas['g_a'] *  thisdiodes[0].rad
        fparas['sd_g_axFluence'] = fparas['g_axFluence'] * 0.1
    
    plt.title(fparas['ID'])
    allparas.append(fparas)
    #plt.plot(x, fitter.NEff(x, diodes['2003'].rad, diodes['2003'].NEff_norad()))
    #plt.plot(x, fitter.NEff(x, diodes['2102'].rad, diodes['2102'].NEff_norad()))
    plt.xscale('log')
    plt.xlabel('t [min]')
    plt.ylabel(r'$N_{eff}\ [1/cm^{3}]$')
    plt.tight_layout()
    plt.savefig(outdir+"fit_"+str(i)+".pdf")
    
    
from tabulate import tabulate
#reorder a bit

#print(headers)
#print(data)
#make a table out of this
#print(allparas)
print(tabulate(allparas, headers='keys'))


df = pandas.DataFrame(allparas)


#df2 = df[df['sd_g_y']<10]
#print(df2)


unitdict={
    'g_a':r'[$\Phi_{eq}^{-1} cm^{-1}$]',
    'g_axFluence': r'[$cm^{-3}$]',
    'tau_a': r'[min]',
    'N_C+N_{Eff,0}': r'[$cm^{-3}$]',
    'g_y':r'[$\Phi_{eq}^{-1} cm^{-1}$]',
    'tau_y': r'[min]',
    'U_{dep}^{min}': r'[V]',
    }

for var in ['g_a','g_axFluence','tau_a','N_C+N_{Eff,0}','g_y','tau_y','U_{dep}^{min}']:
    fluence_offsets={}
    for f in df['fluence']:
        fluence_offsets[f]=0
    plt.close()
    fig, ax = plt.subplots()
    for  i,idxrow in enumerate(df.iterrows()):
        idx,row = idxrow
        marker='o'
        color='tab:blue'
        if row['thickness'] == 200:
            marker='v'
            color='tab:orange'
        if row['material'] == 'EPI':
            marker='x'
            color='tab:green'
            
        varval = row[var]
        if (not type(varval) is float) and len(varval):
            varval = np.mean(varval)
        varvalerr = row['sd_'+var]
        if (not type(varvalerr) is float) and len(varvalerr):
            varvalerr = np.mean(varvalerr)
        
        plt.errorbar(row['fluence']+fluence_offsets[row['fluence']], varval, varvalerr,
                     xerr=row['fluence']*0.1,
                     label=row['ID'],marker=marker,
                     color = color, ecolor = color)
        fluence_offsets[row['fluence']]+=5e13
    plt.xlabel('Fluence [neq/cm$^2$]')
    varstr=var
    if varstr[0:3]=='tau':
        varstr = '\\' + varstr
    plt.ylabel('$'+ varstr +'$ '+unitdict[var])
    
    plt.tight_layout()
    plt.xscale('log')
    
    plt.legend(handles=styles.createManualLegends(
        [
            {'marker':'o', 'label':'300 µm FZ','color': 'tab:blue'},
            {'marker':'v', 'label':'200 µm FZ','color': 'tab:orange'},
            {'marker':'x', 'label':'120 µm EPI','color': 'tab:green'}
            ]
        ))
    
    plt.savefig(outdir+var+'_fitted.pdf')
    plt.close()



allfits=[]


styles.setstyles()
#120µm
###############################["3003_UL","3007_UL","3008_UL"]

ax = newplot()
print('3003_UL, 1e16')
xs = pointsets.addToPlot("NEff", ["3003_UL"],["UL, 120µm"],
                         add_rel_y_unc=0.1)
#allfits.append( ( addfitted(["3003_UL"]), r'1.0e16 neq/$cm^2$', 10.) )
cosmetics(xs,ax,"NEff $[1/cm^{3}]$")
plt.ylim([0.5e13,7.5e13])
plt.xlim([0.5,5000])
plt.savefig(outdir+"/allNEff_120f10_"+addout+".pdf")


ax = newplot()
print('3007_UL, 2.5e15')
xs = pointsets.addToPlot("NEff", ["3007_UL"],["UL, 120µm"],
                         add_rel_y_unc=0.1)
#allfits.append( ( addfitted(["3007_UL"]), r'2.5e15 neq/$cm^2$', 2.5) )
cosmetics(xs,ax,"NEff $[1/cm^{3}]$")
plt.ylim([0.5e13,7.5e13])
plt.xlim([0.5,5000])
plt.savefig(outdir+"/allNEff_120f2.5_"+addout+".pdf")


ax = newplot()
print('3008_UL, 1.5e15')
xs = pointsets.addToPlot("NEff", ["3008_UL"],["UL, 120µm"],
                         add_rel_y_unc=0.1)
#allfits.append( ( addfitted(["3008_UL"]), r'1.5e15 neq/$cm^2$', 10.) )
cosmetics(xs,ax,"NEff $[1/cm^{3}]$")
plt.ylim([0.5e13,7.5e13])
plt.xlim([0.5,5000])
plt.savefig(outdir+"/allNEff_120f1.5_"+addout+".pdf")



#exit()


ax = newplot()
pointsets.addToPlot("Udep",["6002_6in"],
                    colors='fluence',
                    #colors=['tab:red','tab:purple'],
                    marker='x')
kneepointset.addToPlot("Udep",["6002_6in"],[" first kink"],
                       #colors=['tab:red','tab:purple']
                    colors='fluence',
                       )
pointsets.addToPlot("Udep",["1002_UR", "1003_UR", "1102_UR"],[" UR"," UR"," UR"],
                    #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x')


xs = pointsets.addToPlot("Udep",["1002_UL", "1003_UL", "1102_UL"],
                    #colors=['tab:blue','tab:green','tab:orange']
                    colors='fluence',
                    )
cosmetics(xs,ax)
plt.savefig(outdir+"/300_6inch"+addout+".pdf")


ax = newplot()
pointsets.addToPlot("Udep",["6002_6in"],[" second kink"],
                    #colors=['tab:red','tab:purple'],
                    colors='fluence',
                    marker='x')
xs = kneepointset.addToPlot("Udep",["6002_6in"],[" first kink"],
                       #colors=['tab:red','tab:purple']
                    colors='fluence',
                       )
cosmetics(xs,ax)
plt.savefig(outdir+"/6inch"+addout+".pdf")


ax = newplot()
pointsets.addToPlot("Udep", ["1002_UL","1003_UL","1102_UL"],
                    #colors=['tab:blue','tab:green','tab:orange']
                    colors='fluence',
                    )
xs = pointsets.addToPlot("Udep", ["1002_UR","1003_UR","1102_UR"],[" UR"," UR"," UR"],
                    #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x')
cosmetics(xs,ax)
plt.savefig(outdir+"/300"+addout+".pdf")


ax = newplot()
pointsets.addToPlot("Udep", ["2002_UL","2003_UL","2102_UL"],
                    #colors=['tab:blue','tab:green','tab:orange']
                    colors='fluence',
                    )
xs = pointsets.addToPlot("Udep", ["2002_UR","2003_UR","2102_UR"],[" UR"," UR"," UR"],
                    #colors=['tab:blue','tab:green','tab:orange'],
                    colors='fluence',
                    marker='x')
cosmetics(xs,ax)
plt.savefig(outdir+"/200"+addout+".pdf")


ax = newplot()
xs = pointsets.addToPlot("Udep", ["3003_UL","3007_UL","3008_UL"],
                    #colors=['tab:blue','tab:green','tab:orange']
                    colors='fluence',
                    )

cosmetics(xs,ax)
plt.savefig(outdir+"/120"+addout+".pdf")


newplot()
xs = pointsets.addToPlot("NEff", ["3003_UL","3007_UL","3008_UL"],
                         colors='fluence',
                         marker='o')
xs = pointsets.addToPlot("NEff", ["2002_UL","2003_UL","2102_UL"],
                         colors='fluence',
                         marker='x')
xs = pointsets.addToPlot("NEff", ["1002_UR","1003_UR","1102_UR"],
                         colors='fluence',
                         marker='+')

cosmetics(xs,ax,"NEff $[1/cm^{3}]$")
plt.savefig(outdir+"/allNEff"+addout+".pdf")



##### with fits






#make the fits:


    





















