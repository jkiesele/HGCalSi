#!/usr/bin/env python3

from argparse import ArgumentParser
import styles
import os
import pickle 

defcvfile="*.cv"

parser = ArgumentParser()
parser.add_argument('inputDir')
parser.add_argument('--cvfile', default=defcvfile)
parser.add_argument('--Cp', default=False,  action='store_true')

parser.add_argument('--var', type=float, default=10.)
parser.add_argument('--rederive', default=False,  action='store_true')
parser.add_argument('--ignoremissing', default=False,  action='store_true')
parser.add_argument('--strictcheck', default=False,  action='store_true')
parser.add_argument('--batch', default=False,  action='store_true')
parser.add_argument('--outfile', default="extracted")
args = parser.parse_args()

def invert(a):
    if a is not None:
        return -a
    return a
           

'''
Use two approaches:
- linear fit of two curves
- smoothed inflection point of two skewed curves
'''

from fitting import  readDepletionData
from tools import getDepletionVoltage, getDiodeAndTime
import matplotlib.pyplot as plt
from plotting import curvePlotter

globalpath=os.getenv("DATAPATH")+'/'
outpath=os.getenv("DATAOUTPATH")+'/'
os.system('mkdir -p '+outpath+args.inputDir)
outpath = outpath+args.inputDir+"/"
outprefix = outpath+args.outfile


high_end= None
high_start= None
low_end = None
low_start = None
const_cap=None

mode = "CVs"
if args.Cp:
    mode = "CV"
variation = float(args.var)
if args.rederive or (not args.cvfile==defcvfile) or (not os.path.isfile(outpath+args.outfile+".depl") and not args.ignoremissing):

    cv_plotter = curvePlotter(mode=mode,path=globalpath)
    cv_plotter.addPlotFromFile(args.inputDir+"/"+args.cvfile,
                                   label="test")
    
    
    #read diode from inputdir
    diodestr = args.inputDir[0:4]
    from diodes import diodes
    try:
        print('ideal capacitance',diodes[diodestr].Cideal(), '1/Cideal^2', 1/(diodes[diodestr].Cideal()**2))
    except:
        pass
    print('fit rising edge from to (positive)')
    print('fit constant from to')
    print('constant value (if any)')
    
    print('fit rising edge from to (positive)')
    print('fit constant from to')
    print('constant value (if any)')
    cv_plotter.showPlot()
    high_end= - float(input())
    high_start= - float(input())
    low_end = - float(input())
    low_start = - float(input())
    
    

    const_cap=input()
    if not const_cap == "":
        const_cap=float(const_cap)
    else:
        const_cap=None
else:
    d = readDepletionData(outpath,args.outfile+".depl")
    
    #print(d)
    
    
    
    high_end= d['high_end']
    high_start= d['high_start']
    low_end = d['low_end']
    low_start = d['low_start']
    const_cap=d['const_cap']
    if 'rangevar' in d.keys():
        variation=d['rangevar']
        
    print('high_end',high_end)
    print('high_start',high_start)
    print('low_end',low_end)
    print('low_start',low_start)
    print('const_cap',const_cap)
    print('variation',variation)
    
d,t = getDiodeAndTime(args.inputDir)

plt.title(d.paperlabel() + '\n'+str(t)+' min')

diodestr = args.inputDir[0:4]
from diodes import diodes
cideal = diodes[diodestr].Cideal()
v = getDepletionVoltage(globalpath+args.inputDir+"/"+args.cvfile,
                        min_x=-900,
                        const_cap=const_cap,
                        debug=True,
                        low_start = low_start,
                        low_end = low_end,
                        high_start = high_start,
                        high_end = high_end,
                        variation = variation,
                        strictcheck=args.strictcheck,
                        savedatapath=outprefix+".depl",
                        plotprefix = outpath+'/'+'cv_fit_'+args.inputDir,
                        mode=mode,
                        cideal = cideal,
                        interactive=not args.batch)


if not defcvfile == args.cvfile:
    exit()

d2 = readDepletionData(outpath,args.outfile+".depl")
neff = diodes[diodestr].NEff(-d2['depletion_nominal'])
print('NEff ', neff, 
      'rel: ', diodes[diodestr].NEff(-d2['depletion_down'])/neff * 100. -100., 
      '', diodes[diodestr].NEff(-d2['depletion_up'])/neff* 100. -100., 
      '% , depl voltage',-v)
    
#smoothen IV curves and save as dicts for easy processing
plt.close()
ivpl = curvePlotter(mode="IV",path=globalpath)
ivpl.addPlotFromFile(args.inputDir+"/*.iv",label="I")
xs,ys = ivpl.getXYSmooth()
#plt.plot(-xs,-ys, label='I (smooth)')
plt.title(d.paperlabel() + ', '+str(t)+' min')
plt.legend()
ivpl.savePlot(outprefix+"_iv.pdf",True)

ivgr = curvePlotter(mode="IVGR",path=globalpath)
ivgr.addPlotFromFile(args.inputDir+"/*.iv",label="I (guard ring)")
_,ygr = ivgr.getXYSmooth()
#plt.plot(-xs,-ys, label='I (smooth)')
plt.legend()
ivpl.savePlot(outprefix+"_ivgr.pdf",True)

data = {'iv_x': ivpl.x,
        'iv_y': ivpl.y,
        'iv_tot': ivpl.y+ivgr.y,
        'iv_x_smooth': xs,
        'iv_y_smooth': ys,
        'iv_y_tot_smooth': ygr+ys,
        }

with open(outprefix+'.iv', 'wb')  as filehandler:
    pickle.dump(data, filehandler)


