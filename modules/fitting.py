from numpy import arange
import numpy as np
import math
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot
from sklearn.cluster import KMeans 
from scipy.optimize import fsolve
from fileIO import fileReader
from plotting import curvePlotter

from lmfit.models import LinearModel
import sklearn.gaussian_process as gp
from matplotlib import pyplot as plt
import scipy.odr as odr
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pickle



def smoothen(x,y):
    from scipy.interpolate import interp1d
    
    ysmooth = savgol_filter(y, 5,3)
    f2 = interp1d(x, ysmooth, kind='cubic')
    xint = np.arange(x[-1],x[0],1)
    #flip?
    yint = f2(xint)
    
    return xint, yint
 

class AnnealingFitter(object):
    def __init__(self):
        self.a=None
        self.tau_0=None
        self.N_c =None
        self.N_Yinf = None
        self.tau_y=None
        
        self.scalex=100.
        self.scaley=100.
        
    def _DNeff(self, B, t):
        a, tau_0, N_c, N_Yinf, tau_y = B
        return a*np.exp(- t/self.scalex/tau_0) + N_c + N_Yinf * ( 1. - 1./(1.+t/self.scalex/tau_y))
       
    def __DNeff(self, t, a, tau_0, N_c, N_Yinf, tau_y):
        return a*np.exp(- t/self.scalex/tau_0) + N_c + N_Yinf * ( 1. - 1./(1.+t/self.scalex/tau_y))
       
       
    def DNeff(self, t):
        if not len(t):
            t=np.array(t)
        return self.a*np.exp(- t/self.scalex/self.tau_0) + self.N_c + self.N_Yinf * ( 1. - 1./(1.+t/self.scalex/self.tau_y))


    def fit(self, x, y, xerr, yerr=None):
        
        if True:
            m = odr.Model(self._DNeff)
            if yerr is None:
                yerr = y/1000.
            mydata = odr.RealData(x, y, sx=xerr, sy=yerr)
            myodr = odr.ODR(mydata, m, beta0=[ 2.37213506e+02 , 
                                               3.45510553e-01 ,
                                               1.96604468e+02 ,
                                               -5.23269275e+06,
                                               -1.91549327e+05])
            out=myodr.run()
            #print(out)
            popt = out.beta
            self.a=popt[0]
            self.tau_0=popt[1]
            self.N_c =popt[2]
            self.N_Yinf = popt[3]
            self.tau_y=popt[4]
        
            return
        
        popt, _ = curve_fit(self.__DNeff, x, y)
        print(popt)
        self.a=popt[0]
        self.tau_0=popt[1]
        self.N_c =popt[2]
        self.N_Yinf = popt[3]
        self.tau_y=popt[4]
        
        
    

class Linear(object):
    def __init__(self, a=None,b=None):
        self.a=a 
        self.b=b 
        
    def __call__(self,x,a=None,b=None):
        if a is None and b is None:
            return 1e19*self.a*(x/10 + 9) + self.b*1e22
        else:
            return 1e19*a*(x/10 + 9) + b*1e22
        
    def getA(self):
        return 1e19*self.a
    
    def getB(self):
        return 1e22*self.b
    
    
class DoubleLinear(object):
    def __init__(self,
        a0=None         ,
        b0=None         ,
        a1=None         ,
        b1=None         ,
        mixpoint=None   ,
        mixsmooth=None
        ):
        self.a0=a0
        self.b0=b0
        self.a1=a1
        self.b1=b1
        self.mixpoint=mixpoint
        self.mixsmooth=mixsmooth
        
    def __call__(self,x,a0=None,b0=None, a1=None,b1=None, mixsmooth=None, mixpoint=None):
        if a0 is None:
            a0=self.a0
            b0=self.b0
            a1=self.a1
            b1=self.b1
            mixsmooth=self.mixsmooth
            mixpoint=self.mixpoint
            
        import tensorflow as tf
            
        lin0 = 1e19* a0*(x/10 + 9) +  b0*1e22
        lin1 = 1e19* a1*(x/10 + 9) +  b1*1e22
        sigx = mixsmooth * x/100. - mixpoint
        sig = 1. / (1 + tf.exp(-sigx))
        return lin0*sig + lin1*(1.-sig)
        

def readDepletionData(directory, file="extracted.dv"):
    import os
    allpath = os.path.join(directory,file)
    with open(allpath, 'rb')  as filehandler:
        return pickle.load(filehandler)
    
 
class DepletionFitter(object):
    def __init__(self,  x=None, y=None, 
                 const_cap=None,
                 rising=2, constant=8,
                 debugfile=None,
                 low_start=None,
                 low_end=None,
                 high_start=None,
                 high_end=None,
                 varcut=10,
                 strictcheck=True,
                 interactive=False,
                 addextracuncat=-1000,
                 extraunc=10.):
        
        self.varcut=varcut
        self.strictcheck=strictcheck
        self.data=None
        self.rising = rising
        self.constant = constant
        self.const_cap = const_cap
        self.debugfile = debugfile
        self.addextracuncat=addextracuncat
        self.extraunc=extraunc
        if x is not None and y is not None:
            self._setData(x,y)
            
        self.low_start=low_start
        self.low_end=low_end
        self.high_start=high_start
        self.high_end=high_end
        
        self.interactive=interactive
        
        #save info
        self.data.update(
            {'low_start': low_start,
             'low_end': low_end,
             'high_start': high_start,
             'high_end': high_end,
             'varcut': varcut,
             'const_cap': const_cap
             }
            )
        
    def _setData(self, x,y):
        #prepare only edges
        slope=[]
        offset=[]
        s=0
        for i in range(len(x)-1):
            s = (y[i+1]-y[i])/(x[i+1]-x[i])
            slope.append(s)
            offset.append(y[i] - s*x[i])
        s = (y[1]-y[0])/(x[1]-x[0])
        slope.insert(0, s)
        offset.insert(0, (y[i] - s*x[i]))
        slope = np.array(slope)
        offset = np.array(offset)
        
        self.data={'slope': slope,
                   'offset': offset,
                   'x': x,
                   'y': y}
        
        self._smoothen()
        
    def _checkchange(self,c):
        delta = abs(self.data['x'][0]-self.data['x'][1])#min one point distance
        if not self.strictcheck:
            return c
        if not c == 0 and delta > abs(c):
            if c>0:
                c=delta
            else:
                c=-delta
        return c
        
    def _findcut(self):
        #simple comparison of average slope and current slope
        
        
        x = self.data['x_smooth']
        #y = self.data['y_smooth']
        
        #create smoothed data here
        
        outsel=[]
        for s in [[self.low_start, self.low_end], [self.high_start,self.high_end]]:
            defstart, defend = s[0],s[1]
            thissel=[]
            for var1 in [0, -self.varcut, self.varcut]:
                for var2 in [ 0, -self.varcut, self.varcut]:
                    
                    change1 = abs(defstart-defend)*(var1/100.)
                    change2 = abs(defstart-defend)*(var2/100.)
                    
                    start = defstart + self._checkchange(change1)
                    end = defend + self._checkchange(change2)
                    
                    sel = np.logical_and(x >= start, x < end)
                    thissel.append(sel)
            outsel.append(thissel)
             
        return outsel   
        #for lowstart in [self.low_start, self.low_start-self.varcut,self.low_start+self.varcut]:
        #    for lowend in [self.low_start, self.low_start-self.varcut,self.low_start+self.varcut]:
        #
        #constant = np.logical_and(self.data['x'] >= self.low_start,self.data['x'] < self.low_end)
        #constup =     np.logical_and(self.data['x'] >= self.low_start+20,self.data['x'] < self.low_end+20)
        #constdown = np.logical_and(self.data['x'] >= self.low_start-20,self.data['x'] < self.low_end-20)
        #    
        #rising = np.logical_and(self.data['x'] >= self.high_start,self.data['x'] < self.high_end)
        #riseup
        #risedown
        #
        #return rising, constant, riseup, risedown, constup, constdown
       
    def _linear(self, x, a, b): 
        return a * x + b
    
    def _dofit(self, debugplot=False, savedatapath=None):
        #
        # assume cuts
        # fit lines, store
        # get value, store
        #
        cons,rise = self._findcut()
        #each [sela, selb, ...]
        riselines=[]
        conslines=[]
        for srise in rise:
            for scons in cons:
                x = self.data['x_smooth'][srise]
                y = self.data['y_smooth'][srise]
                popt, _ = curve_fit(Linear(), x, y)
                a, b = popt
                line = Linear(a=a, b=b)
                riselines.append(line)
                
                if self.const_cap is not None and self.const_cap > 0:
                    #print('using constant ',self.const_cap)
                    conslines.append(Linear(a=0., b=self.const_cap/1e22))
                    conslines.append(Linear(a=0., b=self.const_cap*(1.-0.002*self.varcut)/1e22))
                    conslines.append(Linear(a=0., b=self.const_cap*(1.+0.002*self.varcut)/1e22))
                    continue
                
                x = self.data['x_smooth'][scons]
                y = self.data['y_smooth'][scons]
                popt, _ = curve_fit(Linear(), x, y)
                a, b = popt
                line = Linear(a=a, b=b)
                conslines.append(line)
                
        depl=[]
        for r,c in zip(riselines,conslines):
            vdep  = fsolve(lambda x : r(x) - c(x),-1000)[0]
            depl.append(vdep)
            if vdep < self.addextracuncat:
                depl.append(vdep+(vdep-self.addextracuncat)*(self.extraunc/100.))
                depl.append(vdep-(vdep-self.addextracuncat)*(self.extraunc/100.))
            
        depl=np.array(depl)
        nom,up,down = depl[0], np.max(depl), np.min(depl)
        print(nom,up,down)
        
        if debugplot:
            #plt.close()
            plt.plot(self.data['x'],self.data['y'],marker='x',linewidth=None,label='data')
            plt.plot(self.data['x_smooth'],self.data['y_smooth'],label='smoothened')
            plt.legend()
            for l in riselines+conslines:
                plt.plot(self.data['x'],l(self.data['x']),linewidth=0.5)
            plt.xlabel("U [V]")
            plt.ylabel("$1/C^2 [1/F^2]$")
            plt.ylim([np.min(self.data['y'])/1.1, np.max(self.data['y'])*1.2])
            fig = plt.gcf()
            if savedatapath is not None:
                fig.savefig(savedatapath+'_fig.pdf')
            if self.interactive:
                plt.show()
            
            
        if savedatapath is not None:
            d = {'depletion_nominal':nom,
                 'depletion_up': up,
                 'depletion_down': down,
                 'riselines': riselines,
                 'conslines': conslines,
                 }
            d.update(self.data)
            
            with open(savedatapath, 'wb')  as filehandler:
                pickle.dump(d, filehandler)
            
            
                
        return nom,up,down
            
            
        #############    
            
        rise, cons = self._findcut()
        funcs=[]
        for r in [rise,cons]:
            x = self.data['x'][r]
            y = self.data['y'][r]
            
            if len(x) < 3:
                print(len(x))
                raise ValueError("cut offs wrongly set")
            if debugplot:
                plt.plot(x,y,marker='x')
            
            popt, _ = curve_fit(Linear(), x, y)
            a, b = popt
            print(a,b)
            #exit()
            line = Linear(a=a, b=b)
            funcs.append(line)
            if self.const_cap is not None:
                funcs.append(Linear(a=0, b=self.const_cap))
                break
            
        self._smoothen()
        if debugplot:
            
            l0,l1 = funcs
            print('cap:' ,l1.getB())
            #plt.close()
            plt.plot(self.data['x'],self.data['y'])
            plt.plot(self.data['x'],l0(self.data['x']))
            plt.plot(self.data['x'],l1(self.data['x']))
            plt.xlabel("U [V]")
            plt.ylabel("$1/C^2 [1/F^2]$")
            if self.debugfile is None:
                plt.show()
            else:
                plt.savefig(self.debugfile+'.pdf')
                plt.close()
        
        return funcs
            
    def getDepletionVoltage(self,debugplot=False, withunc=False, savedatapath=None):
        if withunc:
            return self._dofit(debugplot=debugplot,savedatapath=savedatapath)
        return self._dofit(debugplot=debugplot,savedatapath=savedatapath)[0]
    
    def _smoothen(self):
        x = self.data['x']
        y = self.data['y']
        xint,yint = smoothen(x, y)
        self.data['x_smooth']=xint
        self.data['y_smooth']=yint
        
        
    

class DepletionFitterInflect(object):
    def __init__(self):
        pass
    
    
 
 
class PolFitter(object):
    def __init__(self):
        self.p=[1.,1.,1.]
        self.xscale=1
        self.yscale=1
 
    def _objective2(self,x,a,b,c):
        return a*x**2 + b*x + c
    
    def _objective3(self,x,a,b,c,d):
        return a*x**3 + b*x**2 + c*x + d
        
    def predict(self,x):
        return np.array([self.yscale*self._objective3(x[i]/self.xscale,*self.p) for i in range(len(x))]), None
    
    
    def fit(self,x,y):
        #get scale
        self.xscale=np.mean(x)
        self.yscale=np.mean(y)
        
        xin=x/self.xscale
        yin=y/self.yscale
        
        popt, _ = curve_fit(self._objective3, xin, yin)
        self.p = popt
        self.p = [p for p in self.p]
 
class GPFitter(object):
    def __init__(self, smoothness=.01):
        #self.x=None
        #self.y=None
        self.smoothness=smoothness
        self.model=None
        self.scale_x=1/100.
        self.scale_y=1e-22
        
    
    def fit(self,x,y):
        xt=x * self.scale_x
        xt = np.expand_dims(np.array(xt), axis=1)
        yt=y * self.scale_y
        #fit here
        kernel = gp.kernels.ConstantKernel(self.smoothness, (1e-1, 1e3)) \
         #* gp.kernels.ConstantKernel(self.smoothness, (1e-1, 1e3))\
         #* gp.kernels.RBF(self.smoothness, (1e-3, 1e3))
        kernel = gp.kernels.RBF(length_scale=self.smoothness)
       
         
         
        self.model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=.005, normalize_y=True)
        self.model.fit(xt, yt)
    
    def predict(self,x):
        p,s = self.model.predict(np.expand_dims(np.array(x), axis=1) * self.scale_x, return_std=True)
        return p / self.scale_y, s / self.scale_y
    
      
class fittedIV(object):
    def __init__(self, globalpath, file=None, debug=False, debugfile=None, useGP=False, **kwargs):
        self.fitter=PolFitter()
        if useGP:
            self.fitter=GPFitter()
        self.plter = curvePlotter(path=globalpath, mode="IV")
        self.debugfile=debugfile
        if file is not None:
            self.readFile(file,debug=debug,**kwargs)
        
    def readFile(self, file, debug=False, **kwargs):
        self.plter.addPlotFromFile(infile=file,noplot=True, **kwargs)
        self.fitter.fit(self.plter.x,self.plter.y)
        if debug:
            from matplotlib import pyplot as plt
            plt.close()
            plt.plot(self.plter.x,self.plter.y)
            plt.plot(self.plter.x,self.eval(self.plter.x)[0])
            plt.xlabel("U [V]")
            plt.ylabel("I [A]")
            if self.debugfile is None:
                plt.show()
            else:
                plt.savefig(self.debugfile+".pdf")
                plt.close()
        
        
    def eval(self, voltage):
        try:
            l=len(voltage)
            x=voltage
        except:
            x=[voltage]
        return self.fitter.predict(np.array(x,dtype='float'))
  
        
## define the true objective function
#def objective(x, a, b):
#    return a * x + b
# 
## load the dataset
#url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'
#dataframe = read_csv(url, header=None)
#data = dataframe.values
## choose the input and output variables
#x, y = data[:, 4], data[:, -1]
## curve fit
#popt, _ = curve_fit(objective, x, y)
## summarize the parameter values
#a, b = popt
#print('y = %.5f * x + %.5f' % (a, b))
## plot input vs output
#pyplot.scatter(x, y)
## define a sequence of inputs between the smallest and largest known inputs
#x_line = arange(min(x), max(x), 1)
## calculate the output for the range
#y_line = objective(x_line, a, b)
## create a line plot for the mapping function
#pyplot.plot(x_line, y_line, '--', color='red')
#pyplot.show()
#
#

class fitterBase(object):
    def __init__(self):
        pass
    
    def fitfunc(self, x, *args): #overload
        return 0.
    
    def fit(self, x, y, range = None):
        if range is not None:
            pass #TBI
        
        popt, _ = curve_fit(self.fitfunc, x, y)
        return popt
        