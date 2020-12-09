from numpy import arange
import numpy as np
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot
from sklearn.cluster import KMeans 
from scipy.optimize import fsolve
from fileIO import fileReader
from plotting import curvePlotter

from lmfit.models import LinearModel
import sklearn.gaussian_process as gp

class Linear(object):
    def __init__(self, a=None,b=None):
        self.a=a 
        self.b=b 
        
    def __call__(self,x,a=None,b=None):
        if a is None and b is None:
            return 1e19*self.a*(x/100 + 9) + self.b*1e22
        else:
            return 1e19*a*(x/100 + 9) + b*1e22
        
    def getA(self):
        return 1e19*self.a
    
    def getB(self):
        return 1e22*self.b
 
class DepletionFitter(object):
    def __init__(self,  x=None, y=None, 
                 const_cap=None,
                 rising=2, constant=8,
                 debugfile=None):
        self.data=None
        self.rising = rising
        self.constant = constant
        self.const_cap = const_cap
        self.debugfile = debugfile
        if self.const_cap is not None:
            self.const_cap/=1e22
        if x is not None and y is not None:
            self._setData(x,y)
        
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
        
    def _findcut(self):
        #simple comparison of average slope and current slope
        slope = self.data['slope']
        avgslope = np.mean(np.abs(slope))
        rising = np.abs(slope) > avgslope*self.rising
        constant = np.abs(slope)*self.constant < avgslope
        return rising, constant
       
    def _linear(self, x, a, b): 
        return a * x + b
    
    def _dofit(self, debugplot=False):
        rise, cons = self._findcut()
        funcs=[]
        for r in [rise,cons]:
            x = self.data['x'][r]
            y = self.data['y'][r]
            
            if len(x) < 5:
                print(len(x))
                raise ValueError("cut offs wrongly set")
            if debugplot:
                from matplotlib import pyplot as plt
                plt.plot(x,y,marker='x')
            
            popt, _ = curve_fit(Linear(), x, y)
            a, b = popt
            line = Linear(a=a, b=b)
            funcs.append(line)
            if self.const_cap is not None:
                funcs.append(Linear(a=0, b=self.const_cap))
                break
            
        if debugplot:
            from matplotlib import pyplot as plt
            l0,l1 = funcs
            print('cap:' ,l1.getB())
            plt.close()
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
            
    def getDepletionVoltage(self,debugplot=False):
        l0,l1 = self._dofit(debugplot)
        return fsolve(lambda x : l0(x) - l1(x),-1000)[0]
 
 
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
    def __init__(self, globalpath, file=None, debug=False, debugfile=None, **kwargs):
        self.fitter=PolFitter()
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
        