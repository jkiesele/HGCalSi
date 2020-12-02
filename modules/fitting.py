from numpy import arange
import numpy as np
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot
from sklearn.cluster import KMeans 

from lmfit.models import LinearModel
 
class DepletionFitter(object):
    def __init__(self, mode='kmeans', x=None, y=None, allowed_change=.1):
        self.mode=mode
        self.data=None
        if x is not None and y is not None:
            self._setData(x,y)
        
    def _setData(self, x,y):
        #prepare only edges
        slope=[]
        offset=[]
        s=0
        for i in range(len(x)-1):
            s = (x[i+1]-x[i])/(y[i+1]-y[i])
            slope.append(s)
            offset.append(y[i] - s*x[i])

        slope = np.expand_dims(np.array(slope), axis=1)
        offset = np.expand_dims(np.array(offset), axis=1)
        print(slope.shape)
        print(offset.shape)
        
        slope_off = np.concatenate([slope,offset],axis=-1)
            
        self.data={'slope_off': slope_off,
                   'x': x,
                   'y': y}
        
    def _findcut(self, direction='forward'):
        #fit from the start until chi2 breaks to get first slope
        x,y = self.data['x'], self.data['y']
        residuals=[]
        for i in range(5,len(x)):
            mod = LinearModel()
            pars = mod.guess(y[0:i], x=x[0:i])
            out = mod.fit(y[0:i], pars, x=x[0:i])
            #print(out.params)
            print(out.residual.shape)
            residuals.append(np.abs(out.residual[-1])**2/np.mean(out.residual**2))
            #residuals.append(np.sum(out.residual**2)/i**2)
        return residuals
 
 
 
 
# define the true objective function
def objective(x, a, b):
    return a * x + b
 
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
# choose the input and output variables
x, y = data[:, 4], data[:, -1]
# curve fit
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
a, b = popt
print('y = %.5f * x + %.5f' % (a, b))
# plot input vs output
pyplot.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b)
# create a line plot for the mapping function
pyplot.plot(x_line, y_line, '--', color='red')
pyplot.show()



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
        