

from matplotlib import pyplot as plt
from fileIO import fileReader
#from tools import convertToCs        
import math

def convertToCs(Cp, kappa, freq=10000):

    #Cp=np.abs(Cp)
    #kappa=np.abs(kappa)
    omega = math.pi * freq
    Rp = 1/kappa 
    Cs = (1. + omega**2 * Rp**2 * Cp**2)/(omega**2 * Rp**2 * Cp)
    return Cs


    
class curvePlotter(object):
    def __init__(self, mode, path=""):
        assert mode == "CV" or mode == "IV" or mode=="CVs" or mode == 'IVGR'
        self.mode=mode
        self.plt=plt
        self.x=None
        self.y=None
        if mode=="CVs":
            mode="CV"
        self.fileReader=fileReader(mode=mode, path=path)
        if self.mode == "IVGR":
            self.mode="IV"
        
        
    def addPlotFromFile(self,infile, selection=None, min_x=None,max_x=None, noplot=False, **kwargs):
        x,y,k = self.fileReader.read(infile)
        if self.mode == "CVs":
            y=convertToCs(y,k)
            y=1/y**2
        if self.mode == "CV":
            y=1/y**2
        if selection is not None:
            x,y=selection(x,y)
        if min_x is not None:
            x, y = x[x>min_x],y[x>min_x]
        if max_x is not None:
            x, y = x[x<max_x],y[x<max_x]
        
        self.x=x
        self.y=y
        if not noplot:
            self.plt.plot(-x,y, **kwargs)
        
    def labelAxes(self):
        self.plt.xlabel("-U [V]")
        if self.mode == "CV":
            self.plt.ylabel("$1/C_p^2\ [1/F^2]$")
        elif self.mode == "CVs":
            self.plt.ylabel("$1/C_s^2\ [1/F^2]$")
        elif self.mode == "IV":
            plt.ylabel("I [A]")
        
    def _createPlot(self,nolegend=False):
        self.labelAxes()
        if not nolegend:
            self.plt.legend()
        
        
    def savePlot(self, outfile,nolegend=False):
        self._createPlot(nolegend)
        self.plt.tight_layout()
        self.plt.savefig(outfile)
        self.plt.close()
    
    def showPlot(self):
        self._createPlot()
        self.plt.show()
        
