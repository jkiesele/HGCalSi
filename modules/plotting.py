

from matplotlib import pyplot as plt
from fileIO import fileReader
#from tools import convertToCs        
import math

def convertToCs(Cp, G, freq=10000):

    #Cp=np.abs(Cp)
    #kappa=np.abs(kappa)
    omega = 2. * math.pi * freq
    Rp = 1/G 
    Cs = (1. + omega**2 * Rp**2 * Cp**2)/(omega**2 * Rp**2 * Cp)
    
    #same :
    Cs = (G**2 + omega**2 * Cp**2)/(omega**2 * Cp)
    
    return Cs


    
class curvePlotter(object):
    def __init__(self, mode, path="", read_freq=False):
        assert mode == "CV" or mode == "IV" or mode=="CVs" or mode == 'IVGR'
        self.mode=mode
        self.plt=plt
        self.x=None
        self.y=None
        self.read_freq=read_freq
        if mode=="CVs":
            mode="CV"
        self.fileReader=fileReader(mode=mode, path=path,return_freq=read_freq)
        if self.mode == "IVGR":
            self.mode="IV"
        
        assert ( read_freq and ( mode=="CVs" or mode=="CV" ) ) or not read_freq
        
    def readFreq(self,infile):
        if not self.read_freq:
            return None
        _,_,_, freq = self.fileReader.read(infile)
        return freq
        
    def addPlotFromFile(self,infile, selection=None, min_x=None,max_x=None, noplot=False, **kwargs):
        x,y,k = None, None, None
        freq = 10000
        if self.read_freq:
            x,y,k, freq = self.fileReader.read(infile)
        else:
            x,y,k = self.fileReader.read(infile)
        if self.mode == "CVs":
            y=convertToCs(y,k,freq)
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
            if self.mode=='IV':
                y=-y
            self.plt.plot(-x,y, **kwargs)
            
        return freq
        
            
    def getXY(self):
        return self.x, self.y
    
    def getXYSmooth(self):
        from fitting import smoothen
        return smoothen(self.x,self.y)
        
    def labelAxes(self):
        self.plt.xlabel("-U [V]")
        if self.mode == "CV":
            self.plt.ylabel("$1/C_p^2\ [1/F^2]$")
        elif self.mode == "CVs":
            self.plt.ylabel("$1/C_s^2\ [1/F^2]$")
        elif self.mode == "IV":
            plt.ylabel("-I [A]")
        
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
        
