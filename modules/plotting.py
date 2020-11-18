

from matplotlib import pyplot as plt
from fileIO import fileReader
from tools import convertToCs        
    
    
class curvePlotter(object):
    def __init__(self, mode, path=""):
        assert mode == "CV" or mode == "IV" or mode=="CVs"
        self.mode=mode
        self.plt=plt
        if mode=="CVs":
            mode="CV"
        self.fileReader=fileReader(mode=mode, path=path)
        
        
    def addPlotFromFile(self,infile, selection=None, min_x=None, **kwargs):
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
            
        self.plt.plot(x,y, **kwargs)
        
    def _createPlot(self):
        self.plt.xlabel("U [V]")
        if self.mode == "CV":
            self.plt.ylabel("$1/C^2\ [1/F^2]$")
        elif self.mode == "IV":
            plt.ylabel("I [A]")
        self.plt.legend()
        
        
    def savePlot(self, outfile):
        self._createPlot()
        self.plt.savefig(outfile)
        self.plt.close()
    
    def showPlot(self):
        self._createPlot()
        self.plt.show()
        
