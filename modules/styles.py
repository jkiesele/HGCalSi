
import matplotlib
import matplotlib.pyplot as plt
import math
fontsize=12

def setstyles():
    axes = {'labelsize': fontsize,
            'titlesize': fontsize}
    
    matplotlib.rc('axes', **axes)
    matplotlib.rc('legend',fontsize=fontsize)
    matplotlib.rc('xtick',labelsize=fontsize)
    matplotlib.rc('ytick',labelsize=fontsize)

setstyles()

def addModifiedXAxis(xs, func,  label):
    import numpy as np
    dummy = [1. for _ in xs]
    lims = plt.xlim()
    lims = np.array(lims)
    xs = np.array(xs)
    ax = plt.twiny()
    ax.plot(func(xs),dummy)
    #ax.xlabel(label)
    ax.cla()
    ax.set_xlim(func(lims))
    ax.set_xlabel(label)
    return ax
    
    
    
def createManualLegends(
        markertuples :list
        ):
    
    import matplotlib.lines as mlines
    
    handles=[]
    for mt in markertuples:
        l = mlines.Line2D([], [], **mt, linestyle='None')
        handles.append(l)
    return handles
    
    