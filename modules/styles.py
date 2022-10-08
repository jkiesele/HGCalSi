
use_hep_style = False
if use_hep_style:
    import mplhep as hep
    hep.style.use(hep.style.ROOT)

import matplotlib
import matplotlib.pyplot as plt


g_fontsize=14
def setstyles(fontsize=14):
    global g_fontsize
    g_fontsize=fontsize
    
    axes = {'labelsize': fontsize,
            'titlesize': fontsize}
    
    matplotlib.rc('axes', **axes)
    matplotlib.rc('legend',fontsize=fontsize)
    matplotlib.rc('xtick',labelsize=fontsize)
    matplotlib.rc('ytick',labelsize=fontsize)
    if use_hep_style:
        hep.style.use(hep.style.ROOT)

def fontsize():
    return g_fontsize

def fontdict():
    return  {'size': g_fontsize}

#setstyles()

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
    
    