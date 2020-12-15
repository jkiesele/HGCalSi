
import matplotlib

fontsize=12

axes = {'labelsize': fontsize,
        'titlesize': fontsize}

matplotlib.rc('axes', **axes)
matplotlib.rc('legend',fontsize=fontsize)
matplotlib.rc('xtick',labelsize=fontsize)
matplotlib.rc('ytick',labelsize=fontsize)
