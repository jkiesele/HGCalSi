
from diodes import diodes
import glob
import os
import pickle
import numpy as np
import cmath
import matplotlib.pyplot as plt

from plotting import curvePlotter

datadir=os.getenv("DATAOUTPATH")

class point(object):
    def __init__(self, diodequali: str, minutes: int, fileprefix='extracted', directory=None):
        self.diodequali=diodequali #3007_UL
        self.minutes=minutes
        self.fileprefix=fileprefix
        self.diode=diodes[diodequali[:4]]

        minutestr = str(minutes)+'min'
        if minutes==0 and self.diode.no != '6002':#different here...
            minutestr='_no_ann'
            
        checkdir = datadir+'/'+diodequali+'*'+minutestr
        l=None
        if directory is not None:
            l=directory
        else:
            l = glob.glob(checkdir)
            if len(l)<1:
                raise ValueError("point.init: no directory found with "+checkdir)
            if len(l)>1:
                raise ValueError("multiple directories found with "+checkdir+':\n'+l)
            l=l[0]
            
        with open(l+'/'+fileprefix+'.depl', 'rb')  as filehandler:
            self.data = pickle.load(filehandler)
            #print('read depletion data',l,fileprefix, self.data['depletion_nominal'])
            
        try:
            with open(l+'/'+fileprefix+'.iv', 'rb')  as filehandler:
                self.data.update(pickle.load(filehandler))
        except:
            pass
            
        #try to read IV curve
        
    
    def getDepl(self):
        return self.data['depletion_nominal'], self.data['depletion_nominal']-self.data['depletion_down'], self.data['depletion_up']-self.data['depletion_nominal']
        


class pointSet(object):
    def __init__(self, diodequali, minutes, fileprefix='extracted'):
        self.points=[]
        self.diodequali=diodequali
        for t in minutes:
            self.points.append(point(diodequali,t,fileprefix=fileprefix))

    def diode(self):
        return self.points[0].diode
    
    def fileprefix(self):
        return self.points[0].fileprefix
    
    @staticmethod
    def get_closest_point(xnp, pointat, allowed_difference=5):
        good=True
        xp = np.min(xnp[xnp>pointat])
        idx = np.argwhere(xp==xnp)
        if abs(xnp[idx]-pointat) > allowed_difference:
            good=False 
        return idx, good
    
    def _get_closest_point(self, xnp, pointat, allowed_difference=5):
        return pointSet.get_closest_point(xnp, pointat, allowed_difference)#just historic
    
    def getXYs(self, mode, current_at=None):
        assert mode == "Udep" or mode == "I" or mode == "Itot" or mode == "NEff" \
        or mode == "NEffSlope" or mode == "IperFluence" or mode == "ItimesThickness" or mode == "IperThickness"\
        or mode == "IperVolume"
        current_at_up=[]
        current_at_down=[]
        if type(current_at) == str:
            assert current_at=="Udep"
            current_at=[]
            for p in self.points:
                current_at.append(p.data['depletion_nominal'])
                current_at_up.append(p.data['depletion_up'])
                current_at_down.append(p.data['depletion_down'])
        elif mode == "I" or mode == "IperFluence" or mode == "ItimesThickness" or mode == "IperThickness"or mode == "IperVolume":
            assert current_at is not None
            curr = current_at
            current_at=[]
            for _ in self.points:
                current_at.append(curr)
                current_at_up.append(curr)
                current_at_down.append(curr)
            
        xs=[]
        xerrdown=[]
        xerrup=[]
        ys=[]
        yerrdown=[]
        yerrup=[]
        index = range(len(self.points))
        for p,i in zip(self.points,index):
            x=p.minutes+p.diode.ann_offset
            
            if mode=='Udep':
                nom,down,up = p.getDepl()
                ys.append(-nom)
                yerrdown.append(-down)
                yerrup.append(-up)
            elif mode == "NEff":
                nom,down,up = p.getDepl()
                n,u,d = self.diode().NEff(-nom),self.diode().NEff(-down),self.diode().NEff(-up)
                ys.append(n)
                yerrdown.append(u)
                yerrup.append(d)
                
            elif mode == "I" or mode == "Itot" or mode == "IperFluence"  or mode == "ItimesThickness" or mode == "IperThickness" or mode == "IperVolume":
                #get closest x point (should be no problem, high res enough
                xnp = np.array(p.data['iv_x_smooth'])
                ynp = np.array(p.data['iv_y_smooth'])
                if mode == "Itot":
                    ynp = np.array(p.data['iv_y_tot_smooth'])
                    

                idx, good = self._get_closest_point(xnp, current_at[i])
                if not good:
                    continue 
                idx_up, good = self._get_closest_point(xnp, current_at_up[i])
                if not good:
                    continue 
                idx_down, good = self._get_closest_point(xnp, current_at_down[i])
                if not good:
                    continue 
                
                
                y = -ynp[idx]
                #FIXME, use other metrics here (for Udep e.g. +-)
                yup = -ynp[idx_up]
                ydown = -ynp[idx_down]
                if yup<ydown:
                    _temp = ydown
                    ydown=yup
                    yup = _temp
                    
                ydown = float(abs(y-ydown))
                yup = float(abs(y-yup))
                if mode == "IperFluence":
                    y /= p.diode.rad
                    yup /= p.diode.rad
                    ydown /= p.diode.rad
                if mode == "ItimesThickness":
                    y *= p.diode.thickness
                    yup *= p.diode.thickness
                    ydown *= p.diode.thickness
                if mode == "IperThickness":
                    y /= p.diode.thickness
                    yup /= p.diode.thickness
                    ydown /= p.diode.thickness
                if mode == "IperVolume":
                    y /= p.diode.thickness*p.diode.area
                    yup /= p.diode.thickness*p.diode.area
                    ydown /= p.diode.thickness*p.diode.area
                ys.append(y)
                yerrdown.append(ydown)
                yerrup.append(yup)
            
            xs.append(x)
            xerrdown.append(cmath.sqrt(p.diode.ann_offset_error**2 + (0.025*x)**2))
            xerrup.append(cmath.sqrt(p.diode.ann_offset_error**2 + (0.025*x)**2))    
            
        return xs,[xerrdown,xerrup],ys,[yerrdown,yerrup]


class pointSetsContainer(object):
    def __init__(self):
        self.pointsets={}
        
    def append(self, ps):
        self.pointsets[ps.diodequali]=ps
        
    def getXYsDiodes(self,mode, whichsets,current_at=None):
        xyerss=[]
        for aset in whichsets:
            ps = self.pointsets[aset]
            x,xerrs,y,yerrs = ps.getXYs(mode,current_at)
            xerrs = np.abs(np.array(xerrs))
            xerrs = np.max(xerrs,axis=0)
            yerrs = np.abs(np.array(yerrs))
            yerrs = np.max(yerrs,axis=0)
            yerrs = np.expand_dims(yerrs,axis=1)#just consistency
            
            xyerss.append({
                'diode': ps.diode(),
                't': x,
                'terr':xerrs,
                'y':y,
                'yerr':yerrs
                })
        return xyerss
        
    def addToPlot(self,mode, whichsets,add=[],marker='o',colors=None,linestyle='',linewidth=None,current_at=None):
        while len(add) < len(whichsets):
            add.append("")
        if colors is not None and colors == 'fluence':
            colors=[]
            for aset in whichsets:
                col = self.pointsets[aset].diode().fluenceCol()
                colors.append(col)
        if colors is None:
            colors = [None for _ in whichsets]
        x=None    
        for aset,addstr,c in zip(whichsets,add,colors):
            ps = self.pointsets[aset]
            x,xerr,y,yerr = ps.getXYs(mode,current_at)
            plt.errorbar(x, y,  yerr=yerr, xerr=xerr,label=ps.diode().label()+addstr,
                         linewidth=linewidth,marker=marker,linestyle=linestyle,
                         color=c)
        return np.array(x)
        
    def getRaw(self,mode, whichsets, current_at=None):
        xs=[]
        xerrs=[]
        ys=[]
        yerrs=[]
        for aset in whichsets:
            ps = self.pointsets[aset]
            x,xerr,y,yerr = ps.getXYs(mode,current_at)
            xs.append(x)
            xerrs.append(xerr)
            ys.append(y)
            yerrs.append(yerr)
        return xs, xerrs, ys, yerrs


def loadAnnealings():
    pointsets=pointSetsContainer()

    pointsets.append(pointSet("6002_6in",[0,11,31,74, 103, 149, 386, 639, 1536]))
    
    kneepointset=pointSetsContainer()
    kneepointset.append(pointSet("6002_6in",[0,11,31,74, 103, 149, 386, 639],fileprefix="firstknee"))
    
    pointsets.append(pointSet("2002_UL",[0,73, 103, 153, 247, 378, 645, 1352, 2919]))
    pointsets.append(pointSet("2003_UL",[0,73, 103, 153, 247, 378, 645, 1352, 2919]))
    pointsets.append(pointSet("2102_UL",[0,73, 103, 153, 247, 378, 645, 1352, 2919]))
    
    pointsets.append(pointSet("1002_UL",[0, 103, 153, 247, 378, 645, 1352, 2919]))
    pointsets.append(pointSet("1003_UL",[0,73, 103, 153, 247, 378, 645, 1352, 2919]))
    pointsets.append(pointSet("1102_UL",[0,73, 103, 153, 247, 378, 645, 1352, 2919]))
    
    pointsets.append(pointSet("2002_UR",[0,10,30,74, 103, 149,243]))
    pointsets.append(pointSet("2003_UR",[0,10,30,74, 103, 149,243]))
    pointsets.append(pointSet("2102_UR",[0,10,30,74, 103,243, 386, 639, 1536])) #the 149 is flaud
    
    pointsets.append(pointSet("1002_UR",[0,10,30,74, 103, 149, 386, 639, 1536]))
    pointsets.append(pointSet("1003_UR",[0,10,30,74, 103, 149,243, 386, 639, 1536]))
    pointsets.append(pointSet("1102_UR",[0,10,30,74, 103, 149,243]))
    
    pointsets.append(pointSet("3003_UL",[0,10,30,74, 103, 149,243, 386, 639, 1536]))
    pointsets.append(pointSet("3007_UL",[0,10,30,74, 103, 149,243, 386, 639, 1536]))
    pointsets.append(pointSet("3008_UL",[0,10,30,74, 103, 149,243, 386, 639, 1536]))
    
    
    return pointsets, kneepointset
    
    
    

