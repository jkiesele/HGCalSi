
from diodes import diodes
import glob
import os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt

from plotting import curvePlotter




datadir=os.getenv("DATAOUTPATH")

class point(object):
    def __init__(self, diodequali: str, minutes: int, depl_freq, fileprefix='extracted', directory=None):
        self.diodequali=diodequali #3007_UL
        self.minutes=minutes
        self.fileprefix=fileprefix
        self.diode=diodes[diodequali[:4]]
        self.depl_freq=depl_freq

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
                raise ValueError("multiple directories found with "+checkdir+':\n'+str(l))
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
        if self.depl_freq == 10000:
            return self.data['depletion_nominal'], self.data['depletion_nominal']-self.data['depletion_down'], self.data['depletion_up']-self.data['depletion_nominal']
        else:
            return 0,0,0


class pointSet(object):
    def __init__(self, diodequali, minutes,depl_freq, fileprefix='extracted'):
        self.points=[]
        self.diodequali=diodequali
        for t in minutes:
            self.points.append(point(diodequali,t,depl_freq,fileprefix=fileprefix))

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
        return idx[0][0], good
    
    @staticmethod
    def modeIsVoltage(mode):
        if mode == "Udep" or  mode == "NEff"  or mode == "NEffSlope":
            return True
        return False
    
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
                #print('depl voltage', p.diode.label(),p.data['depletion_nominal'])
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
                #print(nom,down,up)
                n,u,d = self.diode().NEff(-nom),self.diode().NEff(-(nom-down)),self.diode().NEff(-(nom+up))
                u=u-n
                d=n-d
                #neff error
                _,nu,nd = self.diode().NEff(-nom,return_unc=True)
                
                u = np.sqrt( (nu)**2 + (u)**2 )
                d = np.sqrt( (nd)**2 + (d)**2 )
                
                ys.append(n)
                yerrdown.append(d)
                yerrup.append(u)
                
            elif mode == "I" or mode == "Itot" or mode == "IperFluence"  or mode == "ItimesThickness" or mode == "IperThickness" or mode == "IperVolume":
                #get closest x point (should be no problem, high res enough
                xnp = np.array(p.data['iv_x_smooth'])
                ynp = np.array(p.data['iv_y_smooth'])
                if mode == "Itot":
                    ynp = np.array(p.data['iv_y_tot_smooth'])
                    

                idx, good = self._get_closest_point(xnp, current_at[i])
                if not good:
                    print('could not get point for', current_at[i],'V', p.diode.label(),' - maybe just out of measurement range')
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
                    y *= p.diode.thickness_cm
                    yup *= p.diode.thickness_cm
                    ydown *= p.diode.thickness_cm
                if mode == "IperThickness":
                    y /= p.diode.thickness_cm
                    yup /= p.diode.thickness_cm
                    ydown /= p.diode.thickness_cm
                if mode == "IperVolume":
                    y /= p.diode.thickness_cm*p.diode.area
                    yup /= p.diode.thickness_cm*p.diode.area
                    ydown /= p.diode.thickness_cm*p.diode.area
                ys.append(y)
                yerrdown.append(ydown)
                yerrup.append(yup)
            
            xs.append(x)
            
            relxerr = 0.025
            if (mode=='Udep' or mode == "NEff") and x > 720:
                print("WARNING: Hard-conded adding 5% uncertainty on time because of time constant conversion")
                relxerr = np.sqrt(relxerr**2 + 0.05**2)
            
            xerrdown.append(float(math.sqrt(p.diode.ann_offset_error**2 + (relxerr*x)**2)))
            xerrup.append(float(math.sqrt(p.diode.ann_offset_error**2 + (relxerr*x)**2))) 
            
        return np.array(xs),np.array([xerrdown,xerrup]),np.array(ys),np.array([yerrdown,yerrup])

    @staticmethod
    def symmetrisePoints(xs,xerrs,ys,yerrs):
        xerr = np.array(xerrs)
        xerr = np.abs(xerr)
        xerr = np.max(xerr,axis=0)
        yerr = np.array(yerrs)
        yerr = np.abs(yerr)
        yerr = np.max(yerr,axis=0)
        
        return xs,np.array([xerr,xerr]),ys,np.array([yerr,yerr])


#scratch area
'''
pointset is a set of measurements for one sample. can give I Udepl, in points

convert to parametrised dependence. 


'''
class interpolatedPointSet(object):
    
    def __init__(self,
                 pointset : pointSet,
                 mode: str,
                 **kwargs, #this also defines interpolation mode
                 ):
        self.ps = pointset
        self._infunc = None
        self.mode = mode
        #do extraction here, leaving points with errors
        self.x,self.xerrs,self.y,self.yerrs = self.ps.getXYs(mode=mode,**kwargs)
        
        self.xerrs = np.array(self.xerrs) #2 x X
        self.yerrs = np.array(self.yerrs) #2 x X
        
        #print('self.yerrs',self.yerrs.shape)
        
        if not pointSet.modeIsVoltage(mode):
            self.getY = self._currentInterpolate
        else:
            self.getY = self._voltageInterpolate
        
  
    def _currentInterpolate(self,x):
        '''
        linear interpolation in log space
        
        returns: interpolated y point with [up,down] error and bool
        wether it is within allowed interpolation range
        '''
        
        if self._infunc is None:
            #create interpolation
            from scipy.interpolate import interp1d
            logx = np.log(self.x)
            infunc = interp1d(logx, self.y, kind='linear')
            self._infunc = lambda xin : (infunc(np.log(xin)))
        
        return self._interpolate(x)
    
    
    def _getYErrToPoint(self, pidx, inty):
        
        closesty = self.y[pidx]
        ydiff = inty-closesty
        
        closestyerr = np.array(self.yerrs[:,pidx])
        #make this symmetric
        #if ydiff>0: #add to down
        closestyerr[0] = float(math.sqrt(closestyerr[0]**2 + 1./4.*ydiff**2)) #add half the difference
        #else:
        closestyerr[1] = float(math.sqrt(closestyerr[1]**2 + 1./4.*ydiff**2)) #add half the difference
        
        return closestyerr
        
    def _interpolate(self,x):
        
        
        from tools import closestPointIdx
        closestidx = closestPointIdx(self.x, x)
        next_to_closestidx = closestPointIdx(self.x, x, closestidx)
        
        #print(closestidx,self.x[closestidx],x,next_to_closestidx,self.x[next_to_closestidx])
        #is below range
        if closestidx == 0 and self.x[closestidx] - x > 0:
            return 0.,[0.,0.], False
        #is above range
        if closestidx == len(self.x)-1 and x - self.x[closestidx] > 0:
            return 0.,[0.,0.], False
        
        #sanity check, should not trigger
        if self.x[closestidx] > x and self.x[next_to_closestidx] > x or \
           self.x[closestidx] < x and self.x[next_to_closestidx] < x :
            raise RuntimeError("can't be", x, self.x[closestidx], self.x[next_to_closestidx])
        
        
        x0 = self.x[closestidx]
        x1 = self.x[next_to_closestidx]
        
        reldistanceto0 = np.abs(x - x0)/(np.abs(x1 - x0)+1e-12)
        reldistanceto1 = np.abs(x1 - x)/(np.abs(x1 - x0)+1e-12)
        
        #the rest is within range
        #print(x)
        inty = self._infunc(x) #
        closestyerr  = reldistanceto1 * np.expand_dims(self._getYErrToPoint(closestidx,inty),axis=1) # 2 x 1
        nclosestyerr = reldistanceto0 * np.expand_dims(self._getYErrToPoint(next_to_closestidx,inty),axis=1)
        
        #print(x0, x, x1)
        #print(closestyerr, nclosestyerr)
        #print(' ')
        
        closestyerr = np.sum(np.concatenate([closestyerr,nclosestyerr],axis=-1),axis=-1)
        
        #exit()
        #print('closestyerr',closestyerr.shape)
        
        return inty, closestyerr, True        
     
    def _voltageInterpolate(self,x):
        
        if self._infunc is None:
            from fitting import AnnealingFitter
            fitter = AnnealingFitter()
            x,xerr,y,yerr = pointSet.symmetrisePoints(self.x,self.xerrs,self.y,self.yerrs) 
            
            y,yerr = self.diode().NEff(y), self.diode().NEff(yerr)
            
            xerr = xerr[0]
            yerr = yerr[0]
            fitter.fit(x, y, xerr, yerr)
            
            self._infunc = lambda xin : self.diode().VDep(fitter.DNeff(xin))
            
        return self._interpolate(x)
     
    
    def interpolateArray(self, xs):
        xouts,ys,yerrs = [],[],[]
        for x in xs:
            y,yerr,valid = self.getY(x)
            if valid:
                xouts.append(x)
                ys.append(y)
                yerrs.append(yerr)
        return np.array(xouts),np.array(ys),np.transpose(np.array(yerrs),[1,0])#matplotlib error format
        
    def diode(self):
        return self.ps.points[0].diode
    
    
'''
scratch: alphaExtraction class takes interpolated point sets
'''

class pointSetsContainer(object):
    def __init__(self):
        self.pointsets={}
        
    def append(self, ps):
        self.pointsets[ps.diodequali]=ps
        
    def getSet(self,whichset):
        return self.pointsets[whichset]
    
    def getSets(self, whichsets):
        out={}
        for k,v in zip(self.pointsets.keys(),self.pointsets.values()):
            if k in whichsets:
                out[k] = v
        return out
        
    def getXYsDiodes(self,mode, whichsets,
                     current_at=None):
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
                't_odown': x - ps.diode().ann_offset + ps.diode().ann_offset_up,
                't_oup': x - ps.diode().ann_offset + ps.diode().ann_offset_down,
                'y':y,
                'yerr':yerrs
                })
        return xyerss
    
    def getInterpolatedXYsDiodes(self,mode,whichsets,current_at=None,debugplots=None):
        out=[]
        for aset in whichsets:
            ps = self.pointsets[aset]
            ips = interpolatedPointSet(ps,mode,current_at=current_at)
            
            if debugplots is not None:
                x,xerr,y,yerr = ps.getXYs(mode,current_at)
                
                y_scaler = 1.
                if 'y_scaler' in debugplots.keys():
                    y_scaler = debugplots['y_scaler']
                
                ixs,iys,iyerrs = ips.interpolateArray(np.arange(np.min(x),np.max(x)))
                
                plt.close()
                plt.errorbar(ixs,iys*y_scaler,yerr=iyerrs*y_scaler,label='interpolated',
                             linewidth=2,elinewidth=2.,ecolor='tab:gray',color='tab:red')
                
                plt.errorbar(x,y*y_scaler,yerr=yerr*y_scaler,xerr=xerr,label='measured points',
                             linewidth=0,elinewidth=2.,marker='o',color='tab:orange')
                
                plt.xlabel(debugplots['xlabel'])
                plt.ylabel(debugplots['ylabel'])
                plt.legend()
                plt.tight_layout()
                plt.savefig(debugplots['outfile']+aset+'.pdf')
                plt.close()
                
            
            out.append({
                'pointset': ps,#for bookkeeping
                'diode': ps.diode(),
                'f(t)': ips.getY
                })
            
        return out
        
    def addToPlot(self,mode, whichsets,add=[],marker='o',colors=None,linestyle='',
                  linewidth=None,current_at=None, add_rel_y_unc=None,labels=None,
                  labelmode='', scale_y=1):
        while len(add) < len(whichsets):
            add.append("")
        if colors is not None and colors == 'fluence':
            colors=[]
            for aset in whichsets:
                col = self.pointsets[aset].diode().fluenceCol()
                colors.append(col)
        if colors is None:
            colors = [None for _ in whichsets]
        if labels is None:
            labels = [None for _ in whichsets]
        x=None    
        for aset,addstr,c,l in zip(whichsets,add,colors,labels):
            ps = self.pointsets[aset]
            x,xerr,y,yerr = ps.getXYs(mode,current_at)
            if add_rel_y_unc is not None:
                yerr = np.array(yerr)
                yerr = np.sign(yerr)*np.sqrt( yerr**2 + (np.expand_dims(y,axis=0)*add_rel_y_unc)**2 )
            if l is None:
                l = ps.diode().paperlabel(labelmode)+addstr
            y = np.array(y)
            yerr = np.array(yerr)
            y *= scale_y
            yerr *= scale_y
            plt.errorbar(x, y,  yerr=yerr, xerr=xerr,label=l,
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


def loadAnnealings(depl_freq=10000):
    pointsets=pointSetsContainer()

    pointsets.append(pointSet("6002_6in",[0,11,31,74, 103, 149, 386, 639, 1536],depl_freq=depl_freq))
    
    kneepointset=pointSetsContainer()
    kneepointset.append(pointSet("6002_6in",[0,11,31,74, 103, 149, 386, 639],fileprefix="firstknee",depl_freq=depl_freq))
    
    pointsets.append(pointSet("2002_UL",[0,73, 103, 153, 247, 378, 645, 1352, 2919],depl_freq=depl_freq))
    pointsets.append(pointSet("2003_UL",[0,73, 103, 153, 247, 378, 645, 1352, 2919],depl_freq=depl_freq))
    pointsets.append(pointSet("2102_UL",[0,73, 103, 153, 247, 378, 645, 1352, 2919],depl_freq=depl_freq))
    
    pointsets.append(pointSet("1002_UL",[0, 103, 153, 247, 378, 645, 1352, 2919],depl_freq=depl_freq))
    pointsets.append(pointSet("1003_UL",[0,73, 103, 153, 247, 378, 645, 1352, 2919],depl_freq=depl_freq))
    pointsets.append(pointSet("1102_UL",[0,73, 103, 153, 247, 378, 645, 1352, 2919],depl_freq=depl_freq))
    
    pointsets.append(pointSet("2002_UR",[0,10,30,74, 103, 149,243],depl_freq=depl_freq))
    pointsets.append(pointSet("2003_UR",[0,10,30,74, 103, 149,243],depl_freq=depl_freq))
    pointsets.append(pointSet("2102_UR",[0,10,30,74, 103,243, 386, 639, 1536],depl_freq=depl_freq)) #the 149 is flaud
    
    pointsets.append(pointSet("1002_UR",[0,10,30,74, 103, 149, 386, 639, 1536],depl_freq=depl_freq))
    pointsets.append(pointSet("1003_UR",[0,10,30,74, 103, 149,243, 386, 639, 1536],depl_freq=depl_freq))
    pointsets.append(pointSet("1102_UR",[0,10,30,74, 103, 149,243],depl_freq=depl_freq))
    
    pointsets.append(pointSet("3003_UL",[0,10,30,74, 103, 149,243, 386, 639, 1536],depl_freq=depl_freq))
    pointsets.append(pointSet("3007_UL",[0,10,30,74, 103, 149,243, 386, 639, 1536],depl_freq=depl_freq))
    pointsets.append(pointSet("3008_UL",[0,10,30,74, 103, 149,243, 386, 639, 1536],depl_freq=depl_freq))
    
    
    return pointsets, kneepointset
    
    
    

