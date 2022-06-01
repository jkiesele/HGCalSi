

import numpy
numpy.set_printoptions(linewidth=numpy.nan)

from numpy import arange
import numpy as np
import math
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot
from sklearn.cluster import KMeans 
from scipy.optimize import fsolve
from fileIO import fileReader
from plotting import curvePlotter

from lmfit.models import LinearModel
import sklearn.gaussian_process as gp
from matplotlib import pyplot as plt
import scipy.odr as odr
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pickle

class Covariance(object):
    
    def __init__(self,
                 errs=[],
                 assumed_corr=0
                 ):
        
        errarr = np.array(errs)
        corr = np.zeros((len(errarr),len(errarr)))+assumed_corr
        corr = corr*(1.-np.diag(np.ones_like(errarr)))+np.diag(errarr*0.+1.)
        
        self.data=None
        self.makeCovariance(errarr,corr)
        
        
    def makeCovariance(self, errs, corrmatrix):
        errarr = np.array(errs)
        cerra = np.expand_dims(errarr,axis=0)
        cerrb = np.expand_dims(errarr,axis=1)
        
        cov = cerra * corrmatrix * cerrb
        self.data=cov
        
    def set(self, nparr):
        assert nparr.shape[0]==nparr.shape[1]
        self.data=nparr
        
    def getCorrelations(self):
        errs = self.getErrs()
        errsa = np.expand_dims(errs,axis=0)
        errsb = np.expand_dims(errs,axis=1)
        return self.data/errsa/errsb
    
    def data(self):
        return self.data
    
    def __str__(self):
        return np.round(self.data,2).__str__()
    
    def getErrs(self):
        return np.sqrt(np.diagonal(self.data))
    
    
    def __add__(self, other):
        cvnew = Covariance([])
        cvnew.data = self.data + other.data
        return cvnew
    
    def append(self, other, correlation=0):
        allerrs = np.concatenate([self.getErrs(), other.getErrs()],axis=0)
        thiscorr = self.getCorrelations()
        othercorr = other.getCorrelations()
        thiscorrp = np.pad(thiscorr, [0,othercorr.shape[1]],constant_values=correlation)[:thiscorr.shape[1]]
        othercorrp = np.pad(othercorr, [thiscorr.shape[1],0],constant_values=correlation)[thiscorr.shape[1]:]
        appended = np.concatenate([thiscorrp,othercorrp],axis=0)
        
        self.makeCovariance(allerrs, appended)
        
        
class MeasurementWithCovariance(object):
    def __init__(self,
                 data,
                 global_correlation=0.
                 ):
        '''
        data format:
        {'diode': ps.diode(),
                't': x,
                'terr':xerrs,
                'y':y,
                'yerr':yerrs
                })
        '''
        self.x=np.array(data['t'])
        self.y=np.array(data['y'])
        yerr=0
        if len(data['yerr'].shape):
            yerr = np.squeeze(data['yerr'][:,0])
        else:
            yerr = data['yerr']
        self.xerr=np.array(data['terr'])
        
        
        #build a covariance
        self.ycovariance = Covariance(yerr,global_correlation)
    
    def getCovariance(self):
        return self.ycovariance
    
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
    
    def getXErrs(self):
        return self.xerr
    #
    # create individual measurement with:
    #   - partially correlated syst. uncertainty
    # then 
    #   - append the measurements, same syst
    #   - add fluence uncertainty by measurement
    #
    #
    
class MeasurementSet(object):
    def __init__(self,globalcorr=0.):
        self.measurements=[]
        self.globalcorr=globalcorr
    
    def addMeasurement(self, meas: MeasurementWithCovariance):
        self.measurements.append(meas)
    
    def getCombinedData(self, addfluence_unc=0.1, correlatesyst=True):
        #return x,y,and full cov
        #create individual measurement mask
        
        combcov = None
        fluencecov=None
        ys = []
        xs =[]
        xerrs=[]
        correlatedsyst=None
        if len(self.measurements)<2:
            addfluence_unc=0.#n0t for single measurements
        for m in self.measurements:
            y = m.getY()
            ys.append(y)
            xs.append(m.getX())
            xerrs.append(m.getXErrs())
            
            if correlatedsyst is None:
                correlatedsyst = m.getCovariance().getCorrelations()[0,1]
            else:
                thiscorr =  m.getCovariance().getCorrelations()[0,1]
                if correlatesyst and (not abs(thiscorr-correlatedsyst)<0.00001):
                    raise ValueError("getCombinedData: correlatesyst only works if correlations are the same, here"
                                     +str(thiscorr)+' vs '+str(correlatedsyst))
            
            fluenceerr = y*addfluence_unc
            if fluencecov is None:
                fluencecov = Covariance(fluenceerr,1.)
            else:
                fluencecov.append(Covariance(fluenceerr,1.))
                                  
            cm = m.getCovariance() 
            if combcov is None:
                combcov=cm
            else:
                if correlatesyst:
                    combcov.append(cm, correlatedsyst)
                else:
                    combcov.append(cm)
                
        xs = np.concatenate(xs,axis=0)
        ys = np.concatenate(ys,axis=0)
        xerrs = np.concatenate(xerrs,axis=0)
        
        #add covariance matrices
        if addfluence_unc>0:
            totalcov = combcov + fluencecov
        else:
            totalcov = combcov
            
        return {
            'x':xs,
            'y':ys,
            'xerr': xerrs,
            'cov':totalcov.data
            }
        






def smoothen(x,y):
    from scipy.interpolate import interp1d
    
    ysmooth = savgol_filter(y, 5,3)
    f2 = interp1d(x, ysmooth, kind='cubic')
    xint = np.arange(x[-1],x[0],1)
    #flip?
    yint = f2(xint)
    
    return xint, yint
 


class SimpleAnnealingFitter(object):
    def __init__(self,
                 x,
                 y,
                 xerr,
                 ycov
                 ):
        self.fittedparas={}
        
        self._x = x 
        self._y = y
        self._xerr = xerr
        self._ycov = ycov
        
        self.NA0_scale = 1e-3 * 1e15 
        self.tau_a_scale = 10.
        self.NY0_scale = 1e14
        self.tau_y_scale = 1e4
        self.NC0_scale = 1e13
        self.yscale = 1e0
    
    def _NEfffunc(self, B, t):
        NA0,tau_a, NY0, tau_y, t0, NC0 = B
        
        NA0*=self.NA0_scale
        tau_a*=self.tau_a_scale
        NY0*=self.NY0_scale
        tau_y*=self.tau_y_scale
        NC0*=self.NC0_scale
        
        #t+=t0
        
        N_A = NA0 * np.exp(-t / tau_a)
        N_Y = NY0 * (1. - 1./(1 + t/tau_y))
        #print(N_A.shape, N_Y.shape)
        ret = (N_A + NC0 + N_Y)/self.yscale
        return ret
    
    def _spo_NEfffunc(self, t, *B):
        return self._NEfffunc(B,t)
    
    def NEff(self,t,phi,NEff0,NC):
        d = self.fittedparas
        return self.yscale*self._NEfffunc([d['g_a']*phi/self.NA0_scale,
                               d['tau_a']/self.tau_a_scale,
                               d['g_y']*phi/self.NY0_scale,
                               d['tau_y']/self.tau_y_scale,
                               d['t0'],
                               d['N_C+N_{Eff,0}']/self.NC0_scale,
                               ], 
                              t)
    
    def fit(self, 
            phi, 
            useodr=False):
        
        
        startpara = np.array([
                    1., #self._y[0]-np.min(self._y),   #NA0,
                    1., #15.,   #tau_a, 
                    1., #np.max(self._y)-np.min(self._y),   #NY0, 
                    1., #1e3,   #tau_y, 
                    0.,#t0
                    1. #np.min(self._y)   #NC0
             ])
        
        popt, pcov, perr=None,None,None
        
        if useodr:
            m = odr.Model(self._NEfffunc)
            
            mydata = odr.RealData(self._x, self._y/self.yscale, 
                                  sx=self._xerr, sy=np.sqrt(np.diagonal(self._ycov))/self.yscale)
            
            myodr = odr.ODR(mydata, m, beta0=startpara)
            out=myodr.run()
            #print(out)
            popt = out.beta
            pcov = out.sd_beta
            perr = out.sd_beta
            
        else:             
            popt, pcov = curve_fit(self._spo_NEfffunc, self._x, self._y/self.yscale, 
                      p0=startpara,
                      sigma=self._ycov/self.yscale, 
                      absolute_sigma=True,
                      bounds = [6*[0.],4*[1e6]+[0.1]+[1e6]],
                      method='trf'
                      )
            perr = np.sqrt(np.diag(pcov))
        
        NA0,tau_a, NY0, tau_y, t0, NC0 = popt
        sdNA0,sdtau_a, sdNY0, sdtau_y, sdt0, sdNC0 = perr
        
        self.cov = pcov
        
        self.fittedparas={
            'g_a': NA0/phi*self.NA0_scale, 
            'tau_a': tau_a*self.tau_a_scale, 
            'N_C+N_{Eff,0}': NC0*self.NC0_scale, 
            'g_y': NY0/phi*self.NY0_scale, 
            'tau_y': tau_y*self.tau_y_scale,
            't0':t0,
            
            'sd_g_a': sdNA0/phi*self.NA0_scale, 
            'sd_tau_a': sdtau_a*self.tau_a_scale, 
            'sd_N_C+N_{Eff,0}': sdNC0*self.NC0_scale, 
            'sd_g_y': sdNY0/phi*self.NY0_scale, 
            'sd_tau_y': sdtau_y*self.tau_y_scale,
            'sd_t0':sdt0
            }
        
    def getParameters(self):    
        return self.fittedparas
    def collapseNC(self):
        pass
        

class AnnealingFitter(object):
    def __init__(self, useodr=False):
        self._t0 = 0.
        self._g_a=0.02
        self._tau_a=15.
        self._N_C =1e12
        #self._c = 5e-13
        #self._g_c = 1.5e-2
        self._g_y = 0.05
        self._tau_y=1e3
        
        self._t = None
        self._terr = None
        self._NEff = None
        self._NEfferr = None
        self._phi =None
        self._NEff0 = None
        
        self._sdt0 = None
        self._sd_t = None
        self._sd_terr = None
        self._sd_NEff = None
        self._sd_NEfferr = None
        self._sd_phi =None
        self._sd_NEff0 = None
        self._splits = [0]
        
        self.useodr=useodr
        
    def setInputAndFit(self,t,terr, NEffin,NEfferr,phi,NEff0, splits=None):
        assert t.shape == NEffin.shape and NEffin.shape == phi.shape and phi.shape == NEff0.shape
        assert t.shape == terr.shape
        assert NEffin.shape[0] == NEfferr.shape[0]
        
        self._t=t 
        self._terr = terr
        self._NEff = NEffin
        self._NEfferr = NEfferr
        self._phi = phi
        self._NEff0 = NEff0
        self._splits=None
        if splits is not None:
            self._splits = splits
        #self._NEfffunc([self._g_a, self._tau_a, self._N_C, 
        #                self._c, self._g_c, self._g_y, 
        #                self._tau_y],
        #               self._t)
        
        self._fit()
        #print(self._g_a, self._tau_a, self._N_C, self._c, self._g_c, self._g_y, self._tau_y)
        
    def _NEfffunc(self, B, t):
        if (not self._splits is None) and len(self._splits) > 1:
            g_a, tau_a, g_y, tau_y, t0, *N_C = B
        else:
            g_a, tau_a, g_y, tau_y, t0, N_C = B
        return self.__NEff( g_a, tau_a, g_y, tau_y, N_C, t, self._phi, self._NEff0,t0)
    
    def _spo_NEfffunc(self, t, *B):
        if (not self._splits is None) and len(self._splits) > 1:
            g_a, tau_a, g_y, tau_y, t0, *N_C = B
        else:
            g_a, tau_a, g_y, tau_y, t0, N_C = B
        return self.__NEff( g_a, tau_a, g_y, tau_y, N_C, t, self._phi, self._NEff0,t0)
        
    def __NEff(self, g_a, tau_a, g_y, tau_y, N_C, t, phi, NEff0,t0=0,tileNC=True):
        #expand N_C
        #print(len(N_C),len(self._splits))
        t+=t0
        N_Ctiled = []
        if isinstance(N_C, float):
            tileNC=False
        if tileNC:
            for i in range(len(self._splits)-1):
                n = self._splits[i+1]-self._splits[i]
                N_Ctiled.append(np.ones(n, dtype='float')*N_C[i])
            N_Ctiled = np.concatenate(N_Ctiled,axis=0)
        else:
            N_Ctiled = N_C
        
        N_A = phi*g_a * np.exp(-t / tau_a)
        N_Y = g_y*phi * (1. - 1./(1 + t/tau_y))
        #print(N_A.shape, N_Y.shape)
        return N_A + N_Ctiled + N_Y + NEff0
        
    def NEff(self,t,phi,NEff0,NC):
        g_a, tau_a, g_y, tau_y, N_C = self._g_a, self._tau_a, self._g_y, self._tau_y, NC
        return self.__NEff( g_a, tau_a, g_y, tau_y, N_C, t, phi, NEff0,tileNC=False)
     
    def collapseNC(self):
        self._N_C = float(np.mean(self._N_C))
        self._sd_N_C  = float(np.mean(self._sd_N_C ))

    #gives back all variations
    def dNEff(self,t,phi,NEff0):    
        dneff=[]
        varsarr = np.array([self._g_a,    self._tau_a,    self._g_y,    self._tau_y, self._N_C ]       )
        sds  = self.cov
        
        #totalerr=0.
        #for i in range(len(varsarr)):
        #    for j in range(len(sds)):
        #        
        #    mask = np.zeros_like(sds)
        #    mask[i] = 1.
        #    thesevars = varsarr + mask*sds
        #    up = self.__NEff(*thesevars, t,phi,NEff0)
        #    mask[i] = -1.
        #    thesevars = varsarr + mask*sds
        #    down = self.__NEff(*thesevars, t,phi,NEff0)
        #    dneff.append([up,down])
        return dneff


    def spo_fit(self):
        #curve_fit
        N_Cs = [self._N_C]
        if self._splits is not None:
            N_Cs = [self._N_C for _ in range(len(self._splits)-1)]

        mydata = None
        if len(self._NEfferr)>1:
            mydata = odr.RealData(self._t, self._NEff, sx=self._terr, covy=self._NEfferr)
        else:
            mydata = odr.RealData(self._t, self._NEff, sx=self._terr, sy=self._NEfferr)
        
        startbeta = np.array([ self._g_a, self._tau_a, self._g_y, self._tau_y, self._t0] +  N_Cs)
                 
        popt, pcov = curve_fit(self._spo_NEfffunc, self._t, self._NEff, 
                  p0=startbeta,
                  sigma=self._NEfferr, 
                  absolute_sigma=True
                  )
        perr = np.sqrt(np.diag(pcov))
        
        self._g_a, self._tau_a,  self._g_y, self._tau_y, self._t0, *self._N_C = popt
        self._sd_g_a, self._sd_tau_a, self._sd_g_y, self._sd_tau_y, self._sdt0, *self._sd_N_C = perr
        
        self.cov = pcov
        
    def _fit(self):
        if self.useodr:
            self.odr_fit()
        else:
            self.spo_fit()
        
        
    def odr_fit(self):

        m = odr.Model(self._NEfffunc)
        
        N_Cs = [self._N_C]
        if self._splits is not None:
            N_Cs = [self._N_C for _ in range(len(self._splits)-1)]

        mydata = None
        if len(self._NEfferr)>1:
            mydata = odr.RealData(self._t, self._NEff, sx=self._terr, sy=np.sqrt(np.diagonal(self._NEfferr)))
        else:
            mydata = odr.RealData(self._t, self._NEff, sx=self._terr, sy=self._NEfferr)
        
        startbeta = np.array([ self._g_a, self._tau_a, self._g_y, self._tau_y, self._t0] +  N_Cs)
        myodr = odr.ODR(mydata, m, beta0=startbeta)
        out=myodr.run()
        #print(out)
        popt = out.beta
        self._g_a, self._tau_a,  self._g_y, self._tau_y, self._t0, *self._N_C = popt
        
        errs = out.sd_beta
        self._sd_g_a, self._sd_tau_a, self._sd_g_y, self._sd_tau_y, self._sdt0, *self._sd_N_C = errs
        
        self.cov = out.cov_beta
        

        
    def getParameters(self):    
        return {
            't0': self._t0, 
            'g_a': self._g_a, 
            'tau_a': self._tau_a, 
            'N_C+N_{Eff,0}': self._N_C, 
            'g_y': self._g_y, 
            'tau_y': self._tau_y,
            
            
            'sd_t0': self._sdt0, 
            'sd_g_a': self._sd_g_a, 
            'sd_tau_a': self._sd_tau_a, 
            'sd_N_C+N_{Eff,0}': self._sd_N_C, 
            'sd_g_y': self._sd_g_y, 
            'sd_tau_y': self._sd_tau_y
            }
        
    
    
class alphaExtractor(object):
    def __init__(self, interpolated_pointset_dict : dict,
                 rel_fluence_uncertainty=0.1
                 ):
        '''
        takes a list of dict of interpolated point sets.
        dict contains:
                'pointset': ps,#for bookkeeping
                'diode': ps.diode(),
                'f(t)': ips.getY
        
        '''
        self.interpolated_pointset_dict =interpolated_pointset_dict
        self.rel_fluence_uncertainty = rel_fluence_uncertainty
        
    def extractForTimes(self, timearr_in_min):
        pass
    
    def _createFitInputForTime(self, t_in_min : float):
        
        diodes=[]
        fluences = []
        currents = []
        currenterrup = []
        currenterrdown = []
        for psd in self.interpolated_pointset_dict:
            curr,errs,valid = psd['f(t)'](t_in_min)
            if valid:
                fluences.append(psd['diode'].rad)
                diodes.append(psd['diode'])
                currents.append(curr)
                currenterrdown.append(errs[0])
                currenterrup.append(errs[1])
        
        xs = np.array(fluences)
        ys = np.array(currents) * 1e-2/1e-6 #from µm to cm
        yerrs = np.max(np.array([currenterrdown,currenterrup]),axis=0)* 1e-2/1e-6 
        
        return xs,ys,yerrs,diodes
    
    def extractAlphaForTime(self, t_in_min : float, plot=False):
        
        def linear(a,x):
            return a*1e-19*x
        
        xs,ys,yerrs,diodes = self._createFitInputForTime(t_in_min)
        if len(xs)<1:
            print('no data found for', t_in_min)
            return None, None
        
        if plot:
            for i,d in enumerate(diodes):
                plt.errorbar(xs[i],ys[i],yerrs[i],xs[i]*self.rel_fluence_uncertainty,
                             c=d.thicknessCol(),label=d.thickness_str(),marker='o')
                
        m = odr.Model(linear)
        mydata = odr.RealData(xs, ys, sy=yerrs, sx = xs*self.rel_fluence_uncertainty)
        myodr = odr.ODR(mydata, m, beta0=[7.])
        out=myodr.run()
        #print(out.beta)
        #print(out.sd_beta)
        minx = np.min(xs)
        maxx = np.max(xs)
        alpha = round(out.beta[0],2)
        alphaerr = round(out.sd_beta[0],2)
        
        #the fluence is worst case a correlated uncertainty, needs to be considered here:
        #alphaerr = math.sqrt(alphaerr**2 + (alpha*self.rel_fluence_uncertainty)**2)
        
        if plot:
            plt.plot([minx,maxx],[linear(out.beta,minx),linear(out.beta,maxx)],
                 )#label=r"$\alpha$(fit)="+str(alpha)+"$\pm$"+str(alphaerr)+"$10^{-19}$A/cm")
        
        if alphaerr:
            return alpha,alphaerr
        else:
            return None, None
        
    def extractAlpha(self, time_array_minutes, plotstring=None):
        
        doplot = False
        if plotstring is not None:
            doplot=True
            
        alphas, alphaserrs, validtimes =[],[],[]
        for i,t in enumerate(time_array_minutes):
            plt.close()
            a,aerr = self.extractAlphaForTime(t,plot=doplot)
            if doplot and a is not None:
                
                handles, labels = plt.gca().get_legend_handles_labels()
                unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
                
                plt.legend(*zip(*unique))
                
                plt.xlabel("Fluence [neq/cm$^2$]")
                plt.ylabel("I(U$_{dep}$)/Volume [A/cm$^3$] @ -20˚C")
                plt.tight_layout()
                plt.savefig(plotstring+'_'+str(t)+'min.pdf')
                plt.close()
            if a is not None:
                validtimes.append(t)
                alphas.append(a)
                alphaserrs.append(aerr)
        return np.array(alphas),np.array(alphaserrs),np.array(validtimes)
        

class Linear(object):
    def __init__(self, a=None,b=None):
        self._a=a 
        self.b=b 
        
    def __call__(self,x,a=None,b=None):
        if a is None and b is None:
            return 1e19*self._a*(x/10 + 9) + self.b*1e22
        else:
            return 1e19*a*(x/10 + 9) + b*1e22
        
    def getA(self):
        return 1e19*self._a
    
    def getB(self):
        return 1e22*self.b
    
    
class DoubleLinear(object):
    def __init__(self,
        a0=None         ,
        b0=None         ,
        a1=None         ,
        b1=None         ,
        mixpoint=None   ,
        mixsmooth=None
        ):
        self.a0=a0
        self.b0=b0
        self.a1=a1
        self.b1=b1
        self.mixpoint=mixpoint
        self.mixsmooth=mixsmooth
        
    def __call__(self,x,a0=None,b0=None, a1=None,b1=None, mixsmooth=None, mixpoint=None):
        if a0 is None:
            a0=self.a0
            b0=self.b0
            a1=self.a1
            b1=self.b1
            mixsmooth=self.mixsmooth
            mixpoint=self.mixpoint
            
        import tensorflow as tf
            
        lin0 = 1e19* a0*(x/10 + 9) +  b0*1e22
        lin1 = 1e19* a1*(x/10 + 9) +  b1*1e22
        sigx = mixsmooth * x/100. - mixpoint
        sig = 1. / (1 + tf.exp(-sigx))
        return lin0*sig + lin1*(1.-sig)
        

def readDepletionData(directory, file="extracted.dv"):
    import os
    allpath = os.path.join(directory,file)
    with open(allpath, 'rb')  as filehandler:
        return pickle.load(filehandler)
    
 
class DepletionFitter(object):
    def __init__(self,  x=None, y=None, 
                 const_cap=None,
                 rising=2, constant=8,
                 debugfile=None,
                 low_start=None,
                 low_end=None,
                 high_start=None,
                 high_end=None,
                 varcut=10,
                 strictcheck=True,
                 interactive=False,
                 addextracuncat=-1000,
                 extraunc=10.,
                 cideal= None,):
        
        self.varcut=varcut
        self.strictcheck=strictcheck
        self.data=None
        self.rising = rising
        self.constant = constant
        self.const_cap = const_cap
        self.debugfile = debugfile
        self.addextracuncat=addextracuncat
        self.extraunc=extraunc
        if x is not None and y is not None:
            self._setData(x,y)
            
        self.low_start=low_start
        self.low_end=low_end
        self.high_start=high_start
        self.high_end=high_end
        
        self.interactive=interactive
        self.cideal = cideal
        
        #save info
        self.data.update(
            {'low_start': low_start,
             'low_end': low_end,
             'high_start': high_start,
             'high_end': high_end,
             'varcut': varcut,
             'const_cap': const_cap
             }
            )
        
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
        
        self._smoothen()
        
    def _checkchange(self,c):
        delta = abs(self.data['x'][0]-self.data['x'][1])#min one point distance
        if not self.strictcheck:
            return c
        if not c == 0 and delta > abs(c):
            if c>0:
                c=delta
            else:
                c=-delta
        return c
        
    def _findcut(self):
        #simple comparison of average slope and current slope
        
        
        x = self.data['x_smooth']
        #y = self.data['y_smooth']
        
        #create smoothed data here
        
        outsel=[]
        for s in [[self.low_start, self.low_end], [self.high_start,self.high_end]]:
            defstart, defend = s[0],s[1]
            thissel=[]
            for var1 in [0, -self.varcut, self.varcut]:
                for var2 in [ 0, -self.varcut, self.varcut]:
                    
                    change1 = abs(defstart-defend)*(var1/100.)
                    change2 = abs(defstart-defend)*(var2/100.)
                    
                    start = defstart + self._checkchange(change1)
                    end = defend + self._checkchange(change2)
                    
                    sel = np.logical_and(x >= start, x < end)
                    thissel.append(sel)
            outsel.append(thissel)
             
        return outsel   
        #for lowstart in [self.low_start, self.low_start-self.varcut,self.low_start+self.varcut]:
        #    for lowend in [self.low_start, self.low_start-self.varcut,self.low_start+self.varcut]:
        #
        #constant = np.logical_and(self.data['x'] >= self.low_start,self.data['x'] < self.low_end)
        #constup =     np.logical_and(self.data['x'] >= self.low_start+20,self.data['x'] < self.low_end+20)
        #constdown = np.logical_and(self.data['x'] >= self.low_start-20,self.data['x'] < self.low_end-20)
        #    
        #rising = np.logical_and(self.data['x'] >= self.high_start,self.data['x'] < self.high_end)
        #riseup
        #risedown
        #
        #return rising, constant, riseup, risedown, constup, constdown
       
    def _linear(self, x, a, b): 
        return a * x + b
    
    def _dofit(self, debugplot=False, savedatapath=None):
        #
        # assume cuts
        # fit lines, store
        # get value, store
        #
        cons,rise = self._findcut()
        #each [sela, selb, ...]
        riselines=[]
        conslines=[]
        for srise in rise:
            for scons in cons:
                x = self.data['x_smooth'][srise]
                y = self.data['y_smooth'][srise]
                popt, _ = curve_fit(Linear(), x, y)
                a, b = popt
                line = Linear(a=a, b=b)
                riselines.append(line)
                
                if self.const_cap is not None and self.const_cap > 0:
                    #print('using constant ',self.const_cap)
                    conslines.append(Linear(a=0., b=self.const_cap/1e22))
                    conslines.append(Linear(a=0., b=self.const_cap*(1.-0.002*self.varcut)/1e22))
                    conslines.append(Linear(a=0., b=self.const_cap*(1.+0.002*self.varcut)/1e22))
                    continue
                
                x = self.data['x_smooth'][scons]
                y = self.data['y_smooth'][scons]
                popt, _ = curve_fit(Linear(), x, y)
                a, b = popt
                line = Linear(a=a, b=b)
                conslines.append(line)
                
        depl=[]
        for r,c in zip(riselines,conslines):
            vdep  = fsolve(lambda x : r(x) - c(x),-1000)[0]
            depl.append(vdep)
            if vdep < self.addextracuncat:
                depl.append(vdep+(vdep-self.addextracuncat)*(self.extraunc/100.))
                depl.append(vdep-(vdep-self.addextracuncat)*(self.extraunc/100.))
            
        depl=np.array(depl)
        nom,up,down = depl[0], np.max(depl), np.min(depl)
        print('[',nom,',',up-nom,',',nom-down,']')
        
        if debugplot:
            #plt.close()
            plt.plot(self.data['x'],self.data['y'],marker='x',linewidth=None,label='data')
            plt.plot(self.data['x_smooth'],self.data['y_smooth'],label='smoothened')
            if self.cideal is not None:
                plt.plot( [np.min(self.data['x']),np.max(self.data['x'])], 
                          [1/self.cideal**2,1/self.cideal**2],label=r'C_{end} (ideal)')
            plt.legend()
            for l in riselines+conslines:
                plt.plot(self.data['x'],l(self.data['x']),linewidth=0.5)
            plt.xlabel("U [V]")
            plt.ylabel("$1/C^2 [1/F^2]$")
            plt.ylim([np.min(self.data['y'])/1.1, np.max(self.data['y'])*1.2])
            
                
            fig = plt.gcf()
            if savedatapath is not None:
                fig.savefig(savedatapath+'_fig.pdf')
            if self.interactive:
                plt.show()
            
            
        if savedatapath is not None:
            d = {'depletion_nominal':nom,
                 'depletion_up': up,
                 'depletion_down': down,
                 'riselines': riselines,
                 'conslines': conslines,
                 }
            d.update(self.data)
            
            with open(savedatapath, 'wb')  as filehandler:
                pickle.dump(d, filehandler)
            
            
                
        return nom,up,down
            
            
        #############    
            
        rise, cons = self._findcut()
        funcs=[]
        for r in [rise,cons]:
            x = self.data['x'][r]
            y = self.data['y'][r]
            
            if len(x) < 3:
                print(len(x))
                raise ValueError("cut offs wrongly set")
            if debugplot:
                plt.plot(x,y,marker='x')
            
            popt, _ = curve_fit(Linear(), x, y)
            a, b = popt
            print(a,b)
            #exit()
            line = Linear(a=a, b=b)
            funcs.append(line)
            if self.const_cap is not None:
                funcs.append(Linear(a=0, b=self.const_cap))
                break
            
        self._smoothen()
        if debugplot:
            
            l0,l1 = funcs
            print('cap:' ,l1.getB())
            #plt.close()
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
            
    def getDepletionVoltage(self,debugplot=False, withunc=False, savedatapath=None):
        if withunc:
            return self._dofit(debugplot=debugplot,savedatapath=savedatapath)
        return self._dofit(debugplot=debugplot,savedatapath=savedatapath)[0]
    
    def _smoothen(self):
        x = self.data['x']
        y = self.data['y']
        xint,yint = smoothen(x, y)
        self.data['x_smooth']=xint
        self.data['y_smooth']=yint
        
        
    

class DepletionFitterInflect(object):
    def __init__(self):
        pass
    
    
 
 
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
    def __init__(self, globalpath, file=None, debug=False, debugfile=None, useGP=False, **kwargs):
        self.fitter=PolFitter()
        if useGP:
            self.fitter=GPFitter()
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
#def objective(x, _a, b):
#    return _a * x + b
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
#_a, b = popt
#print('y = %.5f * x + %.5f' % (_a, b))
## plot input vs output
#pyplot.scatter(x, y)
## define _a sequence of inputs between the smallest and largest known inputs
#x_line = arange(min(x), max(x), 1)
## calculate the output for the range
#y_line = objective(x_line, _a, b)
## create _a line plot for the mapping function
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
        