

import numpy as np
import matplotlib.pyplot as plt

#test

'''
varset

keys

list

asso

need:
  private properties, e.g.
  startpoint, endpoint
  
  
in fit:
- add all sets
- get var list as init

in fit function:
- get current var list
- apply back to varsets
- take each VarSet, loop
   - evaluate functions given x ranges (given by len of varset)
   - apply evaluation to y (append)
- return y=concat(y)

after fit:
- apply list back to varsets
'''


from jax import numpy as jnp 
class VarSet(object):
    '''
    Has points and the fit function definition
    '''
    def __init__(self,
                 xs,
                 ys,
                 yerr,
                 xerr=None,
                 varsdict:dict ={},
                 constdict:dict={}):
        self.xs=np.array(xs)
        self.ys=np.array(ys)
        self.yerr=np.squeeze(np.array(yerr))
        self.vars=varsdict
        self.constdict = constdict
        self.xerr = None
        if xerr is not None:
            self.xerr = np.array(xerr)
           
    
        
    def len(self):
        return len(self.xs)
    
    def debugPlot(self):
        plt.errorbar(self.xs,self.ys,self.yerr,self.xerr,linewidth=0,marker='o',elinewidth=1)
        
    def debugPlotEval(self):
        xs = np.arange(np.min(self.xs),np.max(self.xs),2)
        plt.plot(xs,self.eval(xs))
    #maybe only in derived classes?
    def NC(self,x):
        return self.vars['g_c']*self.constdict['phi']*jnp.ones_like(x) + self.constdict['NC0']
    
    def NA(self,x):
        return self.constdict['phi']*self.vars['g_a'] * jnp.exp(-x / (jnp.abs(self.vars['tau_a'])+1e-9))
    
    def NY(self,x):
        return self.vars['g_y']*self.constdict['phi'] * (1. - 1./(1 + x/self.vars['tau_y']))
    
    def eval(self,x):
        return self.NC(x)+self.NA(x)+self.NY(x)
        
    
       
class VarSetSet(object):
    def __init__(self,
                 global_keys=[
                     
                     ]):
        #which are to be connected
        self.global_keys = global_keys
        self._association={} #(list idx, (key, [varset idxs]))
        self.varsets=[]
        self.scaling=None
        self.yscale=1e12
        '''
        for k in keys
        if k in global
        else
        '''
    
        
    def addVarSet(self, varset: VarSet):
        self.varsets.append(varset)
        self._associate()
    
    def sliceX(self,x):
        out=[]
        last=0
        for v in self.varsets:
            out.append(x[last:v.len()+last])
            last=v.len()
        return out
    
    #list index, key, set indices
    def _associate(self):
        asso={} # key to list index only for globals
        assoidx={}# (list idx, (key, [varset idxs]))
        usedkeys=[]
        for i in range(len(self.varsets)):
            for k in self.varsets[i].vars:
                if k in self.global_keys and k in usedkeys:
                    lidx = asso[k]
                    assoidx[lidx][1].append(i)
                    continue
                elif k in self.global_keys:
                    asso[k]=len(assoidx)
                
                assoidx[len(assoidx)]=(k,[i])
                usedkeys.append(k)
        
        self._association = assoidx
        
    def getVarList(self, returndict=False):
        v = [0. for _ in range(len(self._association))]
        ks = ["" for _ in range(len(self._association))]
        for ik in self._association.keys():
            k,vids = self._association[ik]
            var=0.
            for i in vids:
                var+=self.varsets[i].vars[k]
            var /= float(len(vids))
            v[ik] = var
            ks[ik]=k
        if not returndict:
            return v
        else:
            d={}
            for k,vv in zip(ks,v):
                d[k]=vv
            return d
    
    def applyVarList(self,l,unc=None):
        assert len(l)==len(self._association)
        if unc is not None:
            assert len(unc)==len(l)
        for ik in self._association.keys():
            k,vids = self._association[ik]
            for i in vids:
                self.varsets[i].vars[k] = l[ik]
                if unc is not None:
                    self.varsets[i].vars[k+'_std'] = unc[ik]
        
    def createStartScale(self):
        v = jnp.array(self.getVarList())
        self.scaling = v
        
    def scaleFittedVars(self, v):
        return jnp.array(v)*self.scaling
    
    def allSetChi2(self, v):
        self.applyVarList(v*self.scaling)

        y = self.allSetYs()/self.yscale
        ys=[]
        for i in range(len(self.varsets)):
            ys.append(self.varsets[i].eval(self.varsets[i].xs)/self.yscale)
        ys = jnp.concatenate(ys,axis=0)
        diff = (ys-y)**2/(self.allSetYErrs()/self.yscale)#DEBUG
        return jnp.sum(diff)
    
        
    def allSetFuncODR(self,v,x):
        self.applyVarList(v)
        xs = self.sliceX(x)
        ys=[]
        for i in range(len(xs)):
            ys.append(self.varsets[i].eval(xs[i]))
        out= np.concatenate(ys,axis=0)
        return out
    
    def allSetXs(self):
        xs = []
        for vs in self.varsets:
            xs.append(vs.xs)
        return np.concatenate(xs,axis=0)
    
    def allSetXErrs(self):
        xerrs = []
        for vs in self.varsets:
            if vs.xerr is None:
                return None
            else:
                xerrs.append(vs.xerr)
        return np.concatenate(xerrs,axis=0)
    
    def allSetYs(self):
        ys=[]
        for vs in self.varsets:
            ys.append(vs.ys)
        return np.concatenate(ys,axis=0)
    
    def allSetYErrs(self):
        yerrs=[]
        for vs in self.varsets:
            yerrs.append(vs.yerr)
        return np.concatenate(yerrs,axis=0)
    


def test():
    
    import scipy.odr as odr
    xs = np.arange(0,10)
    ys = np.random.rand(10)
    yerr = np.sqrt(ys)/3.
    
    vs1 = VarSet(
        xs,ys,yerr,None,
        {'A':1., 'B': 2.,})
    ys = np.random.rand(10)
    vs2 = VarSet(
        xs,ys+3.,yerr,None,
        {'A':1.1, 'B': 2.1, })
    
    import matplotlib.pyplot as plt
    
    
    vvs = VarSetSet(['B'])
    vvs.addVarSet(vs1)
    print(vvs._association)
    vvs.addVarSet(vs2)
    print(vvs._association)
    print(vvs.getVarList())
    
    vvs.applyVarList([5.,6.,7.])
    for vs in vvs.varsets:
        print(vs.vars)
        vs.debugPlot()
    plt.show()
    
    
    m = odr.Model(vvs.allSetFuncODR)
    
    mydata = odr.RealData(vvs.allSetXs(), vvs.allSetYs(), 
                          sx=vvs.allSetXErrs(), 
                          sy=vvs.allSetYErrs())
    
    myodr = odr.ODR(mydata, m, beta0=vvs.getVarList())
    out=myodr.run()
    print(out)
    popt = out.beta
    print(popt)
    pcov = out.sd_beta
    perr = out.sd_beta
    print(perr)
    
    vvs.applyVarList(popt)
    
    for v in vvs.varsets:
        v.debugPlot()
        v.debugPlotEval()
        plt.show()
        plt.close()
    





    
    