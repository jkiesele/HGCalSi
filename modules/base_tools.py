

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc

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



class DNeff_calc(object):
    def __init__(self,initvars: list, constdict: dict):
        '''
        give vars as list of tuples ('name', value)
        '''
        assert len(initvars)>0
        assert isinstance(initvars[0],tuple)
        assert len(initvars[0])==2
        self.asso_i_to_str={ i: initvars[i][0] for i in range(len(initvars))}
        self.asso_str_to_i={ initvars[i][0]: i  for i in range(len(initvars))}
        self.vars={ t[0]:t[1] for t in initvars}
        
        self.constdict=constdict
        
    def name_to(self,name : str):
        return self.asso_str_to_i[name]
    
    def list_to_dict(self,varl):
        return { k: varl[self.asso_str_to_i[k]] for k in self.vars.keys() }
    
    def start_vals(self):
        return [ self.vars[self.asso_i_to_str[i]] for i in range(len(self.vars)) ]
        
    def NC(self,x, varl):
        return varl[self.name_to('g_c')]*self.constdict['phi']*jnp.ones_like(x) + self.constdict['NC0']
    
    def NA(self,x, varl):
        return self.constdict['phi']*varl[self.name_to('g_a')] * jnp.exp(-x / (jnp.abs(varl[self.name_to('tau_a')])+1e-9))
    
    def NY(self,x, varl):
        return varl[self.name_to('g_y')]*self.constdict['phi'] * (1. - 1./(1 + x/varl[self.name_to('tau_y')]))
    
    def eval(self,x, varl):
        return self.NC(x,varl)+self.NA(x,varl)+self.NY(x,varl)
    
    def plot(self,x,varl,label=None):
        #plt.plot(x,self.NC(x,varl),label=r'$N_C$')
        #plt.plot(x,self.NA(x,varl),label=r'$N_A$')
        #plt.plot(x,self.NY(x,varl),label=r'$N_Y$')
        plt.plot(x,self.eval(x,varl),label=label)
    
class fitPoints(object):
    
    def __init__(self,x,y,yerr, functclass, direct_neighbour_corr = 0.5):
        self.x=dc(x)
        self.y=dc(y)
        self.yerr=dc(yerr)
        self.xerr=None
        self.functclass=functclass
        self.covinv = np.diag(np.ones_like(x))
        
        #just do brute force
        for i in range(len(x)):
            for j in range(len(x)):
                if abs(i-j)==1:
                    is_break = False
                    if i>j:
                        is_break = x[i]<x[j]
                    if j>i:
                        is_break = x[j]<x[i]
                    if not is_break:
                        self.covinv[i][j]=direct_neighbour_corr
                 
        #print(self.covinv)   
        # self.yerr: D
        # self.cov: D x D
        self.covinv = np.expand_dims(self.yerr,axis=0)* self.covinv * np.expand_dims(self.yerr,axis=1)
        self.covinv = np.linalg.inv(self.covinv)
        
    def enlargeYErrs(self):
        self.yerr *= 10.
        
    def chi2(self,varl):
        
        y = self.y
        x = self.x
        ys = self.functclass.eval(x,varl)
        delta = ys-y
        chi2 = jnp.expand_dims(delta,axis=0) * self.covinv * jnp.expand_dims(delta,axis=1)
        return jnp.sum(chi2)
    
    def plot(self,varl=None, label=None):
        #np.logspace()
        xs = np.logspace(np.log(np.min(self.x)),np.log(np.max(self.x)),base=np.exp(1),num=200)
        if varl is not None:
            self.functclass.plot(xs,varl, label= label)
        #plt.errorbar(self.x,self.y,self.yerr,self.xerr,linewidth=0,marker='o',elinewidth=1,label='data',color='k')
        

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
        v = jnp.array(self.getVarList())*0. + 1.
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
        diff = (ys-y)**2/(self.allSetYErrs()/self.yscale)**2#DEBUG
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
    





    
    