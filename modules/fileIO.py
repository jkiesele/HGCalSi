'''

reading (and writing) of files

'''
import numpy as np

def calibration(temp):
    return temp

def readTempLog(filename):
    x=[]
    y=[]
    with open(filename) as file:
        for line in file:
            if len(line) < 1:
                continue
            sline = line.split(' ')
            x.append(int(sline[0]) )
            tempstr=sline[1][:-1]
            if tempstr[-4:] == ".625":
                tempstr=tempstr[:-4]+".0625"
            y.append(float(tempstr))
            
    tmp = np.array(y)
    return np.array(x), calibration(tmp)



class fileReader(object):
    def __init__(self, mode="CV", path=""):
        self.mode=mode
        self.path=path
        
    
    def readCV(self,filename):
        infile = open(self.path+filename, 'r')
        lines = infile.read().splitlines()
        fill = False
        V=[]
        C=[]
        kappa=[]
        for line in lines:
            if fill==True:
                if line == 'END': break
                info = line.split('\t')
                V.append(float(info[0]))
                C.append(float(info[1]))
                kappa.append(float(info[2]))
        
            if line == 'BEGIN': fill = True
        
        return np.array(V),np.array(C),np.array(kappa)
    
    def readIV(self,filename):
        infile = open(self.path+filename, 'r')
        lines = infile.read().splitlines()
        fill = False
        V=[]
        I=[]
        for line in lines:
            if fill==True:
                if line == 'END': break
                info = line.split('\t')
                V.append(float(info[0]))
                I.append(float(info[2]))
        
            if line == 'BEGIN': fill = True
        return np.array(V),np.array(I), None


    def read(self, filename):
        if self.mode == "CV":
            return self.readCV(filename)
        elif self.mode == "IV":
            return self.readIV(filename)
        
        

    
    
        