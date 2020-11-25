'''

reading (and writing) of files

'''
import numpy as np
import glob

def calibration(temp):
    return temp

def readTempLog(filename, calib=False, readLast=True):
    x=[]
    y=[]
    z=[]
    with open(filename) as file:
        for line in file:
            if len(line) < 1:
                continue
            sline = line.split(' ')
            if readLast and int(sline[0]) == 1:
                x=[]
                y=[]
                z=[]
            x.append(int(sline[0]) )
            tempstr=sline[1]
            if tempstr[-4:] == ".625":
                tempstr=tempstr[:-4]+".0625"
            y.append(float(tempstr))
            if calib:
                z.append(float(sline[2]))
            
    tmp = np.array(y)
    if calib:
        return np.array(x), calibration(tmp), np.array(z)
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
    
    def readIV(self,filename,GR=False):
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
                if GR:
                    I.append(float(info[1])-float(info[2]))
                else:    
                    I.append(float(info[2]))
        
            if line == 'BEGIN': fill = True
        return np.array(V),np.array(I), None


    def read(self, filename):
        if "*" in filename:
            all=glob.glob(self.path+filename)
            if len(all) == 0:
                print("file", self.path+filename, " not found")
            filename = all[-1]
            print('auto select last file:',filename)
            filename = filename[len(self.path):]
            
        
        
        if self.mode == "CV":
            return self.readCV(filename)
        elif self.mode == "IV":
            return self.readIV(filename)
        elif self.mode == "IVGR":
            return self.readIV(filename,GR=True)
        
        

    
    
        