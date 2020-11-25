import math
import numpy as np

def convertToCs(Cp, kappa, freq=10000):

    #Cp=np.abs(Cp)
    #kappa=np.abs(kappa)
    omega = math.pi * freq
    Rp = 1/kappa 
    Cs = (1. + omega**2 * Rp**2 * Cp**2)/(omega**2 * Rp**2 * Cp)
    return Cs




'''

'''

'''
no_anneal
V_detector [V]  Capacitance [F] Conductivity [S]        Bias [V]        Curre
BEGIN
-8.727273E+2    1.342080E-11    1.823960E-8     -8.727273E+2    -2.461609E-5
-8.818182E+2    1.341960E-11    1.775680E-8     -8.818182E+2    -2.478013E-5
-8.909091E+2    1.341810E-11    1.750070E-8     -8.909091E+2    -2.486393E-5
-9.000000E+2    1.341720E-11    1.734480E-8     -9.000000E+2    -2.507583E-5
'''


'''
7min
V_detector [V]  Capacitance [F] Conductivity [S]        Bias [V]        Curren

-8.698305E+2    -3.279593E-12   -7.811066E-6    -8.698305E+2    -2.146823E-5
-8.849153E+2    -3.280993E-12   -7.810917E-6    -8.849153E+2    -2.170022E-5
-9.000000E+2    -3.281693E-12   -7.810734E-6    -9.000000E+2    -2.202053E-5
'''