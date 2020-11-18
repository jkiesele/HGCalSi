import math

def convertToCs(Cp, kappa, freq=10000):
    omega = math.pi * freq
    Rp = 1/kappa #Cp/kappa
    Cs = (1. + omega**2 * Rp**2 * Cp**2)/(omega**2 * Rp**2 * Cp)
    return Cs