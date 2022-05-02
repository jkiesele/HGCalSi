#!/usr/bin/python3

from argparse import ArgumentParser
import numpy as np
import math
parser = ArgumentParser()
parser.add_argument('radtime')
args = parser.parse_args()
radtime=float(args.radtime)

from diodes import radiation_annealing_Ljubljana

print('equivalent time at 60 degrees 45C',radiation_annealing_Ljubljana(radtime,True))
print('equivalent time at 60 degrees 50C',radiation_annealing_Ljubljana(radtime,True,50))
print('equivalent time at 60 degrees 55C',radiation_annealing_Ljubljana(radtime,True,55))

    
    


