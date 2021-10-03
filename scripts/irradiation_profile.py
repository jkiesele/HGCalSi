#!/usr/bin/python3

from argparse import ArgumentParser
import numpy as np
import math
parser = ArgumentParser()
parser.add_argument('radtime')
args = parser.parse_args()
radtime=float(args.radtime)

from diodes import radiation_annealing_Ljubljana

print('equivalent time at 60 degrees ',radiation_annealing_Ljubljana(radtime,False))

    
    


