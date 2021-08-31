'''
Created on 31 Aug 2021

@author: jkiesele
'''

import ctypes
import os

_file = 'libalpha.so'
_path = os.path.join(*(os.path.split(__file__)[:-1] + (_file, )))
_mod = ctypes.cdll.LoadLibrary(_path)


_alpha = _mod._Z5alphadd
_alpha.argtypes = (ctypes.c_double, ctypes.c_double)
_alpha.restype = ctypes.c_double

def alpha(temperature_in_C : float, time_in_minutes : float):
    return _alpha(temperature_in_C,time_in_minutes)

_find_intersection = _mod._Z17find_intersectiondddd
_find_intersection.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
_find_intersection.restype = ctypes.c_double


def equiv_annealing_time(temperature_in_C : float, time_in_minutes : float, target_temperature_in_C : float,
                         rel_epsilon: float =1e-5):
    return _find_intersection(temperature_in_C,time_in_minutes, target_temperature_in_C,rel_epsilon)
