'''
Created on 31 Aug 2021

@author: jkiesele
'''

import ctypes
import os

_file = 'libalpha.so'
_path = os.path.join(*(os.path.split(__file__)[:-1] + (_file, )))
_mod = ctypes.cdll.LoadLibrary(_path)


_alpha = _mod._Z5alphaddd
_alpha.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
_alpha.restype = ctypes.c_double

def alpha(temperature_in_C : float, time_in_minutes : float, t_0 :float = 1.):
    return _alpha(temperature_in_C,time_in_minutes,t_0)

_find_intersection = _mod._Z17find_intersectionddddd
_find_intersection.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
_find_intersection.restype = ctypes.c_double


def equiv_annealing_time(temperature_in_C : float, time_in_minutes : float, target_temperature_in_C : float,
                         rel_epsilon: float =1e-5, start_alpha :float =-1.):
    
    #negligible and will also break the intersect calc
    if temperature_in_C < -10 and target_temperature_in_C > 20 and time_in_minutes < 20:
        return 0.
    intersect = _find_intersection(temperature_in_C,time_in_minutes, target_temperature_in_C,rel_epsilon,start_alpha)
    if intersect < 0:
        raise RuntimeError("equiv_annealing_time: could not find intersection for", temperature_in_C, 'C,', time_in_minutes,'min')
        return 0.
    return intersect