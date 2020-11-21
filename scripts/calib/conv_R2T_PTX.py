
from sympy.solvers import solve
from sympy import Symbol

def R2T_PTX_ITS90(R,R0):
    t = Symbol('t')
    A = 3.9083E-3
    B = -5.7750E-7
    C = 0.
    if R < R0 : C = -4.183E-12
    T = solve(R-R0*(1+A*t+B*t*t+C*(t-100)*t*t*t),t)
    return T[0]

def R2T_PTX_IPTS68(R,R0):
    t = Symbol('t')
    A = 3.90802E-3
    B = -5.80195E-7
    C = 0.
    if R < R0 : C = -4.27350E-12
    T = solve(R-R0*(1+A*t+B*t*t+C*(t-100)*t*t*t),t)
    return T[0]


R0 = 1000
for R in range(800,1300):
    T1 = R2T_PTX_ITS90(R,R0)
    T2 = R2T_PTX_IPTS68(R,R0)
    print(R, T1, T2, (T1-T2)/(T1+T2)*2*100, '%')

