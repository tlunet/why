#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:14:50 2023

@author: cpf5546
"""
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt

from pycode.qmatrix import genCollocation, genQDelta


M = 2
tau = sy.symbols(",".join([f"t_{i+1}" for i in range(M)]))
di = sy.symbols(",".join([f"d_{i+1}" for i in range(M)]))
if M == 1:
    tau = [tau]
    di = [di]

V = sy.Matrix([[t**i for i in range(M)] for t in tau])
W = sy.Matrix([[t**(i+1)/(i+1) for i in range(M)] for t in tau])
D = sy.diag(*di)

S = D @ V - W

# L, U, _ = S.LUdecomposition()
# L.simplify()
# U.simplify()

# P = np.eye(M) - (D @ V).inv() @ W

z = sy.Symbol("z")

eq = (1-z)*np.eye(M, dtype=int) + z * (D @ V).inv() @ W


if M == 2 and False:
    S2 = S.subs(di[1], sy.prod(tau)/2/di[0])
    f = S2.LUdecomposition()[1][-1,-1]

    d1_1, d1_2 = sy.solve(f, di[0])

    d2_1 = (sy.prod(tau)/2/d1_1).simplify()
    d2_2 = (sy.prod(tau)/2/d1_2).simplify()

    S_1 = S.subs(di[0], d1_1).subs(di[1], d2_1)
    S_1.simplify()

    L_1, U_1, _ = S_1.LUdecomposition()
    L_1.simplify()
    U_1.simplify()

    S_2 = S.subs(di[0], d1_2).subs(di[1], d2_2)
    S_2.simplify()

    L_2, U_2, _ = S_2.LUdecomposition()
    L_2.simplify()
    U_2.simplify()

    a_1 = -(tau[1]*(1-d2_1*tau[1]/2))/(1-d2_1*tau[1])
    a_2 = -(tau[1]*(1-d2_2*tau[1]/2))/(1-d2_2*tau[1])


    quadType = 'RADAU-RIGHT'
    distr = 'LEGENDRE'
    nodes, _, Q = genCollocation(M, distr, quadType)
    qDelta, _ = genQDelta(nodes, 'MIN-SR-S', Q, None, None)

    d = np.linspace(-2, 2, num=1000)
    d1, d2 = np.meshgrid(d, d)
    t1, t2 = nodes


    f = t2*(d1-t1)*(d2-t2/2)-(d2-t2)*t1*(d1-t1/2)

    f2 = d1*d2

    plt.figure("f")
    plt.contourf(d1, d2, f)
    plt.colorbar()
    plt.contour(d1, d2, f, levels=[0])
    plt.contour(d1, d2, f2, levels=[t1*t2/2])

    plt.plot(*np.diag(qDelta), 'o', c='red')

if M == 3 and False:
    SR = S.copy()

    d1, d2, d3 = di
    t1, t2, t3 = tau

    SR[0, :] *= (d2-t2)*(d3-t3)
    SR[1, :] *= (d1-t1)*(d3-t3)
    SR[2, :] *= (d1-t1)*(d2-t2)

    SR[1, :] -= SR[0, :]
    SR[2, :] -= SR[0, :]

    SR[1, :] -= SR[2, :]

    # SR[1, :] *= SR[2, 1]
    # SR[2, :] *= SR[1, 1]

    # SR[2, :] -= SR[1, :]
