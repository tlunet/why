#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:10:18 2023

@author: telu
"""
import numpy as np
import scipy as sp
from pycode.qmatrix import genCollocation

M = 5
quadType = 'LOBATTO'
distr = 'LEGENDRE'


# RADAU-RIGHT, GAUSS until 5
# RADAU-LEFT until 4

nodes, _, Q = genCollocation(M, distr, quadType)

if quadType in ['LOBATTO', 'RADAU-LEFT']:
    Q = Q[1:, 1:]
    nodes = nodes[1:]
    M = M-1
    
def f(x):
    c = 1
    y = np.empty_like(x)
    K = np.diag(1/np.array(x)) @ Q - np.eye(M)
    for i, z in enumerate(range(M)):
        y[i] = np.linalg.det(c * np.eye(M) + c/(z+1) * K) - c ** M
    return y


sol = sp.optimize.root(f, nodes/M, tol=1e-14, method='hybr')
print(sol)
x = sol.x
print()
print('x = ', x)
K = np.diag(1/x) @ Q - np.eye(M)
Kpow = np.eye(M)
for i in range(M):
    Kpow = K @ Kpow
    print('|prod(I - Dinv @ Q)|_max = {:<7.4e}, m = 1, ..., {:<3}   ro(prod) = {}'.format(
        np.max(np.abs(Kpow)), i + 1, max(np.abs(np.linalg.eigvals(Kpow)))))



