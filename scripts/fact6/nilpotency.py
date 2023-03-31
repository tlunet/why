#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 21:39:39 2023

@author: gaya
"""
import numpy as np

from pycode.qmatrix import genCollocation

# change these:
############################
M = 5
quadType = 'GAUSS'
distr = 'LEGENDRE'
############################

def spectralRadius(x):
    x = np.asarray(x)
    R = Q - np.diag(x)
    return np.max(np.abs(np.linalg.eigvals(R)))

nodes, _, Q = genCollocation(M, distr, quadType)

print('quadType = {}\ndistr = {}\nM = {}'.format(quadType, distr, M))
print('define: D = diag(nodes) / M\n')
D = np.diag(nodes) / M

Dinv = []
for m in range(M, 0, -1):
    D = np.diag(nodes) / m
    Dinv_ = D.copy()
    for i in range(M):
        if D[i, i] != 0:
            Dinv_[i, i] = 1 / D[i, i]
    Dinv.append(Dinv_)

pow = np.eye(M)
for m in range(M):
    pow = (np.eye(M) - Dinv[m] @ Q) @ pow
    print('|prod(I - Dinv[m] Q)|_max = {:<7.4e}, m = 1, ..., {}'.format(np.max(np.abs(pow)), m+1))