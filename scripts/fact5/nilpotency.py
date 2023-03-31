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
M = 3
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

S = (Q- D)
powS = np.eye(M)
for m in range(M+1):
    print('|(Q - D)^{}|_max = {}'.format(m, np.max(np.abs(powS))))
    powS = powS @ S
