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
quadType = 'RADAU-RIGHT'
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

S = (Q - D)
b = np.ones(M)

# power iterations without rescaling
for m in range(2*M):
    b_new = S @ b
    ro = np.dot(b_new, b) / np.dot(b, b)
    print('m = {}, ro(Q - D) = {}'.format(m, ro))
    b = b_new.copy()
