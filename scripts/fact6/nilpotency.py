#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 2023

@author: gaya
"""
import numpy as np

from pycode.qmatrix import genCollocation

# change these:
############################
M = 3
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

Dinv = []
for m in range(1, M + 1, 1):
    D = np.diag(nodes) / m
    Dinv_ = D.copy()
    for i in range(M):
        if D[i, i] != 0:
            Dinv_[i, i] = 1 / D[i, i]
    Dinv.append(Dinv_)

pow = np.eye(M)
for m in range(M):
    pow = (np.eye(M) - Dinv[m] @ Q) @ pow
    print('|prod(I - Dinv[m] Q)|_max = {:<7.4e}, m = 1, ..., {:<3}        ro(prod) = {}'.format(np.max(np.abs(pow)), m+1, max(np.abs(np.linalg.eigvals(pow)))))

print('----------------')
p = np.eye(M)
for m in range(M):
    p = (np.eye(M) - np.linalg.solve(np.diag(np.diag(Q)), Q)) @ p
    print('|prod(I - Dinv[m] Q)|_max = {:<7.4e}, m = 1, ..., {:<3}        ro(prod) = {}'.format(np.max(np.abs(p)), m+1, max(np.abs(np.linalg.eigvals(p)))))