#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:19:23 2022

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from pycode.qmatrix import genQMatrices

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

# QDelta
sweepType = 'BE'

# Quadrature type (wether or not include the right/left bound in the nodes)
# -- 'GAUSS', 'RADAU-I' (left), 'RADAU-II' (right), 'LOBATTO'
quadType = 'LOBATTO'

# Node distribution, can be
# -- 'EQUID', 'LEGENDRE', 'CHEBY-1', 'CHEBY-2', 'CHEBY-3', 'CHEBY-4'
nodeType = 'LEGENDRE'

# Number of nodes
M = 3

num = 10

coll = genQMatrices(M, nodeType, quadType, sweepType)

norm = np.linalg.norm(coll['Q'] - coll['QDelta'], ord=np.inf)

QmQDelta = coll['Q'] - coll['QDelta']

xi = np.linspace(-1, 1, num)

p = cartesian_product(*[xi]*M)

norms = np.linalg.norm(QmQDelta.dot(p.T),axis=0)


plt.figure()
plt.plot(norms)
plt.xlabel('$i$')
plt.ylabel(r'$||(Q-Q_\Delta){\bf x_i}||$')
plt.tight_layout()


i1 = np.argmax(norms)
norms[i1] = 0
i2 = np.argmax(norms)
norms[i2]

print(p[i1])
print(p[i2])
