#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:12:22 2022

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from qmatrix import genQMatrices

sweepType = 'BE'
quadType = 'LOBATTO'
nodeType = 'LEGENDRE'


coll = genQMatrices(2, nodeType, quadType, sweepType, scaling=True)

norm = np.linalg.norm(coll.Q - coll.QDelta, ord=np.inf)

QmQDelta = coll.Q - coll.QDelta

x1 = np.linspace(-1, 1, num=100)
x2 = np.linspace(-1, 1, num=101)

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


p = cartesian_product(x1, x2)
norms = np.linalg.norm(QmQDelta.dot(p.T),axis=0)

norms.shape = (x1.size, x2.size)

plt.pcolor(x1, x2, norms.T)
plt.colorbar()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(r'$||(Q-Q_\Delta){\bf x}||$')
plt.tight_layout()
