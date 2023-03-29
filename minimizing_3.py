#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:12:22 2022

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

sweepType = 'FE'
quadType = 'GAUSS'
nodeType = 'EQUID'


coll = genQMatrices(3, nodeType, quadType, sweepType)

norm = np.linalg.norm(coll['Q'] - coll['QDelta'], ord=np.inf)

QmQDelta = coll['Q'] - coll['QDelta']

x1 = np.linspace(-1, 1, num=100)
x2 = np.linspace(-1, 1, num=101)


x3 = np.linspace(-1, 1, num=101)
maxNorm = []

for val in x3:
    p = cartesian_product(x1, x2)
    p = np.concatenate((p, np.repeat(val, p.shape[0])[:, None]), axis=1)
    norms = np.linalg.norm(QmQDelta.dot(p.T),axis=0)
    maxNorm.append(np.max(norms))
    norms.shape = (x1.size, x2.size)


plt.figure()
plt.plot(x3, maxNorm)
plt.xlabel('$x_3$')
plt.ylabel(r'$max||(Q-Q_\Delta){\bf x}||$')
plt.tight_layout()


for val in [-1, 1]:

    p = cartesian_product(x1, x2)
    p = np.concatenate((p, np.repeat(val, p.shape[0])[:, None]), axis=1)
    norms = np.linalg.norm(QmQDelta.dot(p.T),axis=0)
    maxNorm.append(np.max(norms))
    norms.shape = (x1.size, x2.size)

    plt.figure()
    plt.pcolor(x1, x2, norms.T)
    plt.colorbar()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(r'$||(Q-Q_\Delta){\bf x}||, \quad x_3='+f'{val}'+'$')
    plt.tight_layout()
