#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:29:21 2022

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from qmatrix import genQMatrices

sweepType = 'FE'
quadType = 'GAUSS'
nodeType = 'EQUID'


print('n, ||Q-QDelta||, max(QDelta)')

nValues = [5, 10, 20, 50, 100, 200, 300] #, 100, 200, 500]

norms = []
maxDelta = []
minDelta = []

for n in nValues:
    coll = genQMatrices(n, nodeType, quadType, sweepType)

    norm = np.linalg.norm(coll.Q - coll.QDelta, ord=np.inf)
    norms.append(norm)
    maxDelta.append(np.max(coll.QDelta))
    minDelta.append(np.max(coll.QDelta))

    print(n, norm, np.max(coll.QDelta))


maxDelta = np.array(maxDelta)
minDelta = np.array(minDelta)
nValues = np.array(nValues)

plt.figure()
plt.loglog(nValues, norms, label='Norms')
plt.loglog(nValues, maxDelta, label='MaxDelta')
plt.loglog(nValues, maxDelta**(7/8), ':o', label='MaxDelta**(7/8)')
# plt.loglog(nValues, nValues*maxDelta**2, ':s', label='M*MaxDelta**2')
plt.loglog(nValues, (np.pi/(2*nValues+2))**(7/8), ':^', label='(pi/(2M+2))**(7/8)')
# plt.loglog(nValues, (np.diag(coll.QDelta)**2).sum()/minDelta, label='sum(**2)/minDelta')
plt.legend()
plt.grid()
