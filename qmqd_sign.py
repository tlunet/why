#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:14:16 2022

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from qmatrix import genQMatrices

# QDelta
sweepType = 'BE'

# Node distribution, can be
# -- 'EQUID', 'LEGENDRE', 'CHEBY-1', 'CHEBY-2', 'CHEBY-3', 'CHEBY-4'
nodeType = 'LEGENDRE'

# Number of nodes
M = 30

fig, axes = plt.subplots(1, 4)
fig.suptitle(f'Sign of QmQDelta for {nodeType}-{sweepType}'
             ' : yellow=positive, blue=negative, cyan=null')

for i, quadType in enumerate(['GAUSS', 'RADAU-I', 'RADAU-II', 'LOBATTO']):

    coll = genQMatrices(M, nodeType, quadType, sweepType, scaling=True)

    # Rounding to 10 decimals to avoid detecting sign for almost zero values
    QmQDelta = np.round(coll.Q - coll.QDelta, 10)

    axes[i].imshow(np.sign(QmQDelta))
    axes[i].set_title(quadType)

fig.set_size_inches(16.6, 4.7)
plt.tight_layout()
