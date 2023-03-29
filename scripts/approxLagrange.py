#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:02:42 2021

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import lagrange

from pycode.qmatrix import genQMatrices

n = 20
nP = int(n//2)+1

sweepType = 'BE'
quadType = 'LOBATTO'

equid = genQMatrices(n, 'EQUID', quadType, sweepType)
legen = genQMatrices(n, 'LEGENDRE', quadType, sweepType)

def lagrangeBE(nodes, i, x):
    if sweepType == 'TRAP':
        print('Warning : trapezoidal Lagrange interpolation not implemented')
    if sweepType == 'FE':
        i += 1
    return (x >= (nodes[i-1] if i > 0 else 0)) * \
        (x <= (nodes[i] if i < n else 1))*1

def indicator(points, i):
    out = np.zeros_like(points)
    out[i] = 1
    return out

x = np.linspace(0, 1, num=500)

fig, axes = plt.subplots(2, 2)

def plotLagrange(ax, nodes):
    ax.plot(nodes, 0*nodes+1, 'o', c='k')
    for i in range(nP):
        ax.plot(x, lagrange(nodes, indicator(nodes, i))(x))
        c = ax.lines[-1].get_color()
        ax.plot(x, lagrangeBE(nodes, i, x), c=c, ls='--')
    ax.grid()

plotLagrange(axes[0, 0], equid['nodes'])
plotLagrange(axes[1, 0], legen['nodes'])
axes[0, 0].set_title(r'$l_j$ and $\tilde{l}_j$ polynomials' +
                     f' ({sweepType}, {quadType})')
axes[0, 0].set_ylim(-1, 2)

def plotDiffQMat(ax, Q, QDelta):
    im = ax.imshow(np.abs(Q - QDelta))
    fig.colorbar(im, ax=ax)

plotDiffQMat(axes[0, 1], equid['Q'], equid['QDelta'])
plotDiffQMat(axes[1, 1], legen['Q'], legen['QDelta'])
axes[0, 1].set_title(r'$|q_{i,j}-q^{\Delta}_{i, j}|$')

fig.set_size_inches(18.6, 9.72)
plt.tight_layout()
