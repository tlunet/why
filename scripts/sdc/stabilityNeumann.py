#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:49:22 2022

Compute the stability contours of SDC for
"""
import numpy as np
import matplotlib.pyplot as plt

from pycode.dahlquist import IMEXSDC

# -----------------------------------------------------------------------------
# Change these ...
# -- collocation settings
M = 4
nodeDistr = 'LEGENDRE'
quadType = 'RADAU-RIGHT'
# -- SDC settings
varSweeps = [f'DNODES-{i+1}' for i in range(M)]
sweepType = ['THETAPAR-0.5']
# sweepType = varSweeps
initSweep = 'QDELTA'
collUpdate = False
# -- plot settings
zoom = 10  # the larger, the far away ...
# -----------------------------------------------------------------------------

u0 = 1.0
lamReals = -5*zoom, 1*zoom, 256
lamImags = -4*zoom, 4*zoom, 256

xLam = np.linspace(*lamReals)[:, None]
yLam = np.linspace(*lamImags)[None, :]

lams = xLam + 1j*yLam

plt.figure()

IMEXSDC.setParameters(
    M=M, quadType=quadType, nodeDistr=nodeDistr,
    implSweep=sweepType, explSweep='PIC', initSweep=initSweep,
    forceProl=collUpdate)

def plotStabContour(nSweep):
    IMEXSDC.nSweep = nSweep

    solver = IMEXSDC(u0, lams.ravel(), 0)
    solver.step(1.)

    uNum = solver.u.reshape(lams.shape)

    stab = np.abs(uNum)
    coords = np.meshgrid(xLam.ravel(), yLam.ravel(), indexing='ij')

    CS = plt.contour(*coords, stab, levels=[1.0], colors='k')
    plt.clabel(CS, inline=True, fmt=f'K={nSweep}')
    plt.grid(True)
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel(r'$Re(\lambda)$')
    plt.ylabel(r'$Im(\lambda)$')
    plt.gcf().set_size_inches(4, 5)
    plt.title(IMEXSDC.implSweep)
    plt.tight_layout()

    return stab

for nSweep in [1, 2, 3]:
    stab = plotStabContour(nSweep)
