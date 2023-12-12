#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:19:08 2022

@author: cpf5546
"""
import numpy as np
import matplotlib.pyplot as plt

from pycode.dahlquist import IMEXSDC

# -----------------------------------------------------------------------------
# Change these ...
# -- collocation settings
M = 4
nodeDistr = 'EQUID'
quadType = 'RADAU-RIGHT'
# -- SDC settings
implSweep = ['THETAPAR-0.5']
explSweep = 'PIC'
initSweep = 'QDELTA'
collUpdate = False
# -----------------------------------------------------------------------------

u0 = 1.0
lamSlows = 0, 50, 500
lamFasts = 0, 120, 501

lamSlow, lamFast = np.meshgrid(
    1j*np.linspace(*lamSlows), 1j*np.linspace(*lamFasts), indexing='ij')


def plotStabContour(ax, nSweep, implSweep, explSweep,
                    forceProl, qType, M, initSweep):

    IMEXSDC.setParameters(
        M=M, quadType=qType, nodeDistr='LEGENDRE',
        implSweep=implSweep, explSweep=explSweep, initSweep=initSweep,
        forceProl=forceProl)
    IMEXSDC.nSweep = nSweep

    solver = IMEXSDC(u0, lamFast.ravel(), lamSlow.ravel())
    solver.step(1.)

    uNum = solver.u.reshape(lamSlow.shape)

    stab = np.abs(uNum)

    CS1 = ax.contour(lamSlow.imag, lamFast.imag, stab,
                      levels=[0.95, 1.05], colors='gray', linestyles='--')
    CS2 = ax.contour(lamSlow.imag, lamFast.imag, stab,
                      levels=[1.0], colors='black')
    plt.clabel(CS1, inline=True, fmt='%3.2f')
    plt.clabel(CS2, inline=True, fmt='%3.2f')
    ax.plot(lamSlows[:-1], lamSlows[:-1], ':', c='gray')

    ax.set_xlabel(r'$\Delta t \lambda_{slow}$')
    ax.set_ylabel(r'$\Delta t \lambda_{fast}$')
    ax.set_title(f'K={nSweep}')
    plt.gcf().suptitle(f'Q={qType}-{M}, QI={implSweep}, QE={explSweep}, '
                       f'forceProl={forceProl}, initSweep={initSweep}', y=0.95)


lSweeps = [1, 2, 3]
fig, axs = plt.subplots(1, len(lSweeps))
fig.set_size_inches(len(lSweeps)*4, 4)
for i, nSweep in enumerate(lSweeps):
    plotStabContour(axs[i], nSweep=nSweep,
                    implSweep=implSweep, explSweep=explSweep, M=M,
                    qType=quadType, forceProl=collUpdate, initSweep=initSweep)
fig.tight_layout()
plt.savefig('stabContourFWSW.pdf')
