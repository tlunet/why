#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:51:19 2023

@author: cpf5546
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
implSweep = ['TRAPAR']
explSweep = 'PIC'
initSweep = 'QDELTA'
collUpdate = False
# -- Dahlquist settings
u0 = 1.0
lambdaI = 1j
lambdaE = 0
lam = lambdaE+lambdaI
tEnd = 2*np.pi
nSteps = 10
# -----------------------------------------------------------------------------

times = np.linspace(0, tEnd, nSteps+1)

IMEXSDC.setParameters(
    M=M, quadType=quadType, nodeDistr=nodeDistr,
    implSweep=implSweep, explSweep=explSweep, initSweep=initSweep,
    forceProl=collUpdate)

solver = IMEXSDC(u0, lambdaI, lambdaE)

def extractResiduals(solver, dt):
    lamU = (np.array(solver.lamIU[0]) + np.array(solver.lamEU[0])).ravel()
    lam = solver.lambdaI + solver.lambdaE
    return solver.u0 - lamU/lam + dt * solver.Q @ lamU

plt.figure()
for nSweep in [0, 1, 2, 3]:

    IMEXSDC.nSweep = nSweep

    dt = tEnd/nSteps
    times = np.linspace(0, tEnd, nSteps+1)

    np.copyto(solver.u, u0)
    residuals = [extractResiduals(solver, dt)]
    for i in range(nSteps):
        solver.step(dt)
        residuals += [extractResiduals(solver, dt)]
    residuals = np.array(residuals)
    residuals = np.linalg.norm(residuals, ord=np.inf, axis=-1)

    # Plot residuals VS time
    lbl = f'k={IMEXSDC.nSweep}'
    sym = '^-'
    plt.semilogy(times, residuals, sym, label=lbl)

plt.xlabel(r'Time')
plt.ylabel(r'Maximum residuals')
plt.ylim(1e-6, 1)
textArgs = dict(
    fontsize=13,
    bbox=dict(boxstyle="round",
              ec=(0.5, 0.5, 0.5),
              fc=(0.8, 0.8, 0.8)))
plt.text(1, 1e-5, 
         r'$\lambda='f'{lam}$, '
         r'$\Delta T='f'{dt:.2f}$, '
         r'$N_{steps}='f'{nSteps}$', **textArgs)
plt.legend(loc="lower right")
plt.grid(True)
plt.title(IMEXSDC.implSweep)
plt.tight_layout()
