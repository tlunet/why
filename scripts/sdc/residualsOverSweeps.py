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
listImplSweeps = [
    ('BE', '-^'),
    ('LU', '-o'),
    ('BEPAR', '-s'),
    ('MIN-SR-S', '--^'),
    ('MIN-SR-NS', '--o'),
    (['BEPAR', 'MIN-SR-S'], '--s'),
    (['BEPAR', 'MIN-SR-NS'], '--p'),
    ]
# varSweeps = ['BEPAR']+[f'DNODES-{i+1}' for i in range(M)]
# listImplSweeps = [
#     (varSweeps, '--^'),
#     (varSweeps[-1::-1], '--o'),
#     (['BEPAR', 'DNODES'], '->'),
#     ('LU', '-s'),
#     ('DNODES', '-p'),
#     ('OPT-SPECK-0', '-*'),
#     ('BEPAR', '->'),
#     ]
explSweep = 'PIC'
initSweep = 'QDELTA'
collUpdate = False
# -- Dahlquist settings
u0 = 1.0
lambdaI = 1j
lambdaE = 0
tEnd = 2*np.pi
nSteps = 10
# -----------------------------------------------------------------------------

def extractResiduals(solver, dt):
    lamU = (np.array(solver.lamIU[0]) + np.array(solver.lamEU[0])).ravel()
    lam = solver.lambdaI + solver.lambdaE
    return solver.u0 - lamU/lam + dt * solver.Q @ lamU

plt.figure()

for (implSweep, symbol) in listImplSweeps:

    IMEXSDC.setParameters(
        M=M, quadType=quadType, nodeDistr=nodeDistr,
        implSweep=implSweep, explSweep=explSweep, initSweep=initSweep,
        forceProl=collUpdate)

    solver = IMEXSDC(u0, lambdaI, lambdaE)

    residuals = []
    for nSweep in range(6):

        IMEXSDC.nSweep = nSweep

        dt = tEnd/nSteps

        np.copyto(solver.u, u0)
        for i in range(nSteps):
            solver.step(dt)
        residuals += [extractResiduals(solver, dt)]

    residuals = np.array(residuals)
    residuals = np.linalg.norm(residuals, ord=np.inf, axis=-1)

    # Plot residuals VS sweeps
    sym = '^-' if symbol == '' else symbol
    plt.semilogy(residuals, sym, label=str(implSweep))

plt.xlabel(r'Sweeps')
plt.ylabel(r'Maximum residuals')
# plt.ylim(1e-13, 1)
plt.legend()
plt.grid(True)
plt.title(f'M={M}, {nodeDistr}, {quadType}, {initSweep}')
plt.tight_layout()
