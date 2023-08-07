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
    ('TRAPAR', '-*'),
    # (['BEPAR', 'MIN-SR-S'], '--s'),
    # (['BEPAR', 'MIN-SR-NS'], '--p'),
]
st1 = 'TRAPAR'
st2 = 'MIN-SR-NS'
listImplSweeps = [
    ('LU', '-^'),
    ('TRAP', '-o'),
    ('MIN3', '-s'),
    ('TRAPAR', '--*'),
    (['TRAPAR', 'TRAPAR', 'MIN-SR-NS'], '-p'),
    (['TRAPAR', 'TRAPAR', 'MIN-SR-S'], '-<'),
    ('MIN-SR-NS', '--p'),
    ('MIN-SR-S', '--<'),
    # ([st1]*2 + [st2], '-o'),
    # ([st2]*3 + [st1], '-p'),
    # (st2, '-s'),
    # (['BEPAR', 'MIN-SR-S'], '--s'),
    # (['BEPAR', 'MIN-SR-NS'], '--p'),
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
lam = lambdaI + lambdaE
tEnd = np.pi/10
nSteps = 1
# -----------------------------------------------------------------------------


def extractResiduals(solver, dt):
    lamU = (np.array(solver.lamIU[0]) + np.array(solver.lamEU[0])).ravel()
    lam = solver.lambdaI + solver.lambdaE
    return solver.u0 - lamU/lam + dt * solver.Q @ lamU


def extractError(solver, dt):
    return solver.u - np.exp(lam*dt)*u0


plt.figure()

for (implSweep, symbol) in listImplSweeps:

    IMEXSDC.setParameters(
        M=M, quadType=quadType, nodeDistr=nodeDistr,
        implSweep=implSweep, explSweep=explSweep, initSweep=initSweep,
        forceProl=collUpdate)

    solver = IMEXSDC(u0, lambdaI, lambdaE)

    residuals = []
    error = []
    for nSweep in range(12):

        IMEXSDC.nSweep = nSweep

        dt = tEnd/nSteps

        np.copyto(solver.u, u0)
        for i in range(nSteps):
            solver.step(dt)
        residuals += [extractResiduals(solver, dt)]
        error += [extractError(solver, dt)]

    residuals = np.array(residuals)
    residuals = np.linalg.norm(residuals, ord=np.inf, axis=-1)

    error = np.abs(error)

    # Plot residuals VS sweeps
    sym = '^-' if symbol == '' else symbol
    plt.semilogy(residuals, sym, label=str(implSweep))
    # plt.semilogy(error, sym, label=str(implSweep))


plt.xlabel(r'Sweeps')
plt.ylabel(r'Maximum residuals')
plt.ylim(ymax=10)
textArgs = dict(
    fontsize=13,
    bbox=dict(boxstyle="round",
              ec=(0.5, 0.5, 0.5),
              fc=(0.8, 0.8, 0.8)))
plt.text(2.5, 1,
         r'$\lambda='f'{lam}$, '
         r'$\Delta T='f'{dt:.2f}$, '
         r'$N_{steps}='f'{nSteps}$', **textArgs)
plt.legend(loc="lower left")
plt.grid(True)
plt.title(f'M={M}, {nodeDistr}, {quadType}, {initSweep}')
plt.tight_layout()
