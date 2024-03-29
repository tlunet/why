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
M = 3

test = ['IMEX-1', 'IMEX-2', 'IMEX-3', 'IMEX-4', 'IMEX-5', 'IMEX-6', 'IMEX-7', 'IMEX1-8']
dnodes = ['DNODES-1', 'DNODES-2', 'DNODES-3', 'DNODES-4', 'DNODES-5', 'DNODES-6', 'DNODES-7', 'DNODES-8']

nodeDistr = 'LEGENDRE'
quadType = 'RADAU-RIGHT'
# -- SDC settings
listImplSweeps = [
    ('LU', '-^'),
    #('MIN3', '-s'),
    ('TRAPAR', '--*'),
    (['MIN-SR-NS'], '-p'),
    (['MIN-SR-S'], '-p'),
    (['IMEX-NS', '--X']),
    (test[1:M], '--X'),
    (dnodes[1:M], '-<')
]
explSweep = 'PIC'
initSweep = 'COPY'
collUpdate = False
# -- Dahlquist settings
u0 = 1.0
lambdaI = 0.6j
lambdaE = 0.4j
lam = lambdaI + lambdaE
tEnd = np.pi
nSteps = 1
# -----------------------------------------------------------------------------


def extractResiduals(solver, dt):
    lamU = (np.array(solver.lamIU[0]) + np.array(solver.lamEU[0])).ravel()
    lam = solver.lambdaI + solver.lambdaE
    return solver.u0 - lamU/lam + dt * solver.Q @ lamU


def extractError(solver, dt):
    return solver.u - np.exp(lam*dt)*u0


plt.figure(figsize=(10, 8))
min_err = np.inf
for (implSweep, symbol) in listImplSweeps:

    IMEXSDC.setParameters(
        M=M, quadType=quadType, nodeDistr=nodeDistr,
        implSweep=implSweep, explSweep=explSweep, initSweep=initSweep,
        forceProl=collUpdate, lambdaI=lambdaI, lambdaE=lambdaE)

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
    min_err = min(min_err, min(error))

    # Plot residuals VS sweeps
    sym = '^-' if symbol == '' else symbol
    plt.semilogy(residuals, sym, label=str(implSweep))
    # plt.semilogy(error, sym, label=str(implSweep))


plt.semilogy(list(range(12)), min_err * np.ones(12), 'k--')

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
plt.show()
