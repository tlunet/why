#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:30:23 2022

@author: cpf5546
"""
import numpy as np
import matplotlib.pyplot as plt

from pycode.dahlquist import IMEXSDC

# -----------------------------------------------------------------------------
# Change these ...
# -- collocation settings
M = 2
nodeDistr = 'LEGENDRE'
quadType = 'RADAU-RIGHT'
# -- SDC settings
implSweep = ["MIN-SR-NS"]
explSweep = 'PIC-1'
# implSweep = 'BE'
# explSweep = 'FE'
initSweep = 'QDELTA'
collUpdate = True
# -- Dahlquist settings
u0 = 1.0
lambdaI = 1j
lambdaE = 0
lambdaI = 1.0j
lambdaE = 0.0
# -----------------------------------------------------------------------------

listNumStep = [2**(i+1) for i in range(8)]

IMEXSDC.setParameters(
    M=M, quadType=quadType, nodeDistr=nodeDistr,
    implSweep=implSweep, explSweep=explSweep, initSweep=initSweep,
    forceProl=collUpdate, lambdaI=lambdaI, lambdaE=lambdaE)

solver = IMEXSDC(u0, lambdaI, lambdaE)

plt.figure()
symList = ['o', '^', 's', '<']
for i, nSweep in enumerate([0, 1, 2, 3]):

    IMEXSDC.nSweep = nSweep

    def getErr(nStep):
        dt = 2*np.pi/nStep
        times = np.linspace(0, 2*np.pi, nStep+1)
        uTh = u0*np.exp(times*(lambdaE+lambdaI))

        np.copyto(solver.u, u0)
        uNum = [solver.u.copy()]
        for i in range(nStep):
            solver.step(dt)
            uNum += [solver.u.copy()]
        uNum = np.array(uNum)
        #err = np.linalg.norm(uNum-uTh, ord=np.inf)
        err = np.abs(uNum-uTh)[-1]
        return dt, err

    # Run all simulations
    dt, err = np.array([getErr(n) for n in listNumStep]).T
    print(err)

    # Plot error VS time step
    lbl = f'SDC, nSweep={IMEXSDC.nSweep}'
    sym = symList[i]+'-'
    plt.loglog(dt, err, sym, label=lbl)

    # Plot order curve
    order = nSweep+1
    c = err[0]/dt[0]**order * 2
    plt.plot(dt, c*dt**order, '--', color='gray')

plt.xlabel(r'$\Delta{t}$')
plt.ylabel(r'error ($L_{inf}$)')
plt.ylim(1e-10, 1e1)
plt.legend()
plt.grid(True)
plt.title(f'{IMEXSDC.implSweep}, {M}-{quadType}-{nodeDistr}',
          fontdict=dict(fontsize=10))
plt.tight_layout()
plt.show()
