#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:54:51 2021

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

from qmatrix import genQMatrices


lM = np.arange(50)+3

distr = 'LEGENDRE'
quadType = 'LOBATTO'
implementation = 'SPECTRAL'
NORM_TYPE = 'L_INF'


def norm(A):
    order = 2 if NORM_TYPE == 'L_2' else np.inf
    return np.linalg.norm(A, order)


for sweepType, sym in zip(['FE', 'BE'],
                          ['s', 'o', '^', 'p']):
    normQmQDelta = [
        norm(Q-QDelta) for Q, QDelta, _, _ in
        [genQMatrices(M, distr, quadType, sweepType, implementation)
         for M in lM]]

    s = slice(None, None)

    def obj(x):
        a, b = x
        return np.linalg.norm(normQmQDelta[s] - (a/lM**(b))[s])

    plt.figure(f'QmQDelta_{distr}_{quadType}_{sweepType}_{NORM_TYPE}')
    plt.loglog(lM, normQmQDelta, '-o', label='Numeric.')

    iSep = 9

    s = slice(None, iSep)
    res = sco.minimize(obj, [1., 1.])
    a1, b1 = res.x
    sa1, sb1 = f'{a1:1.2f}', f'{b1:1.2f}'
    plt.loglog(lM, a1/lM**(b1), '--', label='$'+sa1+'/M^{'+sb1+'}$')

    s = slice(iSep, None)
    res = sco.minimize(obj, [1., 1.])
    a2, b2 = res.x
    sa2, sb2 = f'{a2:1.2f}', f'{b2:1.2f}'
    plt.loglog(lM, a2/lM**(b2), '--', label='$'+sa2+'/M^{'+sb2+'}$')

    plt.grid(True)
    plt.xlabel('$M$')
    plt.ylabel(r'$||Q-Q_\Delta||$')
    plt.legend()
    plt.tight_layout()

    plt.figure(f'QmQDelta_comp_{NORM_TYPE}')
    plt.loglog(lM, normQmQDelta, '-'+sym, label=f'{sweepType}')
    plt.legend()
    plt.xlabel('$M$')
    plt.ylabel(r'$||Q-Q_\Delta||$')
    plt.grid(True)
    plt.tight_layout()

plt.show()
