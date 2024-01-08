#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:58:48 2024

@author: cpf5546
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from pycode.qmatrix import genCollocation

quadType = 'RADAU-RIGHT'
distr = 'LEGENDRE'
M = 4

nodes, _, Q = genCollocation(M, distr, quadType)
tau = np.linspace(0, 1, num=500)

def fit(d):
    def func(ab):
        a, b = ab
        return np.linalg.norm(a*nodes**b - d)
    sol = sp.optimize.minimize(func, [1,1], method="nelder-mead")
    return sol.x

def nilpotency(d, Q):
    if quadType in ['LOBATTO', 'RADAU-LEFT']:
        d = d[1:]
        Q = Q[1:, 1:]
    M = d.size
    D = np.diag(1/d)
    K = np.eye(M)- D @ Q

    nil = np.linalg.norm(np.linalg.matrix_power(K, M), ord=np.inf)
    sr = np.max(np.abs(np.linalg.eigvals(K)))
    return nil, sr


def computeCoefficients(d0=None):
    nCoeffs = M
    _nodes = nodes
    _Q = Q
    if quadType in ['LOBATTO', 'RADAU-LEFT']:
        nCoeffs -= 1;
        _Q = Q[1:, 1:]
        _nodes = nodes[1:]

    def func(coeffs):
        coeffs = np.asarray(coeffs)
        kMats = [(1-z)*np.eye(nCoeffs) + z*np.diag(1/coeffs) @ _Q
                  for z in _nodes]
        vals = [np.linalg.det(K)-1 for K in kMats]
        return np.array(vals)

    if d0 is None:
        d0 = _nodes/M

    coeffs = sp.optimize.fsolve(func, d0, xtol=1e-15)

    if quadType in ['LOBATTO', 'RADAU-LEFT']:
        coeffs = np.array([0] + list(coeffs))

    return coeffs

plt.figure()

d = []

d.append(computeCoefficients())
print('d[0] :', nilpotency(d[0], Q))
plt.plot(nodes, d[0]*M, 'o-', label="d[0]")

a, b = fit(d[0]*M)
plt.plot(tau, a*tau**b, ':', c="gray")

nSubs = 2*M
for i in range(1, nSubs):
    tau0 = a*((tau - i/nSubs) % 1)**b
    plt.plot(tau, tau0, ':', c="gray")

    d0 = a*((nodes - i/nSubs) % 1)**b
    d.append(computeCoefficients(d0=d0/M))
    nil = nilpotency(d[i], Q)
    print(f'd[{i}] :', nil)
    if nil[0] < 1e-10:
        plt.plot(nodes, d[i]*M, 'o-', label=f'd[{i}]')

plt.legend()
plt.grid()
plt.tight_layout()
