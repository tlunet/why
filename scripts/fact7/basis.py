#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:12:29 2023

@author: cpf5546
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from pycode.qmatrix import genCollocation, genQDelta

quadType = 'RADAU-RIGHT'
distr = 'LEGENDRE'


def fit(d, tau):

    def func(ab):
        a, b = ab
        return np.linalg.norm(a*tau**b - d)

    sol = sp.optimize.minimize(func, [1,1], method="nelder-mead")

    return sol.x


def nilpotency(d, Q):
    if quadType in ['LOBATTO', 'RADAU-LEFT']:
        d = d[1:]
        Q = Q[1:, 1:]
    M = d.size
    D = np.diag(d)
    K = (np.eye(M)-np.linalg.solve(D, Q))
    return np.linalg.norm(
        np.linalg.matrix_power(K, M), ord=np.inf)


incremental = False
a = None
b = None
idx = "ijklmnopqrstuvwxyzabcdefgh"
nils = []
plt.figure()
for M in range(2, 20):

    nodes, _, Q = genCollocation(M, distr, quadType)

    if not incremental:
        qDelta, _ = genQDelta(nodes, 'MIN-SR-S', Q, None, None)

    else:
        # Compute coefficients using the determinant (numerical approach)
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

        if a is None:
            d0 = _nodes/M
        else:
            d0 = a*_nodes**b/M

        coeffs = sp.optimize.fsolve(func, d0, xtol=1e-15)

        if quadType in ['LOBATTO', 'RADAU-LEFT']:
            coeffs = [0] + list(coeffs)

        qDelta = np.diag(coeffs)

    di = np.diag(qDelta)

    di = np.array(di)
    plt.plot(nodes, di*M, 'o-')

    a, b = fit(di*M, nodes)
    print("----", M, "---")
    print(a, b)
    nil = nilpotency(di, Q)
    print(nil)
    nils.append(nil)

print("----")
plt.plot(nodes, a*nodes**b, '--')
plt.grid()

plt.figure('nilpotency')
plt.semilogy(nils, label=f'incremental={incremental}')
plt.grid(True)
plt.legend()

M = 4
nodes, _, Q = genCollocation(M, distr, quadType)
qDelta, _ = genQDelta(nodes, 'MIN-SR-S', Q, None, None)
d = np.diag(qDelta)

if quadType in ['LOBATTO', 'RADAU-LEFT']:
    M -= 1
    nodes = nodes[1:]
    d = d[1:]
    Q = Q[1:, 1:]
D = np.diag(1/d)

V = np.vander(nodes, M)
fac = (1/(np.arange(M)+1))[-1::-1]
W = nodes[:, None] * np.vander(nodes, M) * fac

S = V - D @ W

c = np.linalg.solve(S[1:,1:], -S[1:,0])
rest = S[0,0] + S[0, 1:].dot(c)
print(c, rest)
print(nodes[1]-1/2)
