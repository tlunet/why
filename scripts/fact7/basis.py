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
    D = np.diag(1/d)
    K = np.eye(M)- D @ Q
    return np.linalg.norm(
        np.linalg.matrix_power(K, M), ord=np.inf)


incremental = False
a = None
b = None
idx = "ijklmnopqrstuvwxyzabcdefgh"
nils = []

qDeltas = {}

plt.figure()
mVals = range(2, 10)
for M in mVals:

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

        V = np.vander(_nodes, nCoeffs)
        fac = (1/(np.arange(nCoeffs)+1))[-1::-1]
        W = _nodes[:, None] * np.vander(_nodes, nCoeffs) * fac

        def newton(j, x):
            return np.prod([x-t for t in _nodes[:j]])

        N = np.array([[newton(j, _nodes[i]) for j in range(i+1)]+(nCoeffs-i-1)*[0]
                      for i in range(nCoeffs)])

        def func2(coeffs):
            coeffs = np.asarray(coeffs)[:, None]
            I = np.eye(nCoeffs)
            kMats = [(1-z)*I + z*np.linalg.solve(N, 1/coeffs*Q @ N)
                      for z in _nodes]
            vals = [np.linalg.det(K) - 1 for K in kMats]
            return np.array(vals)

        idx = "ijklmnopqrstuvwxyzabcdefgh"
        def g(d, m):

            def T(i,m,k):
                if k == 0:
                    return 1
                elif k == 1:
                    return Q[i,m]
                else:
                    sSum = ','.join(idx[:k-1])
                    arrays = [d]*(k-1)

                    sSum += ','+idx[0]
                    arrays += [Q[i,:]]
                    for j in range(k-2):
                        sSum += ','+idx[j]+idx[j+1]
                        arrays += [Q]
                    sSum += ','+idx[k-2]
                    arrays += [Q[:,m]]

                    return np.einsum(sSum, *arrays)

            coeffs = np.array([
                (d[i] if i == m else 1) * sum(
                    sp.special.binom(M, k+1) * (-1)**(k+1) * T(i, m, k+1)
                    for k in range(M))
                for i in range(M)
                ])
            coeffs[m] += 1

            return coeffs

        def func3(coeffs):
            return g(1/coeffs, 0)

        if a is None:
            d0 = _nodes/M
        else:
            d0 = a*_nodes**b/M

        sol = sp.optimize.root(func, d0, tol=1e-15, method="hybr")
        coeffs = sol.x

        if quadType in ['LOBATTO', 'RADAU-LEFT']:
            coeffs = [0] + list(coeffs)

        qDelta = np.diag(coeffs)

    qDeltas[M] = qDelta

    di = np.diag(qDelta)

    di = np.array(di)
    plt.plot(nodes, di*M, 'o-')

    a, b = fit(di*M, nodes)
    print("---- M =", M, "---")
    print(a, b)
    nil = nilpotency(di, Q)
    print(nil)
    nils.append(nil)

print("----")
plt.plot(nodes, a*nodes**b, '--')
plt.grid()

plt.figure('nilpotency')
plt.semilogy(mVals, nils, 'o-', label=f'incremental={incremental}')
plt.grid(True)
plt.legend()

M = 4
nodes, _, Q = genCollocation(M, distr, quadType)
qDelta = qDeltas[M]

d = np.diag(qDelta)
# d = nodes/M

if quadType in ['LOBATTO', 'RADAU-LEFT']:
    M -= 1
    nodes = nodes[1:]
    d = d[1:]
    Q = Q[1:, 1:]
D = np.diag(1/d)

V = np.vander(nodes, M)
fac = (1/(np.arange(M)+1))[-1::-1]
W = nodes[:, None] * np.vander(nodes, M) * fac


def newton(j, x):
    return np.prod([x-t for t in nodes[:j]])

N = np.array([[newton(j, nodes[i]) for j in range(i+1)]+(M-i-1)*[0]
              for i in range(M)])

S = V - D @ W
print("Rank(S) =", np.linalg.matrix_rank(S))

print(np.linalg.cond(np.linalg.solve(qDelta, Q)))
print(np.linalg.cond(np.linalg.solve(V, D @ Q @ V)))
print(np.linalg.cond(np.linalg.solve(N, D @ Q @ N)))

for _ in range(M):
    c = np.linalg.solve(S[1:,1:], -S[1:,0])
    rest = S[0,0] + S[0, 1:].dot(c)
    print(c, rest)
    S = np.roll(S, 1, axis=0)

p = np.array([1] + c.tolist())
print('p =', p)
basis = [p]
for m in range(M-1):
    basis += [np.linalg.solve(S, np.polyval(basis[-1], nodes))]
    print(' --', basis[-1])
