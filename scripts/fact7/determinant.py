#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 18:26:32 2023

@author: cpf5546
"""
import numpy as np
import scipy as sp
import sympy as sy
from pycode.qmatrix import genCollocation, genQDelta

M = 4
quadType = 'RADAU-RIGHT'
distr = 'LEGENDRE'

nodes, _, QFull = genCollocation(M, distr, quadType)

Q = QFull
if quadType in ['LOBATTO', 'RADAU-LEFT']:
    Q = Q[1:, 1:]
    nodes = nodes[1:]

qDetTh = np.prod(nodes)/sp.special.factorial(M)

qDetNum = np.linalg.det(Q)

print(f'M={M}, {quadType}-{distr}')
print(f' -- detTh  : {qDetTh}')
print(f' -- detNum : {qDetNum}')

QDelta, dtau = genQDelta(nodes, "MIN-SR-S", Q)

Ks = (np.eye(M) - np.linalg.solve(QDelta, Q))

print(f"found with determinant approach : {np.diag(QDelta)}")
print("  nilpotency norm :", np.linalg.norm(np.linalg.matrix_power(Ks, M), np.inf))


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


# d = 1/np.diag(QDelta)
# # d = 1/nodes
# for m in range(M):
#     print(g(d, m))


def f(d):
    return g(d, M-1)


sol = sp.optimize.root(f, 1/np.diag(QDelta), tol=1e-14)
QDeltaF = np.diag(1/sol.x)
print(f"refined with Lagrange approach : {np.diag(QDeltaF)}")
Ks = (np.eye(M) - np.linalg.solve(QDeltaF, Q))
print("  nilpotency norm :", np.linalg.norm(np.linalg.matrix_power(Ks, M), np.inf))


# Qsym = sy.Matrix([[sy.Symbol("q_{"+str(i)+","+str(j)+"}")
#                    for j in range(M)]
#                   for i in range(M)])

# Dsym = sy.eye(M)
# for i in range(M):
#     Dsym[i,i] = sy.Symbol("d_{"+str(i)+"}")

# KSsym = np.eye(M, dtype=int) - Qsym @ Dsym

# Psym = np.eye(M, dtype=int)
# for i in range(M):
#     Psym = Psym @ KSsym

# e = sy.zeros(M, 1)
# e[0] = 1
