#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 21:39:39 2023

@author: gaya
"""
import numpy as np

from pycode.qmatrix import genCollocation

# change these:
############################
M = 4
quadType = 'LOBATTO'
distr = 'LEGENDRE'
############################


def power_iteration(A, num_iterations: int):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_iterations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k


def spectralRadiusPower(x):
    """Warning : x are the diagonal coefficients of QDelta !"""
    D = np.diag(x)
    S = Q-D
    bK = power_iteration(S, 50000)
    return bK.dot(S.dot(bK))


def spectralRadius(x):
    x = np.asarray(x)
    R = Q - np.diag(x)
    return np.max(np.abs(np.linalg.eigvals(R)))

nodes, _, Q = genCollocation(M, distr, quadType)

print('quadType = {}\ndistr = {}\nM = {}'.format(quadType, distr, M))
print('define: D = diag(nodes) / M')
D = np.diag(nodes) / M

print('define: S = Q-D')
S = (Q- D)
print(S)
powS = np.eye(M)
for m in range(M+1):
    print('|(Q - D)^{}|_max = {}'.format(m, np.max(np.abs(powS))))
    powS = powS @ S

print('eigVals(S):')
print(np.linalg.eigvals(S))
print('cond(S):')
print(np.linalg.cond(S))
