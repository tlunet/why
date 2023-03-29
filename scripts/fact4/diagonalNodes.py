#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 21:39:39 2023

@author: telu
"""
import numpy as np

from pycode.qmatrix import genCollocation

# change these:
############################
M = 4
quadType = 'LOBATTO'
distr = 'EQUID'
############################

def spectralRadius(x):
    x = np.asarray(x)
    R = Q - np.diag(x)
    return np.max(np.abs(np.linalg.eigvals(R)))

nodes, _, Q = genCollocation(M, distr, quadType)

print('Coefficients with nodes/M :')
print(nodes/M)
# assumption: (Q - diag(t))(1, 1, ..., 1) = 0 on [0, 1]

if quadType == 'RADAU-RIGHT':
    print('sum(di) =', M / (2 * M - 1))

elif quadType == 'RADAU-LEFT':
    print('sum(di) =', (M - 1) / (2 * M - 1))

elif quadType == 'GAUSS':
    print('sum(di) =', 0.5)

elif quadType == 'LOBATTO':
    print('sum(di) =', 0.5)

print('\ndefine D = diag(nodes) / m\nsum(nodes) / m =', sum(nodes) / M)
D = np.diag(nodes) / M
print('|| (Q - m * D) e || = ', np.linalg.norm((Q - M * D) @ np.ones(M)))

eig = max(np.abs(np.linalg.eigvals(Q - D)))  # is this the one?
print('ro(Q - D) = ', eig)

print('Flip nodes :')
D = np.diag(nodes[-1::-1]) / M
eig = max(np.abs(np.linalg.eigvals(Q - D)))
print('ro(Q - D) = ', eig)



