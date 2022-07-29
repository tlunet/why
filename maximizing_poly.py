#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:27:47 2022

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from qmatrix import genPolyApprox


M = 50
num = 500

# Build the maximizing polynomial
p = np.ones(M)
p[M//2:] = -1

# Node distribution, can be
# -- 'EQUID', 'LEGENDRE', 'CHEBY-1', 'CHEBY-2', 'CHEBY-3', 'CHEBY-4'
nodeType = 'LEGENDRE'

# Quadrature type (wether or not include the right/left bound in the nodes)
# -- 'GAUSS', 'RADAU-I' (left), 'RADAU-II' (right), 'LOBATTO'
quadType = 'LOBATTO'

# Generate nodes and polynomial approximation
ap = genPolyApprox(
    M, distr=nodeType, quadType=quadType,
    implementation='LAGRANGE', scaling=True)

# Plot discretization
t = np.linspace(0, 1, num)

# Interpolation matrix
P = ap.getInterpolationMatrix(t)

# Plotting
plt.figure()
plt.plot(t, P.dot(p), label=f'{quadType}-{nodeType}')
plt.plot(ap.points, p, 'o')
plt.grid()
plt.title(f'Interpolant of maximizing poly. for M={M}')
plt.xlabel('Time')
plt.legend()
plt.gcf().set_size_inches(14, 5)
plt.tight_layout()
plt.show()
