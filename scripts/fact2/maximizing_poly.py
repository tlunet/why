#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:27:47 2022

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from pycode.qmatrix import genCollocation
from pycode.lagrange import LagrangeApproximation


M = 5

# Plot discretization
num = 500
t = np.linspace(0, 1, num)

# Build the maximizing polynomial values
p = np.ones(M)
p[M//2:] = -1

# Node distribution, can be
# -- 'EQUID', 'LEGENDRE', 'CHEBY-1', 'CHEBY-2', 'CHEBY-3', 'CHEBY-4'
nodeType = 'CHEBY-4'

# Quadrature type (wether or not include the right/left bound in the nodes)
# -- 'GAUSS', 'RADAU-LEFT', 'RADAU-RIGHT', 'LOBATTO'
quadType = 'RADAU-RIGHT'

# Generate nodes and polynomial approximation
nodes = genCollocation(M, nodeType, quadType)[0]
approx = LagrangeApproximation(nodes)
interpol = approx.getInterpolationMatrix(t)
poly = interpol.dot(p)

# Plotting
plt.figure()
plt.plot(t, poly, label=f'{quadType}, {nodeType}')
plt.plot(nodes, p, 'o')
plt.grid()
plt.title(f'Interpolant of maximizing poly. for M={M}')
plt.xlabel('Time')
plt.legend()
plt.gcf().set_size_inches(14, 5)
plt.tight_layout()
plt.show()
