#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 18:26:32 2023

@author: cpf5546
"""
import numpy as np
import scipy as sp
from pycode.qmatrix import genCollocation, genQDelta

M = 3
quadType = 'GAUSS'
distr = 'LEGENDRE'

nodes, _, QFull = genCollocation(M, distr, quadType)
qDelta, dTau = genQDelta(nodes, 'MIN-SR-S', QFull, None, None)

Q = QFull
if quadType in ['LOBATTO', 'RADAU-LEFT']:
    Q = Q[1:, 1:]
    nodes = nodes[1:]

qDetTh = np.prod(nodes)/sp.special.factorial(M)

qDetNum = np.linalg.det(Q)
qDeltaDetNum = np.linalg.det(qDelta)

print(f'M={M}, {quadType}-{distr}')
print(f' -- detTh  : {qDetTh}')
print(f' -- detNum : {qDetNum}')
print(f' -- qDeltaDetNum : {qDeltaDetNum}')
