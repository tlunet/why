#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:12:29 2023

@author: cpf5546
"""
import numpy as np
import matplotlib.pyplot as plt

from pycode.qmatrix import genCollocation, genQDelta

quadType = 'GAUSS'
distr = 'EQUID'

plt.figure()
for M in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    nodes, _, Q = genCollocation(M, distr, quadType)
    qDelta, dTau = genQDelta(nodes, 'MIN-SR-S', Q, None, None)

    plt.plot(nodes, np.diag(qDelta)*M, 'o-')
plt.grid()

nodes, _, Q = genCollocation(2, distr, quadType)
qDelta, dTau = genQDelta(nodes, 'MIN-SR-S', Q, None, None)
