#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 20:53:53 2023

@author: cpf5546
"""
import numpy as np
import sympy as sy

import matplotlib.pyplot as plt

from pycode.qmatrix import genCollocation

M = 4

Q = sy.Matrix(
    [[sy.Symbol(f'q{j}{i}')
      for i in range(M)]
     for j in range(M)]
    )

d = [sy.Symbol(f'd{i}') for i in range(M)]
tau = [sy.Symbol(f't{i}') for i in range(M)]

z = sy.Symbol("z")
K = (1-z)*sy.eye(M, dtype=int)+ z*sy.diag(*d) @ Q

eq = K.det() - 1

f = sy.Matrix([eq.subs(z, t) for t in tau])
jac = f.jacobian(d)


quadType = 'RADAU-RIGHT'
distr = 'LEGENDRE'
nodes, _, Qnum = genCollocation(M, distr, quadType)

def nilpotency(d, Q):
    if quadType in ['LOBATTO', 'RADAU-LEFT']:
        d = d[1:]
        Q = Q[1:, 1:]
    M = d.size
    D = np.diag(1/d)
    K = np.eye(M)- D @ Q
    return np.linalg.norm(
        np.linalg.matrix_power(K, M), ord=np.inf)


vals = {s: v for s, v in zip(Q, Qnum.ravel())}
vals.update({s:v for s,v in zip(tau, nodes)})

f = f.subs(vals)
jac = jac.subs(vals)

dn = M/nodes
niln = np.inf
nils = []
res = []
for i in range(30):
    print(i)

    dPrev = dn
    nilPrev = niln

    dVals = {s: v for s, v in zip(d, dn)}
    fn = np.array(f.subs(dVals), dtype=float).ravel()
    Jn = np.array(jac.subs(dVals), dtype=float)

    rhs = Jn @ dn - fn
    dn = np.linalg.solve(Jn, rhs)

    res.append(np.linalg.norm(dPrev-dn, ord=np.inf))
    niln = nilpotency(1/dn, Qnum)
    nils.append(niln)

    if niln > nilPrev and i > 10:
        print("stopped at nil =", nilPrev)
        dn = dPrev
        break

plt.semilogy(nils, label='nilpotency')
# plt.semilogy(res, label='residuum')
plt.legend()
plt.grid(True)
