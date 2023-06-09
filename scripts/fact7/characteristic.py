#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:10:18 2023

@author: telu
"""
import numpy as np
import scipy as sp
import sympy as sy
from pycode.qmatrix import genCollocation

M = 3
quadType = 'LOBATTO'
distr = 'LEGENDRE'
fullSym = True

nodes, _, QNum = genCollocation(M, distr, quadType)

if quadType in ['LOBATTO', 'RADAU-LEFT']:
    M -= 1;
    QNum = QNum[1:, 1:]
    nodes = nodes[1:]
    
qDetNum = np.prod(nodes)/sp.special.factorial(M)
    

qCoeffs = [[sy.Symbol('q_{'+str(i)+','+str(j)+'}') 
            for i in range(M)] 
           for j in range(M)]

if fullSym:
    Q = sy.Matrix(qCoeffs)
else:
    Q = sy.Matrix(QNum)

xCoeffs = [sy.Symbol(f'x_{i}') for i in range(M)]

D = sy.diag(*xCoeffs)

lam = sy.Symbol('\\lambda')

Korig =  D.inv()*Q - sy.eye(M)
K = D.inv()*Q - (1+lam)*sy.eye(M)

print('Computing determinant')
poly = sy.poly(K.det(), lam)

coeffs = poly.coeffs()

xProd = sy.prod(xCoeffs)
equations = [c*xProd for c in coeffs[1:]]

# equations.append(xProd-qDetNum)

qDet = sy.Symbol('\\Delta')

# nullCoeffs = [c.subs({xProd: qDet}) for c in nullCoeffs]

if not fullSym:
    print('Solving symbolically')
    sol = sy.solve(equations, *xCoeffs)
    
    if len(sol) > 0:
        print('Found solution(s) symbolically')
    else:
        print('Solving numerically')
        sol = [sy.nsolve(equations, xCoeffs, nodes)]
        if len(sol) > 0:
            print('Found solution numerically')
    
    
    sol = [np.array(c, dtype=float).ravel() for c in sol]
    # AHAHA we beat it !
    # sol += [np.array([0.2865524188780046, 0.11264992497015984, 0.2583063168320655])]
    
    if len(sol) > 0:
        for xCoeffNum in sol:
            print(f'diagonal coefficients : {xCoeffNum}')
            DNumInv = np.diag(1/xCoeffNum)
            
            KNum = np.eye(M) - DNumInv @ QNum
            
            KPow = np.eye(M)
            for m in range(M):
                KPow = KNum @ KPow
                print('|prod(I - Dinv @ Q)|_max = {:<7.4e}, m = 1, ..., {:<3}        ro(prod) = {}'.format(
                    np.max(np.abs(KPow)), m+1, max(np.abs(np.linalg.eigvals(KPow)))))
                print(np.linalg.cond(KPow))
    
    
