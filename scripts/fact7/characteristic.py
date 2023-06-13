#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:10:18 2023

@author: telu
"""
import numpy as np
import scipy as sp
import scipy.optimize as sco
import sympy as sy
from pycode.qmatrix import genCollocation

M = 3
quadType = 'GAUSS'
distr = 'LEGENDRE'
fullSym = True

nodes, _, QNum = genCollocation(M, distr, quadType)

nCoeffs = M
if quadType in ['LOBATTO', 'RADAU-LEFT']:
    nCoeffs -= 1;
    QNum = QNum[1:, 1:]
    nodes = nodes[1:]

qDetNum = np.prod(nodes)/sp.special.factorial(M)


qCoeffs = [[sy.Symbol('q_{'+str(i)+','+str(j)+'}')
            for i in range(nCoeffs)]
           for j in range(nCoeffs)]

if fullSym:
    Q = sy.Matrix(qCoeffs)
else:
    Q = sy.Matrix(QNum)

xCoeffs = [sy.Symbol(f'x_{i}') for i in range(nCoeffs)]

D = sy.diag(*xCoeffs)

lam = sy.Symbol('\\lambda')

Korig =  D.inv()*Q - sy.eye(nCoeffs)
K = D.inv()*Q - (1+lam)*sy.eye(nCoeffs)


min3Sol = [False]
def addMIN3Solutions(sol):
    # AHAHA we beat it !
    coeffs = None
    if distr == 'LEGENDRE':
        if quadType == 'LOBATTO':
            if M == 7:
                coeffs = [
                    0.18827968699454273,
                    0.1307213945012976,
                    0.04545003319140543,
                    0.08690617895312261,
                    0.12326429119922168,
                    0.13815746843252427,
                ]
            if M == 5:
                coeffs = [0.2994085231050721, 0.07923154575177252, 0.14338847088077,
                          0.17675509273708057]
            if M == 4:
                coeffs = [0.2865524188780046, 0.11264992497015984, 0.2583063168320655]
            if M == 3:
                coeffs = [0.2113181799416633, 0.3943250920445912]
        if quadType == 'RADAU-RIGHT':
            if M == 5:
                coeffs = [0.2818591930905709, 0.2011358490453793, 0.06274536689514164,
                          0.11790265267514095, 0.1571629578515223]
            if M == 4:
                coeffs = [0.3198786751412953, 0.08887606314792469, 0.1812366328324738,
                          0.23273925017954]
            if M == 3:
                coeffs = [0.3203856825077055, 0.1399680686269595, 0.3716708461097372]
    if distr == 'EQUID':
        if quadType == 'RADAU-RIGHT':
            if M == 7:
                coeffs = [0.0582690792096515, 0.09937620459067688, 0.13668728443669567,
                          0.1719458323664216, 0.20585615258818232, 0.2387890485242656,
                          0.27096908017041393]
            if M == 5:
                coeffs = [0.0937126798932547, 0.1619131388001843, 0.22442341539247537,
                          0.28385142992912565, 0.3412523013467262]
            if M == 4:
                coeffs = [0.13194852204686872, 0.2296718892453916, 0.3197255970017318,
                          0.405619746972393]
            if M == 3:
                coeffs = [0.2046955744931575, 0.3595744268324041, 0.5032243650307717]
    if coeffs is not None:
        min3Sol[0] = True
        sol += [np.array(coeffs)]


if fullSym:
    print('Computing determinant')
    poly = sy.poly(K.det(), lam)

    coeffs = poly.coeffs()

    xProd = sy.prod(xCoeffs)
    equations = [c*xProd for c in coeffs[1:]]

    # equations.append(xProd-qDetNum)
    # qDet = sy.Symbol('\\Delta')
    # equations = [c.subs({xProd: qDet}) for c in nullCoeffs]

    print('Solving symbolically')
    if M < 3:
        sol = sy.solve(equations, *xCoeffs)
    else:
        sol = []

    if len(sol) > 0:
        print('Found solution(s) symbolically')
    else:
        print('Solving numerically')
        sol = [sy.nsolve(equations, xCoeffs, nodes)]
        if len(sol) > 0:
            print('Found solution numerically')


    sol = [np.array(c, dtype=float).ravel() for c in sol]
    addMIN3Solutions(sol)

    if len(sol) > 0:
        for xCoeffNum in sol:
            print(f'diagonal coefficients : {xCoeffNum}')
            DNumInv = np.diag(1/xCoeffNum)

            KNum = np.eye(nCoeffs) - DNumInv @ QNum

            KPow = np.eye(nCoeffs)
            for m in range(nCoeffs):
                KPow = KNum @ KPow
                print('|prod(I - Dinv @ Q)|_max = {:<7.4e}, m = 1, ..., {:<3}   ro(prod) = {}'.format(
                    np.max(np.abs(KPow)), m+1, max(np.abs(np.linalg.eigvals(KPow)))))
                # print(np.linalg.cond(KPow))
        if min3Sol[0]:
            print(" --> last diagonal coefficients are from MIN3 optimization")

else:

    def func(coeffs):
        coeffs = np.asarray(coeffs)
        kMats = [(1-z)*np.eye(nCoeffs) + z*np.diag(1/coeffs) @ QNum
                 for z in nodes]
        vals = [np.linalg.det(K)-1 for K in kMats]
        return np.array(vals)

    sol = [sco.fsolve(func, nodes/M, xtol=1e-14)]
    addMIN3Solutions(sol)

    for xCoeffNum in sol:
        print(f'diagonal coefficients : {xCoeffNum}')
        DNumInv = np.diag(1/xCoeffNum)

        KNum = np.eye(nCoeffs) - DNumInv @ QNum

        KPow = np.eye(nCoeffs)
        for m in range(nCoeffs):
            KPow = KNum @ KPow
            print('|prod(I - Dinv @ Q)|_max = {:<7.4e}, m = 1, ..., {:<3}   ro(prod) = {}'.format(
                np.max(np.abs(KPow)), m+1, max(np.abs(np.linalg.eigvals(KPow)))))
    if min3Sol[0]:
        print(" --> last diagonal coefficients are from MIN3 optimization")
