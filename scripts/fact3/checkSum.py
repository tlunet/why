#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:36:09 2023

@author: telu
"""
import numpy as np
import pandas as pd

from pycode.qmatrix import genCollocation

df = pd.read_table(
    'optimDiagCoeffs.md', sep="|", header=0, index_col=0,
    skipinitialspace=True).dropna(axis=1, how='all').iloc[1:]
df.reset_index(inplace=True, drop=True)
df.columns = [label.strip() for label in df.columns]
df = df.applymap(lambda x: x.strip())
df['rho'] = df.rho.astype(float)
df['M'] = df.M.astype(int)
df['coeffs'] = df.coeffs.apply(
    lambda x: tuple(float(n) for n in x[1:-1].split(', ')))


def sumValue(M, quadType):
    
    if quadType in ['GAUSS', 'RADAU-RIGHT']:
        nCoeff = M
    else:
        nCoeff = M-1
        
    if quadType == 'GAUSS':
        order = 2*M
    elif quadType == 'LOBATTO':
        order = 2*M-2
    else:
        order = 2*M-1
        
    return nCoeff/order


print('Testing sum of diagonal coefficients :')
combs = set()
for line in df.values:
    M, quadType, distr, x, sr = line
    s = sum(x)
    nodes, _, Q = genCollocation(M, distr, quadType)
    sTh = sumValue(M, quadType)
    if (M, quadType, distr) not in combs:
        print(f'M={M}, quadType={quadType}, distr={distr}')
        print(f' -- nCoeff/order : {sTh}')
        print(f' -- nodes/M : {nodes/M}')
        print(f' -- sum(diag(Q)) : {np.sum(np.diag(Q))}')
        combs.add((M, quadType, distr))
    print(f' -- xOpt={x}, sr={sr}, sum={s}')
