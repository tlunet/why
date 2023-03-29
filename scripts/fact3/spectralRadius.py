#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:28:32 2022

@author: cpf5546
"""
# Python imports
import numpy as np
import pandas as pd

# Local imports
from pycode.qmatrix import genCollocation
from pycode.optim import findLocalMinima

# ------------------------
# Change these :
M = 4
distr = 'EQUID'
quadType = 'LOBATTO'
# ------------------------

# Generate collocation and adapt to quadrature type
Q = genCollocation(M, distr, quadType)[2]
if quadType in ['RADAU-LEFT', 'LOBATTO']:
    dim = M-1
    Q = Q[1:,1:]
else:
    dim = M

# Maximum coefficient investigated
maxCoeff = 1.


# Function to minimize
def spectralRadius(x):
    R = (Q - np.diag(x))
    return np.max(np.abs(np.linalg.eigvals(R)))


# Use Monte-Carlo Local Minima finder
res, xStarts = findLocalMinima(
    spectralRadius, dim, bounds=(0, maxCoeff), nSamples=500,
    nLocalOptim=3, localOptimTol=1e-10, alphaFunc=1,
    randomSeed=None, threshold=1e-3)

# Note : reduce the threshold for large M

# Manually compute Rho, and plot for 2D optimization
nPlotPts = int(50000**(1/dim))
limits = [maxCoeff]*dim
grids = np.meshgrid(*[
    np.linspace(0, l, num=nPlotPts) for l in limits])
flatGrids = np.array([g.ravel() for g in grids])

# Computing spectral radius for many coefficients
print('Computing spectral radius values')
rho = []
for coeffs in flatGrids.T:
    rho.append(spectralRadius(coeffs))
rho = np.reshape(rho, grids[0].shape)
rho[rho>1] = 1

print('Optimum diagonal coefficients found :')
found = False
for xOpt in res:
    found = True
    print(f' -- xOpt={xOpt} (rho={res[xOpt]:1.2e})')
if not found:
    print(' -- None !')
    
# Store results in Markdown dataframe
try:
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
except Exception:
    df = pd.DataFrame(
        columns=['M', 'quadType', 'distr', 'coeffs', 'rho'])

def formatCoeffs(c):
    out = tuple(round(v, 6) for v in c)
    if quadType in ['RADAU-LEFT', 'LOBATTO']:
        out = (0.,) + out
    return out

def addCoefficients(line, df):
    cond = (df == line.values)
    l = df[cond].iloc[:, :-1].dropna()
    if l.shape[0] == 1:
        # Coefficients already stored
        idx = l.index[0]
        if line.rho[0] < df.loc[idx, 'rho']:
            df.loc[idx, 'rho'] = line.rho[0]
    else:
        # New computed coefficients
        df = pd.concat((df, line), ignore_index=True)
    return df

for c in res:
    line = pd.DataFrame(
        [[M, quadType, distr, formatCoeffs(c), res[c]]],
        columns=df.columns)
    df = addCoefficients(line, df)

df.sort_values(by=['M', 'quadType', 'coeffs'], inplace=True)
df.to_markdown(buf='optimDiagCoeffs.md', index=False)
