#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function to generate diagonal QDelta matrix minimizing some property
of a part of the SDC iteration matrix.

MIN-SR-NS : 
    Minimize Spectral Radius of Non-Stiff limit, that is Q-QDelta
MIN-SR-S : 
    Minimize Spectral Radius of Stiff limif, that is I-QDelta^{-1}Q
"""
import numpy as np
import scipy as sp
from pycode.utils import getQuadType

# Parameters for generation of diagonal coefficients
SETTINGS = {
    "MIN-SR-S":
        {"solver": "fsolve",  
         # options : "root-hybrid", "fsolve"
         "initSol": "NS",
         # options : "NS" (non-stiff), "NODES", array of M values
         "solverTol": 1e-14,
         # options : any low floating point number
         
         "zValues": "NODES",
         # options : "NODES", "1/M", array of M values
         "cCoeff": 1,
         # options : any positive integer
         }
    }

def genDiagQDelta(nodes, Q, minType='MIN-SR-S', verbose=False):
    
    nodes = np.asarray(nodes).ravel()
    M = nodes.size
    
    if minType == 'MIN-SR-NS':
        return nodes/M
    elif minType != "MIN-SR-S":
        raise ValueError(f'minType={minType} not implemented')
    
    quadType = getQuadType(Q)
    
    nCoeffs = M
    if quadType in ['LOBATTO', 'RADAU-LEFT']:
        nCoeffs -= 1;
        Q = Q[1:, 1:]
        nodes = nodes[1:]
        
    zValues = SETTINGS["MIN-SR-S"]["zValues"]
    if zValues == "NODES":
        zValues = nodes
    elif zValues == "1/M":
        zValues = 1/(np.arange(nCoeffs)+1)
    elif zValues == "M":
        zValues = np.arange(nCoeffs)+1
    else:
        try:
            zValues = np.array(zValues).ravel()
            assert zValues.size == M
        except Exception:
            raise ValueError(f'zValues={zValues} not implemented')
           
            
    c = SETTINGS["MIN-SR-S"]["cCoeff"]
    
    def func(coeffs):
        coeffs = np.asarray(coeffs)
        kMats = [(1-z)*np.eye(nCoeffs) + z*np.diag(1/coeffs) @ Q
                 for z in zValues]
        vals = [np.linalg.det(c*K)-c**M for K in kMats]
        return np.array(vals)
    
    initSol = SETTINGS["MIN-SR-S"]["initSol"]
    if initSol == "NS":
        initSol = nodes/M
    elif initSol == "NODES":
        initSol = nodes
    else:
        try:
            initSol = np.array(initSol).ravel()
            assert initSol.size == M
        except Exception:
            raise ValueError(f'initSol={initSol} not implemented')
        
    tol = SETTINGS["MIN-SR-S"]["solverTol"]
    solver = SETTINGS["MIN-SR-S"]["solver"]
    
    if solver == "fsolve":
        sol, infos, ier, msg = sp.optimize.fsolve(func, initSol, xtol=tol, full_output=True)
        infos.update({'ier': ier, 'msg': msg})
    elif solver == "root-hybrid":
        sol = sp.optimize.root(func, initSol, tol=tol, method='hybr')
        sol, infos = sol.x, sol
        
    if verbose:
        print("Solver infos :")
        print(infos)
        print("Solution infos :")
        print('x = ', sol)
        K = np.diag(1/sol) @ Q - np.eye(nCoeffs)
        Kpow = np.eye(nCoeffs)
        for i in range(nCoeffs):
            Kpow = K @ Kpow
            print('|prod(I - Dinv @ Q)|_max = {:<7.4e}, m = 1, ..., {:<3}   ro(prod) = {}'.format(
                np.max(np.abs(Kpow)), i + 1, max(np.abs(np.linalg.eigvals(Kpow)))))
        return sol, np.max(np.abs(Kpow))
    
    return sol

if __name__ == "__main__":
    from qmatrix import genCollocation
    import matplotlib.pyplot as plt
    
    quadType = 'LOBATTO'
    distr = 'LEGENDRE'

    def plotParameterImpact(parName, values):
        
        params = SETTINGS['MIN-SR-S']
        paramsOrig = params.copy()
        plt.figure()
        
        for par in values:
            params[parName] = par
            
            mValues = np.arange(3, 11)
            epsValues = []
            for M in mValues:
                nodes, _, Q = genCollocation(M, distr, quadType)
                print("-"*80)
                print(f"Computing diagonal coeffs for M={M}")
                print("-"*80)
                coeffs, eps = genDiagQDelta(nodes, Q, verbose=True)
                epsValues.append(eps)
                
            label = f'{par:1.0e}' if parName == "solverTol" else par
            plt.semilogy(mValues, epsValues, label=label)
            
        SETTINGS['MIN-SR-S'] = paramsOrig
            
        plt.legend()
        plt.title(f'{quadType}-{distr}, variation of {parName}')
        plt.xlabel('M')
        plt.ylabel(r'$||K^{M}||_\infty$')
        plt.tight_layout()
        plt.grid()
        
    plotParameterImpact('solverTol', [1e-4, 1e-7, 1e-10, 1e-14])
    plotParameterImpact('zValues', ["NODES", "1/M", "M"])
    plotParameterImpact('initSol', ["NODES", "NS"])
        