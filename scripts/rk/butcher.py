#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Butcher analysis for RK and SDC methods
"""
# See README.md for versions and notes
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from time import time

from pycode.qmatrix import genCollocation, genQDelta
from pycode.vectorize import matVecInv, matVecMul


# Script parameters
sdcParams = dict(
    # number of nodes : 2, 3, 4, ...
    M=2,
    # quadrature type : GAUSS, RADAU-RIGHT, RADAU-LEFT, LOBATTO
    quadType="RADAU-RIGHT",
    # node distribution : LEGENDRE, EQUID, CHEBY-1, ...
    distr="LEGENDRE",
    # list of sweeps, ex :
    # -- ["BE"]*3 : 3 sweeps with Backward Euler
    # -- ["BE", "FE"] : BE for first sweep, FE for second sweep
    sweepList=["BEPAR"]*0,
    # initial sweep : COPY (=PIC), or BE, FE, TRAP, ...
    preSweep="TRAPAR",
    # to retrieve the final solution : QUADRATURE or LASTNODE
    postSweep="LASTNODE"
)

class RKAnalysis(object):

    def __init__(self, A, b, c):

        A = np.asarray(A)      # shape (s, s)
        b = np.asarray(b)      # shape (s,)
        c = np.asarray(c)      # shape (s,)

        # Check A dimension
        assert A.shape[0] == A.shape[1], \
            f"A {A.shape} is not a square matrix"
        self.A = A

        # Check b and c dimensions
        assert b.size == self.s, \
            f"b (size={b.size}) has not correct size : {self.s}"
        assert c.size == self.s, \
            f"c (size={c.size}) has not correct size : {self.s}"
        self.b, self.c = b, c

        # Utility matrices
        self.eye = np.eye(self.s)      # shape (s, s)
        self.ones = np.ones(self.s)    # shape (s,)

        # Storage for order
        self._order = None


    @property
    def s(self):
        """Number of stages"""
        return self.A.shape[0]


    def printTableau(self, digit=4):
        """
        Print the Butcher table

        Parameters
        ----------
        digit : int, optional
            Number of digit for each coefficients. The default is 4.
        """
        print("Butcher table :")
        acc = f" .{digit}f"
        bi = ""
        for i in range(self.s):
            bi += f'{float(self.b[i]):{acc}} '
            Ai = ""
            for j in range(self.s):
                Ai += f'{float(self.A[i,j]):{acc}} '
            line = f'{float(self.c[i]):{acc}} | {Ai}'
            if i == 0:
                print('-'*(len(line)-1))
            print(line)
        print('-'*(len(line)-1))
        print(f"{(digit+3)*' '} | {bi}")
        print('-'*(len(line)-1))


    def stabilityFunction(self, z):
        """
        Compute the stability function of the RK method

        Parameters
        ----------
        z : scalar or vector or matrix or ...
            Complex values where to evaluate the stability function.

        Returns
        -------
        sf : scalar or vector or matrix or ...
            Stability function evaluated on the complex values.
        """
        # Pre-processing for z values
        z = np.asarray(z)
        shape = z.shape
        z = z.ravel()[:, None, None]    # shape (nValues, 1, 1)

        print(f"Computing Butcher stability function for {z.size} values")
        tBeg = time()

        # Prepare Butcher tables
        A = self.A[None, :, :]          # shape (1, s, s)
        b = self.b[None, None, :]       # shape (1, 1, s)
        eye = self.eye[None, :, :]      # shape (1, s, s)
        ones = self.ones[:, None]       # shape (s, 1)

        # Compute 1 + z*b @ (I-zA)^{-1} @ ones
        if self.s == 1:
            sf = 1 + (z*b)*(eye - z*A)**(-1)
        else:
            sf = 1 + (z*b) @ matVecInv(eye - z*A, ones)

        # Reshape to original size (scalar or vector)
        sf.shape = shape

        print(f' -- done in {time()-tBeg:.2f}s')
        return sf

    @property
    def orderFromJulia(self):
        """Compute the order from Butcher table using Julia script"""
        if self._order is not None:
            return self._order
        np.save('A.npy', self.A)
        np.save('b.npy', self.b.ravel())
        np.save('c.npy', self.c.ravel())
        try:
            with open('config.json', 'r') as f:
                conf = json.load(f)
            juliaPath = conf["juliaPath"]
        except Exception:
            juliaPath = "/Applications/Julia-1.9.app/Contents/Resources/julia/bin/julia"
        try:
            print('Computing Butcher order with Julia script')
            tBeg = time()
            out = os.popen(f'{juliaPath} bseries.jl').read()
            print(f' -- done in {time()-tBeg:.2}s')
            order = int(out)
            self._order = order
            return order
        except Exception:
            raise SystemError(f"Error when running julia script : {out}")

    @property
    def order(self):
        conditions = {
            1: [('i', 1)],
            2: [('i,ij', 1/2)],
            3: [('i,ij,ik', 1/3),
                ('i,ij,jk', 1/6)],
            4: [('i,ij,ik,il', 1/4),
                ('i,ij,jl,ik', 1/8),
                ('i,ij,jk,jl', 1/12),
                ('i,ij,jk,kl', 1/24)],
            5: [('i,ij,ik,il,im', 1/5),
                ('i,ij,jk,il,im', 1/10),
                ('i,ij,jk,jl,im', 1/15),
                ('i,ij,jk,kl,im', 1/30),
                ('i,ij,jk,il,lm', 1/20),
                ('i,ij,jk,jl,jm', 1/20),
                ('i,ij,jk,kl,jm', 1/40),
                ('i,ij,jk,kl,km', 1/60),
                ('i,ij,jk,kl,lm', 1/120)],
        }
        print('Computing Butcher order from Python')
        tBeg = time()
        A, b = self.A, self.b
        order = 0
        for o, conds in conditions.items():
            for sSum, val in conds:
                s = np.einsum(sSum, b, *[A]*sSum.count(','), optimize="greedy").sum()
                if not np.isclose(s, val):
                    print(f' -- done in {time()-tBeg:.2}s')
                    return order
            order = o
        print(f' -- done in {time()-tBeg:.2}s')
        return order


    def plotNumericalOrder(self, lam,
                           dts=[1, 1e-1, 1e-2, 1e-3, 1e-4], tEnd=1, u0=1):
        dts = np.asarray(dts)
        nSteps = np.round(tEnd/dts).astype(int)

        # Compute numerical solution with stability function
        uNum = u0 * self.stabilityFunction(lam*dts)**nSteps

        # Compute exact solution
        uExact = u0 * np.exp(lam*dts*nSteps)

        # Compute absolute error
        error = np.abs(uNum-uExact)

        # Plot error in log scale
        plt.figure()
        order = self.order
        plt.loglog(dts, error, linestyle='', marker='x', label='Numerical')
        plt.loglog(dts, error[0]*np.asarray(dts)**order, '--', label=f'Theorical ({order})')
        plt.loglog(dts, error[0]*np.asarray(dts)**(order+1), '--', color="gray")
        plt.loglog(dts, error[0]*np.asarray(dts)**(order-1), '--', color="gray")
        plt.legend()
        plt.xlabel('time-step size')
        plt.ylabel('error')
        plt.tight_layout()


class SDCAnalysis(RKAnalysis):

    def __init__(self, M=3, quadType="RADAU-RIGHT", distr="LEGENDRE",
                 sweepList=2*["BE"], preSweep="BE", postSweep="LASTNODE"):

        # Collocation coefficients
        nodes, weights, Q = genCollocation(M, distr, quadType)
        self.nodes, self.weights, self.Q = nodes, weights, Q

        # SDC coefficients
        if preSweep == "COPY": preSweep = "PIC"
        QDeltaInit, dtau = genQDelta(nodes, preSweep, Q)
        QDeltaList = [genQDelta(nodes, sType, Q)[0] for sType in sweepList]
        self.QDeltaInit, self.dtau, self.QDeltaList = QDeltaInit, dtau, QDeltaList

        # Build Butcher table
        zeros = np.zeros_like(Q)
        nSweep = len(QDeltaList)
        # -- build A
        A = np.hstack([QDeltaInit] + nSweep*[zeros])
        for k, QDelta in enumerate(QDeltaList):
            A = np.vstack(
                [A,
                 np.hstack(k*[zeros]
                           + [Q - QDelta, QDelta]
                           + (nSweep-k-1)*[zeros])
                ])
        # -- build b and c
        zeros = np.zeros_like(nodes)
        self.postSweep = postSweep
        if postSweep == "QUADRATURE":
            b = np.hstack(nSweep*[zeros]+[weights])
        elif postSweep == "LASTNODE":
            if quadType in ["GAUSS", "RADAU-LEFT"]:
                raise ValueError("cannot use LASTNODE with GAUSS or RADAU-LEFT")
            b = A[-1]
        else:
            raise NotImplementedError(f"postSweep={postSweep}")
        c = np.hstack((nSweep+1)*[nodes])
        # -- add dtau term if needed
        if np.any(dtau != 0):
            newA = np.zeros([s+1 for s in A.shape])
            newA[1:, 1:] = A
            newA[1:self.M+1, 0] = dtau
            A = newA
            b = np.hstack([0, b])
            c = np.hstack([0, c])

        # Call parent constructor
        super().__init__(A, b, c)

    @property
    def nSweep(self):
        return len(self.QDeltaList)

    @property
    def M(self):
        return len(self.nodes)


    def numericalSolution(self, z, u0=1):
        """
        Compute the numerical solution using SDC matrices

        Parameters
        ----------
        z : scalar or vector or matrix or ...
            Complex values where to evaluate the stability function.
        u0 : scalar, optional
            Value for the initial solution. The default is 1.

        Returns
        -------
        u : scalar or vector or matrix or ...
            Numerical solution evaluated on the complex values.
        """
        # Pre-processing for z values
        z = np.asarray(z)
        shape = z.shape
        z = z.ravel()[:, None, None]    # shape (nValues, 1, 1)

        print(f"Computing SDC numerical solution for {z.size} values")
        tBeg = time()

        # Prepare SDC matrices to shape (1, M, M)
        Q = self.Q[None, :, :]
        QDeltaInit = self.QDeltaInit[None, :, :]
        QDeltaList = [QDelta[None, :, :] for QDelta in self.QDeltaList]

        # Utilities
        uInit = u0*np.ones(self.M)[None, :]     # shape (1, M)
        I = np.eye(self.M)[None, :, :]          # shape (1, M, M)

        # Pre-sweep : u^{0} = (I-zQDeltaI)^{-1} @ (1+z*dtau)*uInit
        rhs = (1 + z[:, 0]*self.dtau)*uInit     # shape (nValues, M)
        u = matVecInv(I - z*QDeltaInit, rhs)    # shape (nValues, M)

        # Sweeps u^{k+1} = (I-zQDelta)^{-1} @ z(Q-QDelta) @ u^{k+1}
        #                  + (I-zQDelta)^{-1} @ uInit
        for QDelta in QDeltaList:
            L = I - z*QDelta            # shape (nValues, M, M)
            u = matVecInv(L, matVecMul(z*(Q - QDelta), u))
            u += matVecInv(L, uInit)    # shape (nValues, M)

        # Post-sweep
        if self.postSweep == "QUADRATURE":
            u = u0 + z*self.weights[None, None, :] @ u[:, :, None]
        elif self.postSweep == "LASTNODE":
            h = np.zeros(self.M)
            h[-1] = 1
            u = h[None, None, :] @ u[:, :, None]

        # Reshape to original size (scalar or vector)
        u.shape = shape

        print(f' -- done in {time()-tBeg:.2f}s')
        return u


    def plotAll(self, a=-3, b=1, n=200, eps=1e-2):

        real = np.linspace(a, b, n)
        imag = np.linspace(a, -a, n)
        X, Y = np.meshgrid(real, imag)

        z = real[:, None] + 1j*imag[None, :]

        Rz = self.stabilityFunction(z)
        Sz = Rz / np.exp(z)
        Az = Rz - np.exp(z)
        SDCRz = self.numericalSolution(z)

        absRz = abs(Rz.T)
        absSz = abs(Sz.T)
        absAz = abs(Az.T)
        absSDCRz = abs(SDCRz.T)

        fig = plt.figure()
        axs = fig.subplots(nrows=2, ncols=2)

        # Plot parameters
        cParams = dict(levels=[1], colors=['k'], linewidths=2)
        cfParams = dict(colors=['white', 'C1', 'lightgrey'])
        lParams = dict(linewidth=1, linestyle='--', color='k')
        aParams = dict(levels=[1e-1, 1e1], colors=['b', 'r'], linewidths=1)
        custom_line = [Line2D([0], [0], color='k', lw=2)]
        custom_lines = [Line2D([0], [0], color='k', lw=2),
                        Line2D([0], [0], color='b', lw=1),
                        Line2D([0], [0], color='r', lw=1)]
        customLegends = [
            r'$|R(z)/e^z|=1$', r'$|R(z)/e^z|=0.1$', r'$|R(z)/e^z|=10$']
        stabLevels = [0, 1-1e-10, 1+1e-10, 1e50]
        accLevels = [0, 1-1e-15, 1+1e-15, 1e50]

        # Stability contour (from Butcher)
        ax = axs[0, 0]
        ax.contour(X, Y, absRz, **cParams)
        ax.contourf(X, Y, absRz, levels=stabLevels, **cfParams)
        ax.plot([0, 0], [a, -a], **lParams)
        ax.plot([a, b], [0, 0], **lParams)
        ax.set_xlabel(r'$\mathcal{R}(z)$')
        ax.set_ylabel(r'$\mathcal{I}(z)$')
        ax.legend(custom_line, [r'$|R(z)|=1$'])
        ax.set_aspect('equal')

        # Accuracy contour (Sz)
        ax = axs[0, 1]
        ax.contour(X, Y, absSz, **cParams)
        ax.contour(X, Y, absSz, **aParams)
        ax.contourf(X, Y, absSz, levels=accLevels, **cfParams)
        ax.plot([0, 0], [a, -a], **lParams)
        ax.plot([a, b], [0, 0], **lParams)
        ax.set_xlabel(r'$\mathcal{R}(z)$')
        ax.set_ylabel(r'$\mathcal{I}(z)$')
        ax.legend(custom_lines, customLegends)
        ax.set_aspect('equal')

        # Accuracy contour (Az)
        ax = axs[1, 0]
        ax.contour(X, Y, absAz, linewidths=2, levels=[eps])
        ax.set_aspect('equal')
        ax.legend(custom_line, [f'$|R(z)-e^z|={eps}$'])

        ax = axs[1, 1]
        ax.contour(X, Y, absSDCRz, **cParams)
        ax.contourf(X, Y, absSDCRz, levels=stabLevels, **cfParams)
        ax.plot([0, 0], [a, -a], **lParams)
        ax.plot([a, b], [0, 0], **lParams)
        ax.set_xlabel(r'$\mathcal{R}(z)$')
        ax.set_ylabel(r'$\mathcal{I}(z)$')
        ax.legend(custom_line, [r'$|R(z)|=1$'])
        ax.set_aspect('equal')

        fig.tight_layout()


if __name__ == "__main__":
    # Instantiate Butcher analysis object
    sdc = SDCAnalysis(**sdcParams)

    # Analysis
    sdc.printTableau()
    order= sdc.order
    print(f"Order from Python : {order}")

    # Plots
    sdc.plotAll()
    sdc.plotNumericalOrder(1j-0.1)
    plt.show()
