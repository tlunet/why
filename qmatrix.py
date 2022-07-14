#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:03:45 2021

@author: telu
"""
import numpy as np
from scipy.linalg import lu

# Pythag library can be found at https://gitlab.com/tlunet/pythag
from pythag.applications import SpectralApproximation, LagrangeApproximation

import gfma


class Results(object):
    """
    Implement a conveniency object that can be used to return multiple results
    and access them easily.

    Example
    -------
    >>> res = Results(out=[1,2,3], tol=1e-5, nIter=10)
    >>> print(res.out)
    [1, 2, 3]
    >>> print(res['tol'])
    1e-05
    >>> out, tol, nIter = res
    >>> print(out, tol, nIter)
    [1, 2, 3] 1e-05 10
    >>> out, tol = res[:2]
    >>> print(out, tol)
    [1, 2, 3] 1e-05
    >>> print(res[0])
    [1, 2, 3]
    """

    def __init__(self, **kwargs):
        try:
            self._res = dict(kwargs)
        except TypeError:
            raise ValueError(f'invalid argument : {kwargs}')

    @property
    def keys(self):
        return list(self._res.keys())

    def update(self, **kwargs):
        try:
            self._res.update(kwargs)
        except Exception:
            raise ValueError(f'invalid argument : {kwargs}')

    def __len__(self):
        return len(self._res)

    def __getitem__(self, key):
        try:
            # Access by key
            return self._res[key]

        except KeyError:
            try:
                # Access by position
                return self._res[self.keys[key]]
            except IndexError:
                raise IndexError(f'wrong index : {key}')
            except TypeError:
                raise AttributeError(f'wrong attribute : {key}')

        except TypeError:
            try:
                # Access by slice
                return [self._res[self.keys[k]]
                        for k in range(*key.indices(len(self)))]
            except Exception:
                raise AttributeError(f'bad indexing : {key}')

    def __setattr__(self, name, value):
        if name == '_res':
            super().__setattr__(name, value)
        else:
            raise Exception(
                'cannot modify object results, use the update method instead')

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return self.__getitem__(name)

    def __str__(self):
        return self._res.__str__()

    def __repr__(self):
        return self._res.__repr__()


def genPolyApprox(M, distr, quadType, implementation='LAGRANGE', scaling=True):
    if distr == 'GIVEN':
        ap = LagrangeApproximation(
            M, weightComputation='STABLE', scaleRef='MAX')
    else:
        if quadType not in ['GAUSS', 'RADAU-I', 'RADAU-II', 'LOBATTO']:
            raise NotImplementedError(f'quadType={quadType}')
        if distr in ['LEGENDRE', 'CHEBY-1', 'CHEBY-2', 'CHEBY-3', 'CHEBY-4']:
            if implementation == 'SPECTRAL':
                ap = SpectralApproximation(
                    M, pType=distr, qType=quadType,
                    bounds=[0, 1] if scaling else [-1, 1])
            elif implementation == 'LAGRANGE':
                ap = LagrangeApproximation(
                    (distr, M), qType=quadType)
                if scaling:
                    ap.points += 1
                    ap.points /= 2
                ap = LagrangeApproximation(ap.points)
            else:
                raise ValueError(f'implementation={implementation}')
            nodes = ap.points
        elif distr == 'EQUID':
            a, b = (0, 1) if scaling else (-1, 1)
            nodes = np.linspace(a, b, M+2)[1:-1] if quadType == 'GAUSS' else \
                np.linspace(a, b, M+1)[:-1] if quadType == 'RADAU-I' else \
                np.linspace(a, b, M+1)[1:] if quadType == 'RADAU-II' else \
                np.linspace(a, b, M)  # LOBATTO
            ap = LagrangeApproximation(nodes, weightComputation='STABLE',
                                       scaleRef='MAX')
        else:
            raise NotImplementedError(f'distr={distr}')
    return ap


def genQDelta(sweepType, deltas, Q):
    M = Q.shape[0]
    QDelta = np.zeros((M, M))
    if sweepType in ['BE', 'FE']:
        offset = 1 if sweepType == 'FE' else 0
        for i in range(offset, M):
            QDelta[i:, :M-i] += np.diag(deltas[offset:M-i+offset])
    elif sweepType == 'TRAP':
        for i in range(0, M):
            QDelta[i:, :M-i] += np.diag(deltas[:M-i])
        for i in range(1, M):
            QDelta[i:, :M-i] += np.diag(deltas[1:M-i+1])
        QDelta /= 2.0
    elif sweepType == 'LU':
        QT = Q.T.copy()
        [_, _, U] = lu(QT, overwrite_a=True)
        QDelta = U.T
    elif sweepType == 'EXACT':
        QDelta = np.copy(Q)
    else:
        raise NotImplementedError(f'sweepType={sweepType}')
    return QDelta


def genQMatrices(M, distr, quadType, sweepType,
                 implementation='LAGRANGE', cPoints=None, scaling=True):
    """
    Generate the Q, QDelta and H matrix for any given type of SDC sweep

    Parameters
    ----------
    M : int
        Number of quadrature nodes.
    distr : str
        Node distribution. Can be selected from :
    - LEGENDRE : nodes from the Legendre polynomials
    - EQUID : equidistant nodes distribution
    - CHEBY-{1,2,3,4} : nodes from the Chebyshev polynomial (1st to 4th kind)
    quadType : str
        Quadrature type. Can be selected from :
    - GAUSS : do not include the boundary points in the nodes
    - RADAU-I : include left boundary points in the nodes
    - RADAU-II : include right boundary points in the nodes
    - LOBATTO : include both boundary points in the nodes
    sweepType : str
        Type of sweep, that defines QDelta. Can be selected from :
    - BE : Backward Euler sweep (first order)
    - FE : Forward Euler sweep (first order)
    - LU : uses the LU trick
    - TRAP : sweep based on Trapezoidal rule (second order)
    scaling : bool
        Wether or not scale the nodes between [0, 1]

    Returns
    ----
    Q : array (M,M)
        matrix of the collocation problem
    QDelta : array (M,M)
        matrix of the sweep
    H : array (M,M)
        interpolation matrix (from nodes to right bound)
    nodes : array (M,)
        quadrature nodes, scaled to [0, 1] (or not ...)
    """
    # Wether or not coarse level is defined
    useCoarse = cPoints is not None

    # Generate nodes and approximation
    ap = genPolyApprox(M, distr, quadType, implementation, scaling)
    nodes = ap.points
    M = ap.n
    if useCoarse:
        try:
            cPoints = int(cPoints)
        except TypeError:
            distr = 'GIVEN'
            try:
                cPoints = np.asarray(cPoints, dtype=float).ravel()
            except TypeError:
                if not isinstance(cPoints, slice):
                    raise ValueError(f'bad argument for cPoints = {cPoints}')
                cPoints = nodes[cPoints]
        apCoarse = genPolyApprox(cPoints, distr, quadType, implementation)
        TCF = apCoarse.getInterpolationMatrix(ap.points)
        TFC = ap.getInterpolationMatrix(apCoarse.points)
        nodesTilde = apCoarse.points

    # Generate deltas and Q
    deltas = np.copy(nodes)
    deltas[1:] = np.ediff1d(nodes)
    Q = ap.getIntegrationMatrix([(0 if scaling else -1, tau) for tau in nodes])
    if useCoarse:
        deltasTilde = np.copy(nodesTilde)
        deltasTilde[1:] = np.ediff1d(nodesTilde)
        QTilde = apCoarse.getIntegrationMatrix(
            [(0 if scaling else -1, tau) for tau in nodesTilde])

    # Generate QDelta
    QDelta = genQDelta(sweepType, deltas, Q)
    if useCoarse:
        QDeltaTilde = genQDelta(sweepType, deltasTilde, QTilde)

    # Generate interpolation matrix H
    H = ap.getInterpolationMatrix([1]).repeat(M, axis=0)
    if useCoarse:
        HTilde = apCoarse.getInterpolationMatrix([1]).repeat(
            QTilde.shape[0], axis=0)

    res = Results(Q=Q, QDelta=QDelta, H=H, nodes=nodes)
    if useCoarse:
        res.update(QTilde=QTilde, QDeltaTilde=QDeltaTilde, HTilde=HTilde,
            nodesTilde=nodesTilde, TCF=TCF, TFC=TFC)
    return res


class BlockSDC(object):

    GAMMA_EXPONENT = 1
    USE_RESTRICTED = False
    NORM_TYPE = 'L_INF'
    APPROX_EXACT = False
    APPROX_EXACT_COARSE = False

    def __init__(self, lam, u0, tEnd, L, M, cPoints=None,
                 distr='LEGENDRE', quadType='LOBATTO', sweepType='BE',
                 initCond='U0', implementation='LAGRANGE'):

        # Wether or not coarse level is available
        useCoarse = cPoints is not None

        # Build time grid
        times = np.linspace(0, tEnd, num=L+1)
        dt = times[1] - times[0]

        # Generate quadrature operators and nodes
        res = genQMatrices(
            M, distr, quadType, sweepType, implementation, cPoints)
        Q, QDelta, H, nodes = res[:4]
        if self.APPROX_EXACT:
            Q = QDelta
        if useCoarse:
            QTilde, QDeltaTilde, HTilde, nodesTilde, TCF, TFC = res[4:]
            if self.APPROX_EXACT_COARSE:
               QTilde = QDeltaTilde
            MTilde = len(nodesTilde)

        # Build block operators
        M = len(nodes)
        I = np.identity(M)
        ImQDelta = I - lam*dt*QDelta
        ImQ = I - lam*dt*Q

        if useCoarse:
            ITilde = np.identity(MTilde)
            ImQDeltaTilde = ITilde - lam*dt*QDeltaTilde
            ImQTilde = ITilde - lam*dt*QTilde
            HDelta = TFC.dot(H) - HTilde.dot(TFC)

        # Custom norm
        w, V = np.linalg.eig(Q-QDelta)
        Vinv = np.linalg.inv(V)
        self.V, self.Vinv = V, Vinv

        self.normMat = getattr(self, f'normMat_{self.NORM_TYPE}')
        self.normVec = getattr(self, f'normVec_{self.NORM_TYPE}')

        # Coefficients for error bounds
        norm = self.normMat
        if abs(lam*dt)*norm(QDelta) < 1 and self.USE_RESTRICTED:
            gamma = abs(lam*dt) * norm(Q-QDelta) / \
                (1 - abs(lam*dt)*norm(QDelta))
            nN1 = 1
            N1 = 1
            nN2 = norm(H)/(1 - abs(lam*dt)*norm(QDelta))
        else:
            N1 = np.linalg.solve(ImQDelta, Q-QDelta)
            nN1 = norm(N1)
            gamma = abs(lam*dt) * nN1
            N2 = np.linalg.solve(ImQDelta, H)
            nN2 = norm(N2)
        gamma **= self.GAMMA_EXPONENT

        dico = {}
        solve = np.linalg.solve
        dico['GaussSeidel'] = {
            'gamma': norm(solve(ImQDelta, (lam*dt) * (Q-QDelta))),
            'beta': norm(solve(ImQDelta, H))}
        dico['SDC'] = dico['GaussSeidel']
        dico['Jacobi'] = {
            'gamma': dico['GaussSeidel']['gamma'],
            'alpha': dico['GaussSeidel']['beta']}
        if useCoarse:
            dico['CoarseGaussSeidel'] = {
                'gamma': norm(I - TCF @ solve(ImQDeltaTilde, TFC) @ ImQ),
                'beta': norm(TCF @ solve(ImQDeltaTilde, HTilde) @ TFC)}
            dico['TwoGridExact'] = {
                'gamma': norm(I - TCF @ solve(ImQTilde, TFC) @ ImQ),
                'beta': norm(TCF @ solve(ImQTilde, HTilde) @ TFC)}
            dico['STMG'] = {
                'alpha': norm(
                    (I - TCF @ solve(ImQTilde, TFC) @ ImQ) @ solve(ImQ, H)),
                'beta': norm(TCF @ solve(ImQTilde, HTilde) @ TFC)}
            dico['Parareal'] = {
                'alpha': norm(
                    solve(ImQ, H) - TCF @ solve(ImQDeltaTilde, TFC) @ H),
                'beta': norm(TCF @ solve(ImQDeltaTilde, TFC) @ H)}
            dico['PFASST'] = {
                'alpha': norm(
                    (I - TCF @ solve(ImQDeltaTilde, TFC) @ ImQ) @
                    solve(ImQDelta, H)),
                'beta': norm(
                    TCF @ solve(ImQDeltaTilde, HTilde) @ TFC),
                'gamma': norm(
                    (I - TCF @ solve(ImQDeltaTilde, TFC) @ ImQ) @
                    solve(ImQDelta, Q-QDelta) * abs(lam*dt))}
            dico['TFAST'] = {
                'alpha': norm(
                    (I - TCF @ solve(ImQTilde, TFC) @ ImQ) @
                    solve(ImQDelta, H)),
                'beta': norm(
                    TCF @ solve(ImQTilde, HTilde) @ TFC),
                'gamma': norm(
                    (I - TCF @ solve(ImQTilde, TFC) @ ImQ) @
                    solve(ImQDelta, Q-QDelta) * abs(lam*dt))}
        self.gfmCoeff = dico

        # Dictionnary with error bound methods
        self.errBound = {
            'SDC': lambda n, k:
                gfma.sdc(n, k, **self.gfmCoeff['SDC']) * self.delta,
            'GaussSeidel': lambda n, k:
                gfma.pbi(n, k, **self.gfmCoeff['GaussSeidel']) * self.delta,
            'Jacobi': lambda n, k:
                gfma.pbi(n, k, **self.gfmCoeff['Jacobi']) * self.delta,
            'STMG': lambda n, k:
                gfma.pbi(n, k, **self.gfmCoeff['STMG']) * self.delta,
            'Parareal': lambda n, k:
                gfma.pbi(n, k, **self.gfmCoeff['Parareal']) * self.delta,
            'PFASST': lambda n, k:
                gfma.pbi(n, k, **self.gfmCoeff['PFASST']) * self.delta,
            'TFAST': lambda n, k:
                gfma.pbi(n, k, **self.gfmCoeff['TFAST']) * self.delta
            }

        # Variables
        numType = np.array(1.*lam*u0).dtype
        f = np.zeros((L, M), dtype=numType)
        f[0] = u0
        u = np.zeros((L, M), dtype=numType)
        t = np.array([t + dt*nodes for t in times[:-1]])

        # Store attributes
        self.times, self.dt, self.t = times, dt, t
        self.L, self.M, self.nodes = L, M, nodes
        self.Q, self.QDelta, self.H = Q, QDelta, H
        self.ImQDelta, self.ImQ = ImQDelta, ImQ
        if useCoarse:
            self.MTilde, self.nodesTilde = MTilde, nodesTilde
            self.QTilde, self.QDeltaTilde = QTilde, QDeltaTilde
            self.ImQDeltaTilde, self.ImQTilde = ImQDeltaTilde, ImQTilde
            self.HTilde, self.TFC, self.TCF = HTilde, TFC, TCF
            self.HDelta = HDelta
        self.f, self.u, self.t = f, u, t
        self.initCond, self.u0 = initCond, u0
        self.nN1, self.nN2, self.gamma = nN1, nN2, gamma
        self.N1, self.N2 = N1, N2

        # Empty pointers
        self._uExact, self._uCoarse, self._delta = None, None, None

        # Simulate a dictionnary with all run methods
        class dicoRun(object):
            def __getitem__(dico, algo):
                return getattr(self, 'run'+algo)
        self.run = dicoRun()

    # Standard L_inf norm
    def normMat_L_INF(self, A):
        return np.linalg.norm(A, np.inf)

    normVec_L_INF = normMat_L_INF

    # Standard L_2 norm
    def normMat_L_2(self, A):
        return np.linalg.norm(A, 2)

    normVec_L_2 = normMat_L_2

    # Standard L_1 norm
    def normMat_L_1(self, A):
        return np.linalg.norm(A, 1)

    normVec_L_1 = normMat_L_1

    # Custom norm
    def normMat_CUSTOM(self, A):
        return np.linalg.norm(self.Vinv.dot(A).dot(self.V), ord=np.inf)

    def normVec_CUSTOM(self, u):
        return np.linalg.norm(self.Vinv.dot(u), ord=np.inf)

    def setInitCond(self, u):
        if self.initCond == 'ZERO':
            u[:] = 0
        elif self.initCond == 'U0':
            u[:] = self.u0
        elif self.initCond == 'RAND':
            np.random.seed(1990)
            u[:] = np.random.rand(*u.shape)
            if u.dtype == complex:
                u[:] += np.random.rand(*u.shape)*1j
        else:
            raise ValueError(f'wrong initial condition {self.initCond}')

    def sweep(self, u, uStar, f, exactImQ=False, relax=1):
        # Matrix to invert
        ImQ = self.ImQ if exactImQ else self.ImQDelta
        # Right hand side
        rhs = self.H @ uStar + f - self.ImQ @ u
        # Linear solve
        u += np.linalg.solve(relax*ImQ, rhs)

    def sweepCoarse(self, u, uStar, uDelta, f, exactImQ=False):
        # Matrix to invert
        ImQ = self.ImQTilde if exactImQ else self.ImQDeltaTilde
        # Right hand side
        rhs = self.TFC @ (self.H @ uStar + f) + self.HTilde @ self.TFC @ uDelta \
            - self.TFC @ self.ImQ @ u
        u += self.TCF @ np.linalg.solve(ImQ, rhs)

    def propagate(self, u, exactImQ=True, coarsening=False):
        if coarsening:
            ImQ = self.ImQTilde if exactImQ else self.ImQDeltaTilde
            return self.TCF @ \
                np.linalg.solve(ImQ, self.HTilde @ self.TFC @ u)
        else:
            ImQ = self.ImQ if exactImQ else self.ImQDelta
            return np.linalg.solve(ImQ, self.H @ u)

    def residuum(self, u):
        r = self.ImQ.dot(u.T).T
        r[1:] -= self.H.dot(u[:-1].T).T
        r -= self.f
        return r

    def exactSolution(self):
        # Solution of the collocation problem
        u = self.u.copy()
        self.setInitCond(u)
        uStar = u[0]*0
        for l in range(self.L):
            u[l] += np.linalg.solve(
                self.ImQ, self.H.dot(uStar) + self.f[l]-self.ImQ.dot(u[l]))
            uStar = u[l]
        return u

    def coarseSolution(self):
        # Solution using the coarse solver
        u = self.u.copy()
        self.setInitCond(u)
        ut = self.f[0]
        for l in range(self.L):
            u[l] = self.propagate(ut, exactImQ=True, coarsening=True)
            ut = u[l]
        return u

    @property
    def uExact(self):
        if self._uExact is None:
            self._uExact = self.exactSolution()
        return self._uExact

    @property
    def uCoarse(self):
        if self._uCoarse is None:
            self._uCoarse = self.coarseSolution()
        return self._uCoarse

    # ------------------------------------------------------------------------
    # Run methods for standalone block iterations
    # ------------------------------------------------------------------------

    def uInit(self, u):
        """Default initialization method for block variables stored in u"""
        if u is None:
            u = self.u.copy()
            self.setInitCond(u)
        return u

    class _Decorators(object):
        RUNDOC = """
        Parameters
        ----------
        k : int
            Number of iterations to perform on each block.
        u : np.array(L, M), optional
            Global solution to update. If not given, then a new variable is
            created and initialized using the setInitCond method.
        **kwargs :
            Additional arguments.

        Returns
        -------
        u : np.array(L, M)
            Final solution for the M nodes of the L blocks.
        """
        @classmethod
        def addRunDoc(cls, func):
            doc = getattr(func, '__doc__')
            doc = '\n        TODO : DOC\n        ' if doc is None else doc
            setattr(func, '__doc__', doc + cls.RUNDOC)
            return func

    @_Decorators.addRunDoc
    def runSDC(self, k, u=None, **kwargs):
        """
        Run k iterations of sequential SDC on each block
        """
        # Eventual initialization
        u = self.uInit(u)
        # Block iteration
        uStar = u[0]*0
        for l in range(self.L):
            # All sweeps for one interval
            for n in range(k):
                self.sweep(u[l], uStar, self.f[l])
            uStar = u[l]
        return u

    @_Decorators.addRunDoc
    def runGaussSeidel(self, k, u=None, **kwargs):
        """
        Run k iterations of Block Gauss-Seidel SDC on each block
        """
        # Eventual initialization
        u = self.uInit(u)
        # Block iteration
        for n in range(k):
            uStar = u[0]*0
            for l in range(self.L):
                self.sweep(u[l], uStar, self.f[l])
                uStar = u[l]
        return u

    @_Decorators.addRunDoc
    def runCoarseGaussSeidel(self, k, u=None, **kwargs):
        """
        Run k iterations of two level Block Gauss-Seidel SDC on each block
        """
        # Eventual initialization
        u = self.uInit(u)
        # Block iteration
        for n in range(k):
            uStar = u[0]*0
            uDelta = u[0]*0
            for l in range(self.L):
                uStarPrev = u[l].copy()
                self.sweepCoarse(u[l], uStar, uDelta, self.f[l])
                uStar = uStarPrev
                uDelta = u[l]-uStarPrev
        return u

    @_Decorators.addRunDoc
    def runJacobi(self, k, u=None, **kwargs):
        # Eventual initialization
        u = self.uInit(u)
        # Block iteration
        for n in range(k):
            uStar = u[0]*0
            for l in range(self.L):
                uStarPrev = u[l].copy()
                self.sweep(u[l], uStar, self.f[l])
                uStar = uStarPrev
        return u

    @_Decorators.addRunDoc
    def runCoarseJacobi(self, k, u=None, **kwargs):
        # Eventual initialization
        u = self.uInit(u)
        # Block iteration
        for n in range(k):
            uStar = u[0]*0
            uDelta = u[0]*0
            for l in range(self.L):
                uStarPrev = u[l].copy()
                self.sweepCoarse(u[l], uStar, uDelta, self.f[l])
                uStar = uStarPrev
        return u

    @_Decorators.addRunDoc
    def runJacobiExact(self, k, u=None, omega=1, **kwargs):
        # Eventual initialization
        u = self.uInit(u)
        # Block iteration
        for n in range(k):
            uStar = u[0]*0
            for l in range(self.L):
                uStarPrev = u[l].copy()
                self.sweep(u[l], uStar, self.f[l], exactImQ=True, relax=omega)
                uStar = uStarPrev
        return u

    @_Decorators.addRunDoc
    def runTwoGridExact(self, k, u=None, **kwargs):
        # Eventual initialization
        u = self.uInit(u)
        # Block iteration
        for n in range(k):
            uStar = u[0]*0
            uDelta = u[0]*0
            for l in range(self.L):
                uStarPrev = u[l].copy()
                self.sweepCoarse(u[l], uStar, uDelta, self.f[l], exactImQ=True)
                uStar = uStarPrev
                uDelta = u[l]-uStarPrev
        return u

    @_Decorators.addRunDoc
    def runTwoGridJacobi(self, k, u=None, **kwargs):
        # Eventual initialization
        u = self.uInit(u)
        # Block iteration
        for n in range(k):
            uStar = u[0]*0
            uDelta = u[0]*0
            for l in range(self.L):
                uStarPrev = u[l].copy()
                self.sweepCoarse(u[l], uStar, uDelta, self.f[l], exactImQ=True)
                uStar = uStarPrev
        return u

    # ------------------------------------------------------------------------
    # Run methods for iterative PinT algorithms
    # ------------------------------------------------------------------------

    @_Decorators.addRunDoc
    def runPFASST(self, k, u=None, ideal=False, **kwargs):
        # Eventual initialization
        u = self.uInit(u)
        # Block iteration
        for n in range(k):
            self.runJacobi(1, u)
            if ideal:
                self.runGaussSeidel(1, u)
            else:
                self.runCoarseGaussSeidel(1, u)
        return u

    @_Decorators.addRunDoc
    def runSTMG(self, k, u=None, nPreRelax=1, nPostRelax=0, omega=1, **kwargs):
        # Eventual initialization
        u = self.uInit(u)
        # Block iteration
        for n in range(k):
            self.runJacobiExact(nPreRelax, u, omega)
            self.runTwoGridExact(1, u)
            self.runJacobiExact(nPostRelax, u, omega)
        return u

    @_Decorators.addRunDoc
    def runTFAST(self, k, u=None, nRelax=1, **kwargs):
        # Eventual initialization
        u = self.uInit(u)
        # Block iteration
        for n in range(k):
            self.runJacobi(nRelax, u)
            self.runTwoGridExact(1, u)
        return u

    @_Decorators.addRunDoc
    def runParareal(self, k, u=None,
                    twoLevelCoarse=True, sdcCoarse=True, **kwargs):
        # Eventual initialization
        u = self.uInit(u)
        # Coarse and fine progagation settings
        coarse = dict(exactImQ=not sdcCoarse, coarsening=twoLevelCoarse)
        fine = dict(exactImQ=True, coarsening=False)
        # Block iteration
        for n in range(k):
            uk = self.f[0]
            ukp1 = self.f[0]
            for l in range(self.L):
                uGk = self.propagate(uk, **coarse)
                uFk = self.propagate(uk, **fine)
                uGkp1 = self.propagate(ukp1, **coarse)
                uk = u[l].copy()
                u[l] = uFk + uGkp1 - uGk
                ukp1 = u[l]
        return u

    @_Decorators.addRunDoc
    def runMGRIT(self, k, u=None,
                 twoLevelCoarse=True, sdcCoarse=True, **kwargs):
        # Eventual initialization
        u = self.uInit(u)
        # Coarse and fine progagation settings
        coarse = dict(exactImQ=not sdcCoarse, coarsening=twoLevelCoarse)
        fine = dict(exactImQ=True, coarsening=False)
        # Block iteration
        for n in range(k):
            uk = self.f[0]
            ukPrev = self.f[0]
            u[0] = self.propagate(self.f[0], **fine)
            uk = u[0]
            ukp1 = u[0]
            for l in range(1, self.L):
                uGk = self.propagate(self.propagate(ukPrev, **fine), **coarse)
                uFk = self.propagate(self.propagate(ukPrev, **fine), **fine)
                uGkp1 = self.propagate(ukp1, **coarse)
                ukPrev = uk
                uk = u[l].copy()
                u[l] = uFk + uGkp1 - uGk
                ukp1 = u[l]
        return u

    # ------------------------------------------------------------------------
    # Error bounds methods
    # ------------------------------------------------------------------------

    @property
    def delta(self):
        if self._delta is None:
            uExact = self.uExact
            uk0 = self.u.copy()
            self.setInitCond(uk0)
            norm = self.normVec
            self._delta = max([norm(uk0[l]-uExact[l]) for l in range(self.L)])
        return self._delta
