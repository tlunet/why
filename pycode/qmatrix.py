#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
from scipy.linalg import lu

try:
    # Relative import (when used as a package module)
    from .nodes import NodesGenerator
    from .lagrange import LagrangeApproximation
    from .coeffs import OPT_COEFFS, WEIRD_COEFFS
except ImportError:
    # Absolute import (when used as script)
    from nodes import NodesGenerator
    from lagrange import LagrangeApproximation
    from coeffs import OPT_COEFFS, WEIRD_COEFFS

# To avoid recomputing QDelta coefficients with MIN-SR-S
STORAGE = {}

def genQDelta(nodes, sweepType, Q, lambdaI=1, lambdaE=1):
    """
    Generate QDelta matrix for a given node distribution

    Parameters
    ----------
    nodes : array (M,)
        quadrature nodes, scaled to [0, 1]
    sweepType : str
        Type of sweep, that defines QDelta. Can be selected from :

        - BE : Backward Euler sweep (first order)
        - FE : Forward Euler sweep (first order)
        - LU : uses the LU trick
        - TRAP : sweep based on Trapezoidal rule (second order)
        - EXACT : don't bother and just use Q
        - PIC : Picard iteration => zeros coefficient
        - OPT-[...] : Diagonaly precomputed coefficients, for which one has to
          provide different parameters. For instance, [...]='QmQd-2' uses the
          diagonal coefficients using the optimization method QmQd with the index 2
          solution (index starts at 0 !). Quadtype and number of nodes are
          determined automatically from the Q matrix.
        - WEIRD-[...] : diagonal coefficient allowing A-stability with collocation
          update (forceProl=True).
        - DNODES-[...] : nodes divided by a given coefficient. If none is given,
          then divide by M. Note : DNODES-1 corresponds to BEPAR, and DNODES
          correspond to the diagonal matrix that minimizes the spectral radius
          of Q-QDelta.
        - MIN3 : the magical diagonal coefficients, if they exists for this
          configuration

    Q : array (M,M)
        Q matrix associated to the node distribution
        (used only when sweepType in [LU, EXACT, OPT-[...], WEIRD]).

    Returns
    -------
    QDelta : array (M,M)
        The generated QDelta matrix.
    dtau : float
        Correction coefficient for time integration with QDelta
    """
    # Generate deltas
    deltas = np.copy(nodes)
    deltas[1:] = np.ediff1d(nodes)

    # Extract informations from Q matrix
    M = deltas.size
    leftIsNode = np.allclose(Q[0], 0)
    rightIsNode = np.isclose(Q[-1].sum(), 1)
    quadType = 'LOBATTO' if (leftIsNode and rightIsNode) else \
        'RADAU-LEFT' if leftIsNode else \
        'RADAU-RIGHT' if rightIsNode else \
        'GAUSS'

    distr = 'EQUID' if np.allclose(deltas[1:], deltas[1]) else 'LEGENDRE'

    # Compute QDelta
    QDelta = np.zeros((M, M))
    dtau = 0.0
    if sweepType in ['BE', 'FE']:
        offset = 1 if sweepType == 'FE' else 0
        for i in range(offset, M):
            QDelta[i:, :M-i] += np.diag(deltas[offset:M-i+offset])
        if sweepType == 'FE':
            dtau = deltas[0]
    elif sweepType == 'TRAP':
        for i in range(0, M):
            QDelta[i:, :M-i] += np.diag(deltas[:M-i])
        for i in range(1, M):
            QDelta[i:, :M-i] += np.diag(deltas[1:M-i+1])
        QDelta /= 2.0
        dtau = nodes[0]/2.0
    elif sweepType == 'LU':
        QT = Q.T.copy()
        [_, _, U] = lu(QT, overwrite_a=True)
        QDelta = U.T
    elif sweepType == 'EXACT':
        QDelta = np.copy(Q)
    elif sweepType.startswith("PIC"):
        QDelta = np.zeros(Q.shape)
        factor = sweepType.split('-')[-1]
        if factor == 'PIC':
            factor = 0.0
        else:
            try:
                factor = float(factor)
            except (ValueError, TypeError):
                raise ValueError(f"DNODES does not accept {factor} as parameter")
        dtau = factor
    elif sweepType.startswith('OPT'):
        try:
            oType, idx = sweepType[4:].split('-')
        except ValueError:
            raise ValueError(f'missing parameter(s) in sweepType={sweepType}')
        M, idx = int(M), int(idx)
        try:
            coeffs = OPT_COEFFS[oType][M][quadType][idx]
            QDelta[:] = np.diag(coeffs)
        except (KeyError, IndexError):
            raise ValueError('no OPT diagonal coefficients for '
                             f'{oType}-{M}-{quadType}-{idx}')
    elif sweepType == 'BEPAR':
        QDelta[:] = np.diag(nodes)

    elif sweepType == 'TRAPAR':
        QDelta[:] = np.diag(nodes/2)
        dtau = nodes/2

    elif sweepType == "IMEX-NS":
        QDelta[:] = np.absolute(lambdaE + lambdaI) / np.absolute(lambdaI) * np.diag(nodes/M)

    elif sweepType.startswith('IMEX'):
        factor = sweepType.split('-')[-1]
        try:
            factor = float(factor)
        except (ValueError, TypeError):
            raise ValueError(f"IMEX doesn't accept {factor} as parameter")
        QDelta[:] = np.absolute(lambdaE + lambdaI) / np.absolute(lambdaI) * np.diag(nodes) / factor

    elif sweepType.startswith('THETAPAR-'):
        theta = float(sweepType.split('-')[-1])
        QDelta[:] = theta*np.diag(nodes)
        dtau = (1-theta)*nodes

    elif sweepType == 'WEIRD':

        try:
            coeffs = WEIRD_COEFFS[quadType][M]
            QDelta[:] = np.diag(coeffs)
        except (KeyError, IndexError):
            raise ValueError('no WEIRD diagonal coefficients for '
                             f'{M}-{quadType} nodes')

    elif sweepType.startswith('DNODES'):
        factor = sweepType.split('-')[-1]
        if factor == 'DNODES':
            factor = M
        else:
            try:
                factor = float(factor)
            except (ValueError, TypeError):
                raise ValueError(f"DNODES don't accept {factor} as parameter")
        QDelta[:] = np.diag(nodes/factor)

    elif sweepType == "MIN-SR-NS":

        QDelta[:] = np.diag(nodes/M)

    elif sweepType == "MIN-SR-S":

        idString = f'{M}-{quadType}-{distr}'
        if idString in STORAGE:
            # Coefficients have already been computed
            coeffs = STORAGE[idString]
        else:
            # Compute coefficients using the determinant (numerical approach)
            nCoeffs = M
            if quadType in ['LOBATTO', 'RADAU-LEFT']:
                nCoeffs -= 1;
                Q = Q[1:, 1:]
                nodes = nodes[1:]

            def func(coeffs):
                coeffs = np.asarray(coeffs)
                kMats = [(1-z)*np.eye(nCoeffs) + z*np.diag(1/coeffs) @ Q
                          for z in nodes]
                vals = [np.linalg.det(K)-1 for K in kMats]
                return np.array(vals)

            coeffs = sp.optimize.fsolve(func, nodes/M, xtol=1e-14)

            # def func(coeffs):
            #     coeffs = np.asarray(coeffs)
            #     K = np.eye(nCoeffs) - np.linalg.solve(np.diag(coeffs), Q)
            #     return np.linalg.norm(np.linalg.matrix_power(K, nCoeffs))

            # coeffs = sp.optimize.minimize(func, coeffs, method="nelder-mead")
            # coeffs = coeffs.x

            if quadType in ['LOBATTO', 'RADAU-LEFT']:
                coeffs = [0] + list(coeffs)

            STORAGE[idString] = coeffs

        QDelta[:] = np.diag(coeffs)

    elif sweepType == 'MIN3':

        try:
            coeffs = OPT_COEFFS['MIN3'][distr][quadType][M]
        except KeyError:
            raise ValueError('no MIN3 diagonal coefficients for '
                             f'{distr}-{quadType}-{M}')
        QDelta[:] = np.diag(coeffs)

    else:
        raise NotImplementedError(f'sweepType={sweepType}')
    return QDelta, dtau


def genCollocation(M, distr, quadType):
    """
    Generate the nodes, weights and Q matrix for a given collocation method

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
    - RADAU-LEFT : include left boundary points in the nodes
    - RADAU-RIGHT : include right boundary points in the nodes
    - LOBATTO : include both boundary points in the nodes

    Returns
    -------
    nodes : array (M,)
        quadrature nodes, scaled to [0, 1]
    weights : array (M,)
        quadrature weights associated to the nodes
    Q : array (M,M)
        normalized Q matrix of the collocation problem
    """

    # Generate nodes between [0, 1]
    nodes = NodesGenerator(node_type=distr, quad_type=quadType).getNodes(M)
    nodes += 1
    nodes /= 2
    # np.round(nodes, 14, out=nodes)

    # Compute Q and weights
    approx = LagrangeApproximation(nodes)
    Q = approx.getIntegrationMatrix([(0, tau) for tau in nodes])
    weights = approx.getIntegrationMatrix([(0, 1)]).ravel()

    return nodes, weights, Q


def genQMatrices(M, distr, quadType, sweepType):
    nodes, weights, Q = genCollocation(M, distr, quadType)
    QDelta, _ = genQDelta(nodes, sweepType, Q)
    return {
        'nodes': nodes,
        'weights': weights,
        'Q': Q,
        'QDelta': QDelta
    }


def getIterMatrixSDC(M, distr, quadType, sweepType, nSweep, lamDt):
    nodes, _, Q = genCollocation(M, distr, quadType)

    R = np.eye(M)
    for k in range(nSweep):

        # Determine sweepType
        if not isinstance(sweepType, str):
            # List of sweeps
            try:
                # Takes k'st sweepType in list
                sType = sweepType[k]
            except KeyError:
                # Take last sweepType in list
                sType = sweepType[-1]
        else:
            # Only one sweepType given
            sType = sweepType

        # Compute QDelta matrix
        QDelta = genQDelta(nodes, sType, Q)[0]

        # Multiply iteration matrix for each sweeps
        ImQd = (np.eye(M)-lamDt*QDelta)
        R = np.linalg.solve(ImQd, Q-QDelta) @ R
        R *= lamDt

    return R
