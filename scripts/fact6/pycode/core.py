#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 09:25:32 2022

@author: cpf5546
"""
import numpy as np

try:
    # Relative import (when used as a package module)
    from .qmatrix import genCollocation, genQDelta
except ImportError:
    # Absolute import (when used as script)
    from qmatrix import genCollocation, genQDelta

# -----------------------------------------------------------------------------
# Main SDC parameters
# -----------------------------------------------------------------------------
DEFAULT = {
    'nNodes': 3,
    'nSweep': 2,
    'nodeDistr': 'LEGENDRE',
    'quadType': 'RADAU-RIGHT',
    'implSweep': 'BE',
    'explSweep': 'FE',
    'initSweep': 'COPY',
    'forceProl': False,  # Should not change this !
    }

PARAMS = {
    ('nNodes', '-sM', '--sdcNNodes'):
        dict(help='number of nodes for SDC', type=int,
             default=DEFAULT['nNodes']),
    ('quadType', '-sqt', '--sdcQuadType'):
        dict(help='quadrature type for SDC',
             default=DEFAULT['quadType']),
    ('nodeDistr', '-snd', '--sdcNodeDistr'):
        dict(help='node distribution for SDC',
             default=DEFAULT['nodeDistr']),
    ('nSweep', '-sK', '--sdcNSweep'):
        dict(help='number of sweeps for SDC', type=int,
             default=DEFAULT['nSweep']),
    ('initSweep', '-si', '--sdcInitSweep'):
        dict(help='initial sweep to get initialized nodes values',
             default=DEFAULT['initSweep']),
    ('implSweep', '-sis', '--sdcImplSweep'):
        dict(help='type of QDelta matrix for implicit sweep',
             default=DEFAULT['implSweep']),
    ('explSweep', '-ses', '--sdcExplSweep'):
        dict(help='type of QDelta matrix for explicit sweep',
             default=DEFAULT['explSweep']),
    ('forceProl', '-sfp', '--sdcForceProl'):
        dict(help='if specified force the prolongation stage '
             '(ignored for quadType=GAUSS or RADAU-LEFT)',
             action='store_true')
    }

# Printing function
def sdcInfos(nNodes, quadType, nodeDistr, nSweep,
             implSweep, explSweep, initSweep, forceProl,
             **kwargs):
    return f"""
-- nNodes : {nNodes}
-- quadType : {quadType}
-- nodeDistr : {nodeDistr}
-- nSweep : {nSweep}
-- implSweep : {implSweep}
-- explSweep : {explSweep}
-- initSweep : {initSweep}
-- forceProl : {forceProl}
""".strip()

# -----------------------------------------------------------------------------
# Base class implementation
# -----------------------------------------------------------------------------
class IMEXSDCCore(object):

    # Initialize parameters with default values
    nSweep = DEFAULT['nSweep']
    nodeDistr = DEFAULT['nodeDistr']
    quadType = DEFAULT['quadType']
    implSweep = DEFAULT['implSweep']
    explSweep = DEFAULT['explSweep']
    initSweep = DEFAULT['initSweep']
    forceProl = DEFAULT['forceProl']

    # Collocation method attributes
    nodes, weights, Q = genCollocation(DEFAULT['nNodes'], nodeDistr, quadType)
    # IMEX SDC attributes
    QDeltaI, dtauI = genQDelta(nodes, implSweep, Q)
    QDeltaE, dtauE = genQDelta(nodes, explSweep, Q)

    # Default attributes, used later as instance attributes
    # => should be defined in inherited class
    dt = None
    axpy = None

    @classmethod
    def setParameters(cls, M=None, nodeDistr=None, quadType=None,
                      implSweep=None, explSweep=None, initSweep=None,
                      nSweep=None, forceProl=None, calcResidual=None,
                      logResidual=None, logResIter=None):

        # Get non-changing parameters
        keepM = M is None
        keepNodeDistr = nodeDistr is None
        keepQuadType = quadType is None
        keepImplSweep = implSweep is None
        keepExplSweep = explSweep is None

        # Set parameter values
        M = len(cls.nodes) if keepM else M
        nodeDistr = cls.nodeDistr if keepNodeDistr else nodeDistr
        quadType = cls.quadType if keepQuadType else quadType
        implSweep = cls.implSweep if keepImplSweep else implSweep
        explSweep = cls.explSweep if keepExplSweep else explSweep

        # Determine updated parts
        updateNode = (not keepM) or (not keepNodeDistr) or (not keepQuadType)
        updateQDeltaI = (not keepImplSweep) or updateNode
        updateQDeltaE = (not keepExplSweep) or updateNode

        # Update parameters
        if updateNode:
            cls.nodes, cls.weights, cls.Q = genCollocation(
                M, nodeDistr, quadType)
            cls.nodeDistr, cls.quadType = nodeDistr, quadType
        if updateQDeltaI:
            iSweepName = implSweep
            if not isinstance(implSweep, str):
                iSweepName = implSweep[0]
            cls.QDeltaI, cls.dtauI = genQDelta(cls.nodes, iSweepName, cls.Q)
            cls.implSweep = implSweep
        if updateQDeltaE:
            eSweepName = explSweep
            if not isinstance(explSweep, str):
                eSweepName = explSweep[0]
            cls.QDeltaE, cls.dtauE = genQDelta(cls.nodes, eSweepName, cls.Q)
            cls.explSweep = explSweep

        # Eventually update additional parameters
        for par in ['initSweep', 'nSweep', 'forceProl',
                    'calcResidual', 'logResidual', 'logResIter']:
            val = eval(par)
            if val is not None:
                setattr(cls, par, val)

    # -------------------------------------------------------------------------
    # Class properties
    # -------------------------------------------------------------------------
    @classmethod
    def getMaxOrder(cls):
        # TODO : adapt to non-LEGENDRE node distributions
        M = len(cls.nodes)
        return 2*M if cls.quadType == 'GAUSS' else \
            2*M-1 if cls.quadType.startswith('RADAU') else \
            2*M-2  # LOBATTO

    @classmethod
    def getInfos(cls):
        return sdcInfos(
            len(cls.nodes), cls.quadType, cls.nodeDistr, cls.nSweep,
            cls.implSweep, cls.explSweep, cls.initSweep, cls.forceProl)

    @classmethod
    def getM(cls):
        return len(cls.nodes)

    @classmethod
    def rightIsNode(cls):
        return np.isclose(cls.nodes[-1], 1.0)

    @classmethod
    def leftIsNode(cls):
        return np.isclose(cls.nodes[0], 0.0)

    @classmethod
    def doProlongation(cls):
        return not cls.rightIsNode() or cls.forceProl

    @classmethod
    def _setSweep(cls, k):
        if cls.initSweep == 'QDelta':
            k += 1
        # Eventually change implicit QDelta during sweeps
        if not isinstance(cls.implSweep, str):
            try:
                iSweepName = cls.implSweep[k]
            except IndexError:
                iSweepName = cls.implSweep[-1]
            cls.QDeltaI = genQDelta(cls.nodes, iSweepName, cls.Q)[0]
        # Eventually change explicit QDelta during sweeps
        if not isinstance(cls.explSweep, str):
            try:
                eSweepName = cls.explSweep[k]
            except IndexError:
                eSweepName = cls.explSweep[-1]
            cls.QDeltaE = genQDelta(cls.nodes, eSweepName, cls.Q)[0]



if __name__ == '__main__':
    sdc = IMEXSDCCore()
