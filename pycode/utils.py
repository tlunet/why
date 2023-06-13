#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions
"""
import numpy as np


def getQuadType(Q):
    """
    Retrieve the quadrature type from the coefficients of the Q matrix

    Parameters
    ----------
    Q : np.2darray
        The full Q matrix.

    Returns
    -------
    quadType : str
        The quadrature type ('LOBATTO', 'GAUSS', 'RADAU-RIGHT' or 'RADAU-LEFT')

    """
    leftIsNode = np.allclose(Q[0], 0)
    rightIsNode = np.isclose(Q[-1].sum(), 1)
    
    quadType = 'LOBATTO' if (leftIsNode and rightIsNode) else \
        'RADAU-LEFT' if leftIsNode else \
        'RADAU-RIGHT' if rightIsNode else \
        'GAUSS'
        
    return quadType

