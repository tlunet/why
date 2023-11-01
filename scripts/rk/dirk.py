#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 00:59:13 2023

@author: telu
"""
import numpy as np
from butcher import RKAnalysis

PADE_COEFFS = {
    1: {
        "P": [ 1/2, 1],
        "Q": [-1/2, 1]
        },
    2: {
        "P": [1/12,  1/2, 1],
        "Q": [1/12, -1/2, 1]
        },
    3: {
        "P": [ 1/120, 1/10,  1/2, 1],
        "Q": [-1/120, 1/10, -1/2, 1]
        },
    4: {
        "P": [1/1680,  1/84, 3/28,  1/2, 1],
        "Q": [1/1680, -1/84, 3/28, -1/2, 1]
        },
    }


def genButcher(order):
    coeffs = PADE_COEFFS[order]
    p = np.poly1d(coeffs["P"])
    q = np.poly1d(coeffs["Q"])

    gamma = q.roots

    A = np.diag(gamma)
    b = gamma*p(1/gamma)/q(1/gamma)
    c = gamma

    return A, b, c


dirk = RKAnalysis(*genButcher(1))
print("A =", dirk.A)
print("b =", dirk.b)
print("c =", dirk.c)

dirk.plotNumericalOrder(1j-0.1)
