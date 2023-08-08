# lambda = 1j
# dt = pi /10
# t0 = 0
# t1 = pi
# 4 Gauss-Rada-Right
# start initial guess with zeros
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from pycode.qmatrix import genCollocation

# change these:
############################
M = 4
quadType = 'RADAU-RIGHT'
distr = 'LEGENDRE'
T = [0, np.pi]
lamb = 1j
u_initial = 1
############################

# returns the MIN-NS matrix
def MIN_NS(t):
    return np.diag(t) / M

def BEPAR(t):
    return np.diag(t)

def SDC_sweep(u0, D):
    Prec = np.eye(M) - dt * lamb * D
    rhs = dt * lamb * (Q - D) @ u0 + u_initial

    u = sp.linalg.solve(Prec, rhs)
    res = (np.eye(M) - dt * lamb * Q) @ u - u_initial

    return u, np.linalg.norm(res, np.inf)

def SDC_sweep_refinement(u0, D):
    Prec = np.eye(M) - dt * lamb * D
    res = (np.eye(M) - dt * lamb * Q) @ u0 - u_initial

    c = sp.linalg.solve(Prec, res)
    u = u0 - c

    res = (np.eye(M) - dt * lamb * Q) @ u - u_initial

    return u, np.linalg.norm(res, np.inf)

def do_sweeps(prec, sweeper):
    u = np.zeros(M)
    residuals = [np.linalg.norm((np.eye(M) - dt * lamb * Q) @ u - u_initial, np.inf)]
    for s in range(len(prec)):
        if prec[s] == 'BEPAR':
            u, res = sweeper(u, BEPAR(nodes))
        else:
            u, res = sweeper(u, MIN_NS(nodes))
        residuals.append(res)
    return residuals


dt = T[1] / 10
nodes, _, Q = genCollocation(M, distr, quadType)

res_MIN_NS = do_sweeps(['MIN-NS'] * 10, SDC_sweep)
res_BEPAR = do_sweeps(['BEPAR'] * 10, SDC_sweep)
res_MIX = do_sweeps(['BEPAR'] * 5 + ['MIN-NS'] * 5, SDC_sweep)

res_MIN_NS_r = do_sweeps(['MIN-NS'] * 10, SDC_sweep_refinement)
res_BEPAR_r = do_sweeps(['BEPAR'] * 10, SDC_sweep_refinement)
res_MIX_r = do_sweeps(['BEPAR'] * 5 + ['MIN-NS'] * 5, SDC_sweep_refinement)

plt.semilogy(res_MIN_NS, 'o--')
plt.semilogy(res_BEPAR, 'x--')
plt.semilogy(res_MIX, '<--')

plt.semilogy(res_MIN_NS_r, 'o:')
plt.semilogy(res_BEPAR_r, 'x:')
plt.semilogy(res_MIX_r, '<:')

plt.legend(['MIN-NS', 'BEPAR', '5 BEPAR + 5 MIN-RS', 'MIN-NS + ref', 'BEPAR + ref', '5 BEPAR + 5 MIN-RS + ref'])

plt.show()




























