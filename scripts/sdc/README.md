# SDC scripts

To run some experiment with SDC and the diagonal SDC matrix obtained looking a the strange facts ...

It is based on an implementation of generic IMEX SDC for the Dahlquist problem (where you can define a `lambdaI` for
_implicitly solve lambda_ and `lambdaE` for _explicitly solve lambda_), that are implemented in [pycode.core](../../pycode/core.py) and [pycode.dahlquist](../../pycode/dahlquist.py) submodules.

- [convergenceOrder.py](./convergenceOrder.py) : looks at the global convergence order on $[0, 2\pi]$ for the implicit part only and the three first sweeps.
- [stability.py](./stability.py) : looks a stability contours for the first sweeps on the complex plan, for the implicit part only.
- [stabilityFWSW.py](./stabilityFWSW.py) : looks at the Fast-Wave Slow-Wave stability (see article of Daniel and Robert) for purely imaginary values for `lambdaI` and `lambdaE`.
- [residualsOverTime.py](./residualsOverTime.py) : looks at the residual over time for the first three sweeps and a given number of time steps.
- [residualsOverSweeps.py](./residualsOverSweeps.py) : looks at the residual in function of the number of sweeps, for a given simulation time and after a given number of time steps.
