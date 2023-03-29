# Fact n°3

> When minimizing the spectral radius of $Q-Q_{\Delta}(x)$, with $Q_{\Delta}(x)$ a diagonal matrix with $x$ on the diagonal, for any local minimum $x$ we have
> $$
> \sum x_i = \frac{N_{coeff}}{order},
> $$
> where $N_{coeff}$ is the number of non-zero coefficients of $x$, that is
>
> - $N_{coeff} = M$ for GAUSS and RADAU-RIGHT points 
> - $N_{coeff} = M-1$ for LOBATTO and RADAU-LEFT points
> 
> and $order$ is the accuracy order of underlying quadrature scheme, that is
>
> - $order = 2M$ for GAUSS points 
> - $order = 2M-1$ for RADAU-RIGHT and RADAU-LEFT points
> - $order = 2M-2$ for LOBATTO points

## Experiments

Optimum coefficients can be computed with the [spectralRadius.py](../scripts/fact3/spectralRadius.py) script, and are stored into the [optimDiagCoeffs.md](../scripts/fact3/optimDiagCoeffs.md) markdown file (updated for every run of `spectralRadius.py`). Then the [checkSum.py](../scripts/fact3/checkSum.py) script compare sum of diagonal coefficients with the theoretical proposed one.