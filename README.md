# Some strange facts about the $Q-Q_{\Delta}$ matrix (... why ?!?)

_:scroll: This repository contains some open questions following discussions and experiments on the collocation matrices arising in the SDC time-integration method ... some standalone python code is provided to illustrate those facts._

## Summary

### Fact n°1

> For Backward Euler (BE) or Forward Euler (FE) sweep, and any type of quadrature type (GAUSS, LOBATTO, RADAU-[...]), then
> $$
> \underset{M \rightarrow\infty}{\lim} || Q-Q_{\Delta} ||_\infty \simeq \left(\frac{\pi}{2M+2}\right)^{0.885}
> $$
> where $\left(\frac{\pi}{2M+2}\right) \underset{M \rightarrow\infty}{\sim} \Delta\tau_{max}(M)$, with $\Delta\tau_{max}(M)$ the largest gap between the $M$ collocation nodes written in $[0,1]$.

More details in [this notes](./notes/fact1.md), python scripts illustrating this are located [here ...](./scripts/fact1/)

### Fact n°2

> The vector $x$ maximizing $|| (Q-Q_{\Delta})x||_\infty$ is of the form 
> $$
> x = [1, 1, \dots, 1, -1, -1, \dots, -1]
> $$
> with $(M+1)//2$ leading positive coefficients. Also, $-x$ is also maximizing the norm.

More details in [this notes](./notes/fact2.md), python scripts illustrating this are located [here ...](./scripts/fact2/)
