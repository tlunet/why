# Strange facts about $Q-Q_{\Delta}$ (... why ?!?)

_:scroll: This repository contains some open questions following discussions and experiments on the collocation matrices arising in the SDC time-integration method ... some standalone python code is provided to illustrate those facts._

In addition, an IMEX SDC implementation is provided with some script allowing a first analysis of SDC with QDelta matrix arising from those facts, see [more details here ...](./scripts/sdc/README.md)

## Fact n°1

> For Backward Euler (BE) or Forward Euler (FE) sweep, and any type of quadrature type (GAUSS, LOBATTO, RADAU-[...]), then
>
>$$
>\underset{M \rightarrow\infty}{\lim} || Q-Q_{\Delta} ||_\infty \simeq \left(\frac{\pi}{2M+2}\right)^{0.885}
>$$
>
> where $\left(\frac{\pi}{2M+2}\right) \underset{M \rightarrow\infty}{\sim} \Delta\tau_{max}(M)$, with $\Delta\tau_{max}(M)$ the largest gap between the $M$ collocation nodes written in $[0,1]$.

More details in [this notes](./notes/fact1.md), python scripts illustrating this are located [here ...](./scripts/fact1/)


## Fact n°2

> The vector $x$ maximizing $|| (Q-Q_{\Delta})x||_\infty$ is of the form 
>
> $$
> x = [1, \dots, 1, -1, \dots, -1]
> $$
>
> with $(M+1)//2$ leading positive coefficients. Also, $-x$ is also maximizing the norm.

More details in [this notes](./notes/fact2.md), python scripts illustrating this are located [here ...](./scripts/fact2/)


## Fact n°3

> When minimizing the spectral radius of $Q-Q_{\Delta}(x)$, with $Q_{\Delta}(x)$ a diagonal matrix with $x$ on the diagonal, for any local minimum $x$ we have
>
> $$
> \sum x_i = \frac{N_{coeff}}{order},
> $$
>
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

More details in [this notes](./notes/fact3.md), python scripts illustrating this are located [here ...](./scripts/fact3/)

## Fact n°4

> When minimizing the spectral radius of $Q-Q_{\Delta}(x)$, with $Q_{\Delta}(x)$ a diagonal matrix with $x$ on the diagonal, then the **global** minimum $x_{min}$ is
>
> $$
> x_{min} = \frac{\tau}{M},
> $$
>
> where $\tau$ are the collocation nodes written in $[0,1]$. Also, we have $[Q-Q_{\Delta}(\tau)]e=0$ with $e=[1,1,\dots,1]$, and this works **also** for equidistant nodes.
> For Legendre nodes, the combination with Fact n°3 gives
>
> $$
> \sum \tau_i = \frac{M N_{coeff}}{order}
> $$

More details in [this notes](./notes/fact4.md), python scripts illustrating this are located [here ...](./scripts/fact4/)

## Fact n°5

> The matrix $Q-Q_{\Delta}(\frac{\tau}{M})$ is nilpotent, where $Q_{\Delta}(x)$ a diagonal matrix with $x$ on the diagonal. 
> 
> This means that $\rho(Q-Q_{\Delta}(\frac{\tau}{M})) = 0$ and it definitely minimizes the spectral radius!

More details in [this notes](./notes/fact5.md), python scripts illustrating this are located [here ...](./scripts/fact5/)
