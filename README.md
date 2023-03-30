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

> The matrix $Q-Q_{\Delta}(\frac{\tau}{M})$, where $Q_{\Delta}(x)$ is nilpotent. 
> 
> This means that $\rho(Q-Q_{\Delta}(\frac{\tau}{M})) = 0$ and it definitely minimizes the spectral radius!

### Proof
The matrix satisfies $(Q-Q_{\Delta}(\frac{\tau}{M}))\left[\tau_1^{M-1}, \dots, \tau_{M}^{M-1}\right]^T = 0$. 
This is because $Q$ integrates polynomials up to degree $M-1$ in an exact way, in other words

$$
Q\left[\tau_1^{k}, \dots, \tau_{M}^{k}\right]^T = \left[\frac{\tau_1^{k+1}}{k+1}, \dots, \frac{\tau_M^{k+1}}{k+1}\right]^T, \quad 0 \leq k \leq M-1.
$$

Let $x^{(0)}(\tau) = a_{M-1}^{(0)} \tau^{M-1} + \dots + a_1^{(0)}\tau +a_{0}^{(0)}$.
Then,

$$
\left(Q-Q_{\Delta}(\frac{\tau}{M})\right)\left[x^{(0)}(\tau_1), \dots, x^{(0)}(\tau_M)\right]^T = \left[x^{(1)}(\tau_1), \dots, x^{(1)}(\tau_M)\right]^T,
$$

where $x^{(1)}(\tau) = a_{M-1}^{(1)} \tau^{M-1} + \dots + a_1^{(1)}\tau$. This is because the integration with the matrix $Q$ raises the degree of $x^{(0)}$ by 1 and then $Q_{\Delta}$ removes the highest degree.
So if we continue this process, after $k$ applications of our matrix to vector/polynomial $x^{(0)}$, we have

$$
x^{(k)}(\tau) = a_{M-1}^{(k)} \tau^{M-1} + \dots + a_k^{(k)}\tau^k.
$$

More precisely, after $M$ applications, we get that $x^{(M)} = 0$ for an arbitrary starting polynomial, concluding $(Q-Q_{\Delta}(\frac{\tau}{M}))^M = 0$.
Because it is a nilpotent matrix, it's charachteristic polynomials is $\lambda^M$ and the spactral radius is 0, since all the eigenvalues are 0.

### Examples
Run the [nilpotency.py](../scripts/fact5/nilpotency.py) script. The output is the maximum element in the powers of our matrix.
For 4 `GAUSS-LEGENDRE` points we get

```python
|(Q - D)^0|_max = 1.0
|(Q - D)^1|_max = 0.35395300603375063
|(Q - D)^2|_max = 0.10442785312900467
|(Q - D)^3|_max = 0.01922398239399945
|(Q - D)^4|_max = 8.552412457109149e-19
```
