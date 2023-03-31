# Fact n°5

> The matrix $Q-Q_{\Delta}(\frac{\tau}{M})$ is nilpotent, where $Q_{\Delta}(x)$ a diagonal matrix with $x$ on the diagonal. 
> 
> This means that $\rho(Q-Q_{\Delta}(\frac{\tau}{M})) = 0$ and it definitely minimizes the spectral radius!

## Proof
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

## Examples
Run the [nilpotency.py](../scripts/fact5/nilpotency.py) script. The output is the maximum element in the powers of our matrix.
For 4 `GAUSS-LEGENDRE` points we get

```python
|(Q - D)^0|_max = 1.0
|(Q - D)^1|_max = 0.35395300603375063
|(Q - D)^2|_max = 0.10442785312900467
|(Q - D)^3|_max = 0.01922398239399945
|(Q - D)^4|_max = 8.552412457109149e-19
```
