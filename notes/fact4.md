# Fact n°4

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

## Experiments

Just run the [diagonalNodes.py](../scripts/fact4/diagonalNodes.py) script with the parameter you want, and look at the output.
Then you can compare to the diagonal values obtained with minimization of the spectral radius (cf [Fact n°3](fact3.md)) and see that `nodes/M` correspond to one optimal set of coefficients in [optimDiagCoeffs.md](../scripts/fact3/optimDiagCoeffs.md).

In particular for 4 equidistant "Lobatto" points, we get the optimial coefficients :

```python
xOpt = (0.0, 0.083334, 0.166667, 0.25)
```

which is the same as the nodes divided by $M=5$ when you take those nodes :

```python
nodes = [0.0, 1/3, 2/3, 1.0]
```

Same works for equidistant "GAUSS" nodes, with $M \in {3, 4}$, see [optimDiagCoeffs.md](../scripts/fact3/optimDiagCoeffs.md) and associated run with [diagonalNodes.py](../scripts/fact4/diagonalNodes.py).