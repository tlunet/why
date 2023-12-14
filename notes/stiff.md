# Computations for the stiff limit

_:scroll: Attempt to find the coefficients of $D$ (or simply prove their existence) such that $I-DQ$ is nilpotent ..._

## Brute-force computations using Lagrange basis

Define 

$$
D = 
\begin{pmatrix} 
d_1 & & \\
& \ddots & \\
& & d_M
\end{pmatrix}
$$

and $Q$ the collocation matrix (not including $t=0$ as first node, for simplicity ...).

In polynomial space $P_{M-1}(\mathbb{R})$ 
(that have dimension $M$, isomorph to $\mathbb{R}^{M}$)
we have for every polynomial of this space :

$$
D[p] : p \mapsto \sum_{i=1}^{M} d_i p(\tau_i)l_i(x),
$$

$$
Q[p] : p \mapsto \int_{0}^{x} p(t)dt,
$$

with $l_i(x)$ the Lagrange basis of $P_{M-1}(\mathbb{R})$
using the nodes $\tau_i$.

> :bell: For some reason, $Im(Q)$ is not restricted to
$P_{M-1}(\mathbb{R})$, so there is probably some clarification to do on this point.

From the definition of $D$ and $Q$ in polynomial space, we can compute :

$$
\begin{align*}
D \circ Q [l_m] 
&= \sum_{i=1}^{M} d_i \left[\int_{0}^{\tau_i} l_m(t)dt \right]l_i(x), \\
&= \sum_{i=1}^{M} d_i q_{i,m}l_i(x),
\end{align*}
$$

where $q_{i,m}$ are the coefficients of 
$Q$.

Now let us compute the generic expression of 
$(D \circ Q)^k [l_m]$ for each power $k$.
First we have for $k=2$ :

$$
\begin{align*}
(D \circ Q)^2 [l_m]
&= \sum_{i=1}^{M} d_i q_{i,m}
    \sum_{j=1}^{M} d_j q_{j,i}l_j(x) \\
&= \sum_{i=1}^{M}\sum_{j=1}^{M}
    d_i q_{i,m} d_j q_{j,i}l_j(x) \\
&= \sum_{i=1}^{M}\sum_{j=1}^{M}
    d_i d_j q_{i,j} q_{j,m} l_i(x) 
    \quad \text{(switching  $i$ and $j$)} \\
&= \sum_{i=1}^{M} d_i \left[
    \sum_{j=1}^{M} d_j q_{i,j} q_{j,m}
    \right] l_i(x)
\end{align*}
$$

Second for $k=3$ we have :

$$
\begin{align*}
(D \circ Q)^3 [l_m]
&= \sum_{i=1}^{M} d_i q_{i,m}
    \sum_{j=1}^{M} d_j q_{j,i}
    \sum_{n=1}^{M} d_n q_{n,j} l_n(x) \\
&= \sum_{i=1}^{M}\sum_{j=1}^{M}\sum_{n=1}^{M}
    d_i q_{i,m} d_j q_{j,i} d_n q_{n,j} l_n(x)  \\
&= \sum_{i=1}^{M}\sum_{j=1}^{M}\sum_{n=1}^{M}
    d_i d_j d_n q_{i,j} q_{j,n} q_{n,m} l_i(x) 
    \quad \text{(with $i,j,n \rightarrow n,j,i$)} \\
&= \sum_{i=1}^{M} d_i \left[
    \sum_{j=1}^{M}\sum_{n=1}^{M} 
        d_j d_n q_{i,j} q_{j,n} q_{n,m}
    \right] l_i(x)
\end{align*}
$$

And so on for higher $k$, so we can write generally for $k>0$:

$$
(D \circ Q)^k [l_m] 
= \sum_{i=1}^{M} d_i T(i,m,k) l_i(x)
$$

with the $T(i,m,k)$ term for $k>1$ defined using the Einstein sum convention :

$$
\begin{align*}
T(i,m,k) &= q_{i,m} \quad \text{if $k=1$,} \\
&= d_{j(1)}~...~d_{j(k-1)}~
    q_{i,j(1)}~q_{j(1),j(2)}~...~q_{j(k-2),j(k-1)}~q_{j(k-1),m} \quad \text{else.}
\end{align*}
$$

> :bell: While the $T$ term is a bit disgusting for hand computations, it is easy to compute numerically from any given $D$ and $Q$ coefficients using `np.einsum`.

Now let's move to the full stiff operator. For each polynomial of the Lagrange basis :

$$
\begin{align*}
(I-DQ)^M[l_m(x)] 
&= l_m(x) + \sum_{k=1}^{M}\binom{M}{k}(-1)^k 
    (D \circ Q)^k [l_m],\\
&= l_m(x) + \sum_{k=1}^{M}\binom{M}{k}(-1)^k
    \sum_{i=1}^{M} d_i T(i,m,k) l_i(x)
\end{align*}
$$

then, if we suppose that $I-DQ$ is nilpotent to the power $M$, then for each Lagrange basis (i.e each $m$), all coefficients (associated to each $i$ in the sum) that compose the linear combination of Lagrange basis are zeros.
So we have for each $m$ the following non-linear system :

$$
\begin{align*}
&1 + d_i\sum_{k=1}^{M}\binom{M}{k}(-1)^k T(i,m,k) = 0
\quad \text{if $i=m$,} \\
&\sum_{k=1}^{M}\binom{M}{k}(-1)^k T(i,m,k) = 0 
\quad \text{if $i\neq m$.}
\end{align*}
$$

> :mega: This system is numerically implemented in [one python script](../scripts/fact7/determinant.py), and verified for all $m$ using the solution from `MIN-SR-S`.

## To go further ...

_What do we have :_ analytical expression of at least one linear system that has to be solved to find the solution.
Basically, what we wanted to get a few month ago, either by using the determinant approach analytically, 
or trying to find a kernel polynomial ...

_What's next :_

1. If we can evaluate the Jacobian of each of those system, we can use Newton to find the solution (hopefully with a quite good accuracy and stability). This part is still tricky, as I did not find any simple way to generally write for any $\ell$ :

$$
\frac{\partial T(i,m,k)}{\partial d_\ell}
$$

2. We could have obtain exactly the same computation in matrix space, as computing with the Lagrange basis correspond to computing with the natural basis of $\mathbb{R}^M$ in matrix space. But we could try to find an other basis, eventually built incrementally with the $d_i$, such that application of $I-DQ$ is convenient on this basis. As a matter of fact, that's what we did for the non-stiff limit, building incrementally the Vandermonde basis such that application of $Q-D$ on the last basis vector then produced the previous basis vector, etc ... there is probably something to search in that direction for the stiff limit.

