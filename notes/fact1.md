# Fact n°1

> For Backward Euler (BE) or Forward Euler (FE) sweep, and any type of quadrature type (GAUSS, LOBATTO, RADAU-[...]), then
>
>$$
>\underset{M \rightarrow\infty}{\lim} || Q-Q_{\Delta} ||_\infty \simeq \left(\frac{\pi}{2M+2}\right)^{0.885}
>$$
>
> where $\left(\frac{\pi}{2M+2}\right) \underset{M \rightarrow\infty}{\sim} \Delta\tau_{max}(M)$, with $\Delta\tau_{max}(M)$ the largest gap between the $M$ collocation nodes written in $[0,1]$.

## Hypothesis

$Q_\Delta$ defined for Backward Euler like this :

```math
Q_\Delta = \begin{bmatrix}
\Delta\tau_1 & & &\\
\Delta\tau_1 & \Delta\tau_2 & &\\
\vdots & \vdots & \ddots &\\
\Delta\tau_1 & \Delta\tau_2 & \dots & \Delta\tau_M
\end{bmatrix}
```

and for Forward Euler :

```math
Q_\Delta = \begin{bmatrix}
0 & & &\\
\Delta\tau_1 & 0 & &\\
\vdots & \vdots & \ddots &\\
\Delta\tau_1 & \Delta\tau_2 & \dots & 0
\end{bmatrix}
```

Proof of :

```math
\left(\frac{\pi}{2M+2}\right) 
\underset{M \rightarrow\infty}{\sim} 
\Delta\tau_{max}(M)
```

for Gauss nodes (only)  ... incoming ...
