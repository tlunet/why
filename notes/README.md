# Hypothesis

For any kind of orthogonal-based node distribution, we have

```math
||Q-Q_\Delta||_\infty \simeq \left(\frac{\pi}{2M+2}\right)^{7/8} \simeq
\Delta_{max}^{7/8}
```

with $`Q_\Delta`$ defined for Backward Euler like this :

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
\left(\frac{\pi}{2M+2}\right)^{7/8} \simeq
\Delta_{max}^{7/8}
```

for Gauss nodes (only)  ...incoming...
