# Instalation

Download pythag (in current directory) using

```bash
git clone https://gitlab.com/tlunet/pythag pythag_repo
```

Then create a symbolic link to the python package

```bash
ln -sf pythag_repo/pythag pythag
```

# Main modules

- qmatrix.py : generate Q, QDelta, nodes, ... for many type of distribution
- gmfa.py : some analytic functions (not usefull here ...)

# Main Scripts

- approxLagrange.py : visualize lagrange polynomial for equidistant / gauss-type nodes, and Q-QDelta as matrix
- invest_QmQDelta.py : original script to look at ||Q-QDelta||
- intuition.py : new script to investigate ||Q-QDelta||
