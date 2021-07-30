### 66DaysOfData - Day 20

# Optimization for Training Deep Models

## Deep Learning Chapter 8

#### BFGS

The Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm attempts to bring some of the advantages of Newton's method without the computational burden.

Newton's update is given by $\boldsymbol{\theta}^{*}=\boldsymbol{\theta}_{0}-\boldsymbol{H}^{-1} \nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{0}\right)$ 

$H$ is the Hessian of $J$ with respect to $\theta$ evaluated at $\theta_{0}$

The main difficulty is on calculating $H^{-1}$

BFGS calculates the inver Hessian approximation $M_{t}$, 

the direction of descent $\rho_{t}$:

$\boldsymbol{\rho}_{t}=\boldsymbol{M}_{t} \boldsymbol{g}_{t}$

the update of the parameter:

$\boldsymbol{\theta}_{t+1}=\boldsymbol{\theta}_{t}+\epsilon^{*} \boldsymbol{\rho}_{t}$

**Limited Memory BFGS (L-BFGS)**

Decrease the memory cost of BFGS (need to store the inverse Hessian Matrix, which takes $O(n^{2})$ memory, it begins with the assumption that $M^{t-1}$ is the identity matrix, rather than storing the approximation from one step to the next.