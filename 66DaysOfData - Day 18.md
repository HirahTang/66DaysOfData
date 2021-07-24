### 66DaysOfData - Day 18

# Optimization for Training Deep Models

## Deep Learning Chapter 8

### Approximate Second-Order Methods

#### Newton's Method

Newton's method is an optimization scheme based on using a second-order Taylor series expansion to approximate $J(\theta)$ near some point $\theta_{0}$ ignoring derivatives of higher order

$J(\boldsymbol{\theta}) \approx J\left(\boldsymbol{\theta}_{0}\right)+\left(\boldsymbol{\theta}-\boldsymbol{\theta}_{0}\right)^{\top} \nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{0}\right)+\frac{1}{2}\left(\boldsymbol{\theta}-\boldsymbol{\theta}_{0}\right)^{\top} \boldsymbol{H}\left(\boldsymbol{\theta}-\boldsymbol{\theta}_{0}\right)$

$H$ - the Hessian of $J$ with resepct to $\theta$ evaluated at $\theta_{0}$

when we solve for the critical point of this function, we obtain the Newton parameter update rule:

$\boldsymbol{\theta}^{*}=\boldsymbol{\theta}_{0}-\boldsymbol{H}^{-1} \nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{0}\right)$

As long as the Hessian Matrix remains positive definite, Newton's method can be applied iteratively to the minimum.

For non-convex objective function, the eigenvalues of Hessian matrix are not all positive (means the existence of saddle points, etc.).

We avoid this situation by regularizing the Hessian, such as adding a constant $\alpha$ along the diaonal of the Hessian

$\boldsymbol{\theta}^{*}=\boldsymbol{\theta}_{0}-\left[H\left(f\left(\boldsymbol{\theta}_{0}\right)\right)+\alpha \boldsymbol{I}\right]^{-1} \nabla_{\boldsymbol{\theta}} f\left(\boldsymbol{\theta}_{0}\right)$

When strong negative curvature is present, $\alpha$ may need to be so large that Newton's method would ake smaller steps than gradient descent with a properly chosen learning rate.

The significant computational burden also limits the application of Newton's method for deep learning. With $k$ parameters, Newton's method would require the inversion of a $k \times k $ matrix, with computational complexity of $O(k^3)$

#### Conjugate Gradients

Conjugate Gradients avoid the calculation of the inverse Hessian by iteratively descending conjugate directions.

For steepest descent, for quadratic bowl, the trajectory is a ineffective back-and-forth, zig-zag pattern. Because each line search direction, when given by the gradient, is guaranteed to be orthogonal? to the previous line search direction.

https://www.youtube.com/watch?v=h4cG8jLGmKg

The training iteration $t$, the next direction $d_t$ takes the form$\boldsymbol{d}_{t}=\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})+\beta_{t} \boldsymbol{d}_{t-1}$ 

Two directions $d_t$ and $d_{t-1}$ are defined as conjugate if $\boldsymbol{d}_{t}^{\top} \boldsymbol{H} \boldsymbol{d}_{t-1}=0$ 

Two methods of calculating $\beta_{t}$ without calculating the eigenvalues of $H$:

1. Fletcher - Reeves: $\beta_{t}=\frac{\nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{t}\right)^{\top} \nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{t}\right)}{\nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{t-1}\right)^{\top} \nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{t-1}\right)}$
2. Polak-Ribièe: $\beta_{t}=\frac{\left(\nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{t}\right)-\nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{t-1}\right)\right)^{\top} \nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{t}\right)}{\nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{t-1}\right)^{\top} \nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{t-1}\right)}$

For a quadratic surface, conjugate directions let us stay at the minimum along the previous directions. In a k-dimensional parameter space, the conjugate gradient method requires at most $k$ line searches to achieve the minimum.

**Nonlinear Conjugate Gradients**

Conjugate gradients also applicable for trainig neural networks even their objective functions are far from quadratic.

The nonlinear conjugate gradients includes occasional resests where the method of conjugate gradients is restarted with line search along the unaltered gradient.



 