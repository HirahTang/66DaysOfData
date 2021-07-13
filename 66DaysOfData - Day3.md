### 66DaysOfData - Day3

# Numerical Computation

## (Deep Learning Chapter 4)

### Overflow and Underflow

Rounding errors:

* Underflow: when numbers near zero are rounded to zero
* Overflow: When numbers with large magnitude are approximated as $\infty$ or $-\infty$ 

A function that must be stabilized against underflow and overflow is softmax function.
$$
\operatorname{softmax}(\boldsymbol{x})_{i}=\frac{\exp \left(x_{i}\right)}{\sum_{j=1}^{n} \exp \left(x_{j}\right)}
$$
Overflow and underflow will affect softmax for very negative (underflow) or very large and positive (overflow)

It can be resolved by evaluating softmax(z) where $z = x - max_ix_i$

"We can rely on low-level libraries for the numerical issues"

### Poor Conditioning

"Conditioning refers to how rapidly a function changes with respect to small changes in its inputs"

for $f(\boldsymbol{x})=\boldsymbol{A}^{-1} \boldsymbol{x}$ When A has an eigenvalue decomposition, its condition number is
$$
\max _{i, j}\left|\frac{\lambda_{i}}{\lambda_{j}}\right|
$$
When this number is large matrix inversion is particularly sensitive to error in the input.

### Gradient-Based Optimization

$ x^* = argmin f(x)$ 

the derivative is useful for minimizing a function

gradient descent is used based on the derivative of f(x)

$f'(x) = 0$ is the critical points / stationary points where to find a local minima/maxima or saddle points

For functions with multiple inputs, we make use of the concept of partial derivatives.

 $\frac{\partial}{\partial x_{i}} f(\boldsymbol{x})$ measures how $f$ changes at the direction $x_i$ 

the gradient of $f$ is the vector containing all of the partial derivatives, denoted as $\nabla_{\boldsymbol{x}} f(\boldsymbol{x})$ 

**Directional derivative** in direction $u$ is the slope of the function $f$ in direction $u$

$\frac{\partial}{\partial \alpha} f(\boldsymbol{x}+\alpha \boldsymbol{u})$ evalutaes to $\boldsymbol{u}^{\top} \nabla_{\boldsymbol{x}} f(\boldsymbol{x})$ when $u = 0$ (The gradient of f(x) at the direction of u)

we need to find the direction in which f decreases the fastest.
$$
\begin{aligned}
\min _{\boldsymbol{u}, \boldsymbol{u}^{\top} \boldsymbol{u}=1} \boldsymbol{u}^{\top} \nabla_{\boldsymbol{x}} f(\boldsymbol{x}) \\
=\min _{\boldsymbol{u}, \boldsymbol{u}^{\top} \boldsymbol{u}=1}\|\boldsymbol{u}\|_{2}\left\|\nabla_{\boldsymbol{x}} f(\boldsymbol{x})\right\|_{2} \cos \theta
\end{aligned}
$$
$||u||_2 = 1$ $min_ucos\theta$ 

it minimizes where u points in the opposite direction as the gradient. The gradient points directly uphill, and the negative gradient points directly downhill.

We use line search to choose the optimised stepwise $$\epsilon$$ 

#### Beyond the Gradient: Jacobian and Hessian Matrices

**Jacobian matrix**: The matrix containing all partial derivatives of a function whose input and output are both vectors.

For function $\boldsymbol{f}: \mathbb{R}^{m} \rightarrow \mathbb{R}^{n}$, Jacobian matrix $\boldsymbol{J} \in \mathbb{R}^{n \times m}$ of $f$ is defined as such that $J_{i, j}=\frac{\partial}{\partial x_{j}} f(\boldsymbol{x})_{i}$ 

Second derivative: derivative of a derivative



