### 66DaysOfData - Day4

# Numerical Computation

## Deep Learning Chapter 4

#### Beyond the Gradient: Jacobian and Hessian Matrices

second derivative is measuring curvature

When our function has multiple input dimensions, there are many second derivatives. They collected together into a matrix called the Hessian matrix,

$H(f)(x)$ Defined as
$$
\boldsymbol{H}(f)(\boldsymbol{x})_{i, j}=\frac{\partial^{2}}{\partial x_{i} \partial x_{j}} f(\boldsymbol{x})
$$
the Hessian is the Jacobian of the gradient

Hessian matrix is symmetric, $H_{i, j} = H_{j,i}$

![image-20210701155829947](/Users/hirahtang/Library/Application Support/typora-user-images/image-20210701155829947.png)

The Hessian matrix is real and symmetric, so we can decompose it into a set of real eigenvalues and an orthgonal basis of eigenvectors.

"The second derivative is a specific direction represented by a unit vector $d$ is given by $\boldsymbol{d}^{\top} \boldsymbol{H} \boldsymbol{d}$ "For all the directions of $d$, the directional second derivative is a weighted average of all the eigenvalues.

We make a second-order Taylor Series approximation to the function $f(x)$ around the current point $x^{(0)}$ 
$$
f(\boldsymbol{x}) \approx f\left(\boldsymbol{x}^{(0)}\right)+\left(\boldsymbol{x}-\boldsymbol{x}^{(0)}\right)^{\top} \boldsymbol{g}+\frac{1}{2}\left(\boldsymbol{x}-\boldsymbol{x}^{(0)}\right)^{\top} \boldsymbol{H}\left(\boldsymbol{x}-\boldsymbol{x}^{(0)}\right)
$$

$$
f\left(\boldsymbol{x}^{(0)}-\epsilon \boldsymbol{g}\right) \approx f\left(\boldsymbol{x}^{(0)}\right)-\epsilon \boldsymbol{g}^{\top} \boldsymbol{g}+\frac{1}{2} \epsilon^{2} \boldsymbol{g}^{\top} \boldsymbol{H} \boldsymbol{g}
$$
![image-20210701161510013](/Users/hirahtang/Library/Application Support/typora-user-images/image-20210701161510013.png)

**second derivative test**

$f''(x) = 0, f''(x)>0$ - local minimum

$f''(x) = 0, f''(x)<0$ - local maximum

"Using the eigendecomposition of the Hessian matrix, we can generalize the second derivative test to multiple dimensions. At a critical point, where ∇xf(x) = 0, we can examine the eigenvalues of the Hessian to determine whether the critical point is a local maximum, local minimum, or saddle point."

saddle point means x is a local maximum on one cross section of $f$ but a local minimum on another cross section, which makes the multidimensional second derivative test be inconclusive.

First-order optimization & second-order optimization

**Lipschitz continuous**

lipschitz continuous function is a function f whose rate of change is bounded by a Lipschitz constant $\mathcal{L}$
$$
\forall \boldsymbol{x}, \forall \boldsymbol{y},|f(\boldsymbol{x})-f(\boldsymbol{y})| \leq \mathcal{L} \| \boldsymbol{x}-\left.\boldsymbol{y}\right||_{2}
$$
a small change in the input result in a small change in the output

Convex optimization:

Convex functions - functions for which the Hessian is positive semidefinite everywhere.

No saddle points, all local minima are global minima

