### 66DaysOfData - Day2

# Regularization for Deep Learning 

## (Deep Learning Chapter 7)

#### $L^1$ Regularization

Defined as
$$
\Omega(\boldsymbol{\theta})=\|\boldsymbol{w}\|_{1}=\sum_{i}\left|w_{i}\right|
$$
(you don't necessarily regularise the weight to 0, if you are regularising the parameters toward some value $w^{(o)}$, the $L^1$ regularization changes to $\Omega(\boldsymbol{\theta})=\left\|\boldsymbol{w}-\boldsymbol{w}^{(o)}\right\|_{1}=\sum_{i}\left|w_{i}-w_{i}^{(o)}\right|$  )

The regularized objective function $\tilde{J}(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})$ given by
$$
\tilde{J}(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})=\alpha\|\boldsymbol{w}\|_{1}+J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})
$$
With the corresponding gradient
$$
\nabla_{\boldsymbol{w}} \tilde{J}(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})=\alpha \operatorname{sign}(\boldsymbol{w})+\nabla_{\boldsymbol{w}} J(\boldsymbol{X}, \boldsymbol{y} ; \boldsymbol{w})
$$
$sign(w)$ is the sign of $w$

$$ sign(w) = 1 : w > 0, sign(w) = -1:w<0$$

##### Effect:

The regularization contribution to the gradient no longer scales linearly with each $w_i$, it is now a constant factor with a sign equal to $sign(w_i)$ 

We will not necessarily see clean algebraic solutions to quadratic approximations of $J(\boldsymbol{X}, \boldsymbol{y} ; \boldsymbol{w})$ as we did for $L^2$ regularization.

?Simple linear model has a quadratic cost function Taylor series. We could imagine that this is a truncated Taylor series approximating the cost function of a more sophisticated model. The gradient is:?
$$
\nabla_{\boldsymbol{w}} \hat{J}(\boldsymbol{w})=\boldsymbol{H}\left(\boldsymbol{w}-\boldsymbol{w}^{*}\right)
$$
? $H$ is the Hessen matrix of $J$ with respect to $w$ evaluated at $w^*$ ?

$L^1$ penalty does not admit clean algebraic expressions in the case of a fully general Hessian, we make the further simplifying assumption that the Hessian is diagonal $\boldsymbol{H}=\operatorname{diag}\left(\left[H_{1,1}, \ldots, H_{n, n}\right]\right)$ where each $H_{i,i} > 0$. The assumption hold if the data to remove all correlation between the input features (accoplished using PCA).

Our quadratic approximation of the $L^1$ regularized objective function decomposed into a sum over the parameters:?
$$
\hat{J}(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})=J\left(\boldsymbol{w}^{*} ; \boldsymbol{X}, \boldsymbol{y}\right)+\sum_{i}\left[\frac{1}{2} H_{i, i}\left(\boldsymbol{w}_{i}-\boldsymbol{w}_{i}^{*}\right)^{2}+\alpha\left|w_{i}\right|\right]
$$
The analytical solution for it (for each dimension $i$):
$$
w_{i}=\operatorname{sign}\left(w_{i}^{*}\right) \max \left\{\left|w_{i}^{*}\right|-\frac{\alpha}{H_{i, i}}, 0\right\}
$$

1. For $w_{i}^{*} \leq \frac{\alpha}{H_{i, i}}$, the optimal value of $w_i$ = 0. This occurs because the contribution of $J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})$ to the regularised objective $\tilde{J}(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})$ is overwhelmed, which the $L^1$ regularization which pushes the value of $w_i$ to zero.
2. For $w_{i}^{*} > \frac{\alpha}{H_{i, i}}$, the $w_i$ shifts to zero by a distance of $\frac{\alpha}{H_{i, i}}$.

The same for $w_i < 0$.

"$L^1$ regularization results in a solution that is more sparse, which makes it has been used extensively as a feature selection mechanism"

LASSO (least absolute shrinkage and selection operator) model integrates $L^1$ penalty with a linear model.

?Many regularization strategies can be interpreted as MAP Bayesian inference?

"$L^2$ regularization is equivalent to MAP Bayesian inference with Gaussian prior on the weights"?

"For $L^1$ regularization the penalty $\alpha \Omega(\boldsymbol{w})=\alpha \sum_{i}\left|w_{i}\right|$ is equivalent to the log-prior term that is maximized by MAP Bayesian inference when the prior is an 'isotropic' Laplace distribution"

???
$$
\log p(\boldsymbol{w})=\sum_{i} \log \operatorname{Laplace}\left(w_{i} ; 0, \frac{1}{\alpha}\right)=-\alpha\|\boldsymbol{w}\|_{1}+n \log \alpha-n \log 2
$$
**I am not familiar with Hessien Matrix, makes the derivation of L1 intelligible to me. I forget the Tyler Series applied in calulating the gradient of the objective function. The final MAP Bayesian inference interpretation of regularization is intelligible to me either.**

## Norm Penalties as Constrained Optimization

For the cost function regularized by a parameter norm penalty
$$
\tilde{J}(\boldsymbol{\theta} ; \boldsymbol{X}, \boldsymbol{y})=J(\boldsymbol{\theta} ; \boldsymbol{X}, \boldsymbol{y})+\alpha \Omega(\boldsymbol{\theta})
$$
"We can minimize a function subject to constraints by constructing a generalized Lagrange function, consisting of the original objective function plus a set of penalties. | Each penalty is a product between a coefficient, called a Karush-Kagn-Tucker (KKT) multiplier, and a function representing whether the constraint is satisfied."???



When we constrain $\Omega(\theta)$ to be less than some constant k, the Lagrange function:
$$
\mathcal{L}(\boldsymbol{\theta}, \alpha ; \boldsymbol{X}, \boldsymbol{y})=J(\boldsymbol{\theta} ; \boldsymbol{X}, \boldsymbol{y})+\alpha(\Omega(\boldsymbol{\theta})-k)
$$
The solution to the constrained problem is given by 
$$
\boldsymbol{\theta}^{*}=\underset{\boldsymbol{\theta}}{\arg \min } \max _{\alpha, \alpha \geq 0} \mathcal{L}(\boldsymbol{\theta}, \alpha)
$$
(Look at Section 4.4 and Section 4.5 for description & worked examples.)

"In all procedures $\alpha$ must increase whenever $\Omega(\theta) > k$ and decrease whenever $\Omega(\theta) < k$" 

"All positive $\alpha$ encourage $\Omega(\theta)$ to shrink. The optimal value $\alpha^*$ will encourage $\Omega(\theta)$ to shrink, but not so strongly to make $\Omega(\theta)$ become less than k"???

We fix $\alpha^*$ and view the problem as just a function of $\theta$:
$$
\boldsymbol{\theta}^{*}=\underset{\boldsymbol{\theta}}{\arg \min } \mathcal{L}\left(\boldsymbol{\theta}, \alpha^{*}\right)=\underset{\boldsymbol{\theta}}{\arg \min } J(\boldsymbol{\theta} ; \boldsymbol{X}, \boldsymbol{y})+\alpha^{*} \Omega(\boldsymbol{\theta})
$$
Which is the same as the regularised training problem of minimizing a regularized objective functoin.

"A parameter norm penalty is imposing a constraint on the weights"

$\Omega$ is $L^2$ Norm $\rightarrow$ the weights are constrained to lie in a ball

$\Omega$ is $L^1$ Norm $\rightarrow$ the weights are constrained in a region of limited L1 norm.

"We do not know the exact size of the constraint region, but we can control it roughly by changin $\alpha$"

We use explicit constraints instead of weight penalties.

1. We can modify algorithms such as stochastic gradient descent to take a step downhill on $J(\theta)$ and then project $\theta$ back to the nearest point that satisfies $\Omega(\theta) < k$. When we know the appropriate value of k, and do not need to search for the value of $\alpha$.

2. Penalties can cause non-convex optimisation procedures to get stuck in local minima corresponding to small $\theta$, which is preventable by a explicit constraint.

3. They impose some stability on the optimization procedure. To prevent high learning rate $\rightarrow$ poitive feedback loop with large weigts induce large gradients, which makes the weights larger. Explicit constraints can continue to increase the magnitude of the weights without bound.

   **For this part, I am not familiar the Lagrange function and the KKT transformation, I'll review them in the coming days**

