#66DaysOfData - Day1

# **Regularization for Deep Learning **

## (Deep Learning Chapter 7)

Regularisation: Strategies desiged to reduce the test error, possibly at the expense of increased training error.

"Put extra constraints on a machine learning model, such as adding restrictions on the parameter values. Some add extra terms in the objective function that can be thought of as corresponding to a soft constraint on the parameter values."

"Most regulatization strategies are based on regulaizing estimators."

### Parameter Norm Penalties

adding a parameter norm penalty $$\Omega(\boldsymbol{\theta})$$ to the objective function $J$. 

The regularized objective function $\tilde{J}$
$$
\tilde{J}(\boldsymbol{\theta} ; \boldsymbol{X}, \boldsymbol{y})=J(\boldsymbol{\theta} ; \boldsymbol{X}, \boldsymbol{y})+\alpha \Omega(\boldsymbol{\theta})
$$
“Different choices for the parameter norm $\Omega$ can resut in different solutions being preferred.”

"For neural networks, we typically choose to use a parameter norm panalty $\Omega$ that penalized only the weights of the affine transformation at each layer and leaves the bias unregularized."

We do not regularized the bias:

1. The biases typically require less data to fit accurately than the weights.
2. Each bias controls only a single variable.
3. Regularizing the bias parameters can introduce a significant amount of underfitting.

#### $L^2$ Parameter Regularisation

Weight decay

the regularization term: $\Omega(\boldsymbol{\theta})=\frac{1}{2}\|\boldsymbol{w}\|_{2}^{2}$ added to the objective function 

Ridge Regression / Tikhonov regularization

Investigate the $L^2$ weight decay:

* We ignore the bias parameter:

  The objective function:
  $$
  \tilde{J}(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})=\frac{\alpha}{2} \boldsymbol{w}^{\top} \boldsymbol{w}+J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})
  $$

* The corresponding parameter gradient:
  $$
  \nabla_{\boldsymbol{w}} \tilde{J}(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})=\alpha \boldsymbol{w}+\nabla_{\boldsymbol{w}} J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})
  $$

The update of weights in teach single gradient step:
$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\epsilon\left(\alpha \boldsymbol{w}+\nabla_{\boldsymbol{w}} J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})\right)
$$
Also in the form
$$
\boldsymbol{w} \leftarrow(1-\epsilon \alpha) \boldsymbol{w}-\epsilon \nabla_{\boldsymbol{w}} J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})
$$
The weight decay term shrinks the weight vector by a constant factor on each step.

/ Further Investigation, start from the definition of argmin the objective function:

The value of the weights that obtains minimal unregularized training cost $\boldsymbol{w}^{*}=\arg \min _{\boldsymbol{n}} J(\boldsymbol{w})$. For the case of fitting a linear regression model with MSE, the objective function is truly quadratic. Our quadrtic approximation is perfect.



The approxiation $\hat{J}$ is
$$
\hat{J}(\boldsymbol{\theta})=J\left(\boldsymbol{w}^{*}\right)+\frac{1}{2}\left(\boldsymbol{w}-\boldsymbol{w}^{*}\right)^{\top} \boldsymbol{H}\left(\boldsymbol{w}-\boldsymbol{w}^{*}\right)
$$
$H$ is the *Hessian Matrix*? of $J$ respect to $w$ evaluated at $w^*$.

$w^*$ is the location of a minimum of $J$,  *we can conclude that $H$ is positive semidefinite* ?

The minimum of $\hat{J}$ occurs where its gradient 
$$
\nabla_{\boldsymbol{w}} \hat{J}(\boldsymbol{w})=\boldsymbol{H}\left(\boldsymbol{w}-\boldsymbol{w}^{*}\right)
$$
is equal to 0.

We add the weight decay gradient to it.
$$
\alpha \tilde{\boldsymbol{w}}+\boldsymbol{H}\left(\tilde{\boldsymbol{w}}-\boldsymbol{w}^{*}\right)=0
$$
$\tilde{w}$ is the location of the minimum.
$$
(\boldsymbol{H}+\alpha \boldsymbol{I}) \tilde{\boldsymbol{w}}=\boldsymbol{H} \boldsymbol{w}^{*}
$$

$$
\tilde{\boldsymbol{w}}=(\boldsymbol{H}+\alpha \boldsymbol{I})^{-1} \boldsymbol{H} \boldsymbol{w}^{*}
$$

$\alpha \rightarrow 0: \tilde{w} \rightarrow w^* $  

as $\alpha$ grows, because $H$ is real and symmetric, we can decompose it into a diagonal matrix $\boldsymbol{\Lambda}$ and an orthonormal basis of eigenvectors, $Q$, such that

$\boldsymbol{H}=\boldsymbol{Q} \mathbf{\Lambda} \boldsymbol{Q}^{\top}$ 

Thus we have
$$
\begin{aligned}
\tilde{\boldsymbol{w}} &=\left(\boldsymbol{Q} \mathbf{\Lambda} \boldsymbol{Q}^{\top}+\alpha \boldsymbol{I}\right)^{-1} \boldsymbol{Q} \boldsymbol{\Lambda} \boldsymbol{Q}^{\top} \boldsymbol{w}^{*} \\
&=\left[\boldsymbol{Q}(\boldsymbol{\Lambda}+\alpha \boldsymbol{I}) \boldsymbol{Q}^{\top}\right]^{-1} \boldsymbol{Q} \mathbf{\Lambda} \boldsymbol{Q}^{\top} \boldsymbol{w}^{*} \\
&=\boldsymbol{Q}(\boldsymbol{\Lambda}+\alpha \boldsymbol{I})^{-1} \boldsymbol{\Lambda} \boldsymbol{Q}^{\top} \boldsymbol{w}^{*}
\end{aligned}
$$
"the effect of weight decay is to rescale $w^*$ along the axes degined by the eigenvectors of $H$. Specifically, the component of $w^*$ that is aligned with the i-th eigenvector of $H$ is rescaled by a factor of $\frac{\lambda_{i}}{\lambda_{i}+\alpha}$   "?



"Only directions along which the parameters contribute significantly to reducing the objective function are preserved relatively intact."



For linear regression:



The cost function:
$$
(\boldsymbol{X} \boldsymbol{w}-\boldsymbol{y})^{\top}(\boldsymbol{X} \boldsymbol{w}-\boldsymbol{y})
$$
Add in $L^2$ regularisation:
$$
(\boldsymbol{X} \boldsymbol{w}-\boldsymbol{y})^{\top}(\boldsymbol{X} \boldsymbol{w}-\boldsymbol{y})+\frac{1}{2} \alpha \boldsymbol{w}^{\top} \boldsymbol{w}
$$
the regularised weight:
$$
\boldsymbol{w}=\left(\boldsymbol{X}^{\top} \boldsymbol{X}+\alpha \boldsymbol{I}\right)^{-1} \boldsymbol{X}^{\top} \boldsymbol{y}
$$
compared to the original weight:
$$
\boldsymbol{w}=\left(\boldsymbol{X}^{\top} \boldsymbol{X}\right)^{-1} \boldsymbol{X}^{\top} \boldsymbol{y}
$$
The new matrix is the original one plus $\alpha$ to the diagonal, which correspond to the variance of each input feature. $L^2$ regularisation causes the learning algorithm shirks the weights on features whose covariance with the output target is low compared to this added variance.