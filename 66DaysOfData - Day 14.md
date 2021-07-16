### 66DaysOfData - Day 14

# Optimization for Training Deep Models

## Deep Learning Chapter 8

### Basic Algorithms

#### Momentum

The method of momentum is designed to accelerate learning, especially in the face of high curvature, small but consistent gradients, or noisy gradients.

$v$ for velocity - the direction and speed at which the parameters move through parameter space. (A exponentially decaying average of the negative gradient)

Unit mass is assumed, thus the velocity vector $v$ is also regarded as the momentum of the particle.

Hyperparameter $\alpha \in[0,1)$ determines how quickly the contributions of previous gradients exponentially decay
$$
\boldsymbol{v} \leftarrow \alpha \boldsymbol{v}-\epsilon \nabla_{\boldsymbol{\theta}}\left(\frac{1}{m} \sum_{i=1}^{m} L\left(\boldsymbol{f}\left(\boldsymbol{x}^{(i)} ; \boldsymbol{\theta}\right), \boldsymbol{y}^{(i)}\right)\right)
$$

$$
\theta \leftarrow \theta + v
$$

$v$ accumulates the gradient elements, $\alpha$ relates to the affect of previous gradients to the current direction, compare to $\epsilon$

Momentum aims to solve two problems: Poor conditioning of the Hessian matrix, and variance in the stochastic gradient.

With momentum, the size of the step depends on how large and how aligned a *sequence* of gradients are. The step size is largest when many successive gradients point in exactly the same direction.

If the algorithm always observes gradient $g$, it will accelarte in the direction of $-g$ , until reach a terminal velocity, determined by $\epsilon$ and $\alpha$.

$\frac{\epsilon|| \boldsymbol{g}||}{1-\alpha}$ 

![Screenshot 2021-07-16 at 14.17.25](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-07-16 at 14.17.25.png)

#### Nesterov Momentum

The updated rules:

$\boldsymbol{v} \leftarrow \alpha \boldsymbol{v}-\epsilon \nabla_{\boldsymbol{\theta}}\left[\frac{1}{m} \sum_{i=1}^{m} L\left(\boldsymbol{f}\left(\boldsymbol{x}^{(i)} ; \boldsymbol{\theta}+\alpha \boldsymbol{v}\right), \boldsymbol{y}^{(i)}\right)\right]$
$$
\theta \leftarrow \theta + v
$$

The difference is where the gradient is evaluated.

"With Nesterov momentum the gradient is evaluated after the current velocity is applied"

"One can interpret Nesterov momentum as atttempying to add a correction factor to the standard method of momentum"	![Screenshot 2021-07-16 at 15.07.41](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-07-16 at 15.07.41.png)

