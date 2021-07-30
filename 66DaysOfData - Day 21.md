### 66DaysOfData - Day 21

# Optimization for Training Deep Models

## Deep Learning Chapter 8 

### Optimization Strategies and Meta-Algorithms

#### Batch Normalization



Batch Norm explanation by DeepLizard

https://www.youtube.com/watch?v=dXB-KQYkzNU



Batch Norm by Deep Learning Specialization:

C2W3L04-L07

https://www.youtube.com/watch?v=tNIpEZLv_eg

https://www.youtube.com/watch?v=em6dfRxYkYU

https://www.youtube.com/watch?v=nUUqwaxLnWs

https://www.youtube.com/watch?v=5qefnAek8OA



"It is actually not an optimization algorithm at all. Instead it is a method of adaptive reparametrization, motivated by the difficulty of training very deep models"

We assume the parameters will change without changing other layers, so we update all the layers simultaneously, which is not true.

-------

Take a deep neural network that has only one unit per layer and does not use an activation function at each hidden layer as an example:

$\hat{y}=x w_{1} w_{2} w_{3} \ldots w_{l}$

$w_{i}$ - the weight used by layer $i$

the output of layer $i$ $h_{i} = h_{i-1}w_{i}$

When the cost function put a gradient of 1 on $\hat{y}$, the back-propagation algorithm can then compute a gradient $\boldsymbol{g}=\nabla_{\boldsymbol{w}} \hat{y}$ 

When we make an update $\boldsymbol{w} \leftarrow \boldsymbol{w}-\epsilon \boldsymbol{g}$, The first-order Taylor series approximation of $\hat{y}$ will decrease by $\epsilon \boldsymbol{g}^{\top} \boldsymbol{g}$.

However, the actual update will include second-order and third-order effects, on up to effects of order $l$. The new value of $\hat{y}$ is 

$x\left(w_{1}-\epsilon g_{1}\right)\left(w_{2}-\epsilon g_{2}\right) \ldots\left(w_{l}-\epsilon g_{l}\right)$

-------

For example, one second-order term from the update - $\epsilon^{2} g_{1} g_{2} \prod_{i=3}^{l} w_{i}$

All these make it hard to choose an appropriate learning rate. 

Though for second-order optimization algorithms, even higher-order interactions can be significant for very deep networks.

-----

Batch Normalization provides an elegant way of reparametrizing almost any deep  network. 

"The reparametrization significantly reduces the problem of coordinating updates across many layers"

Batch Normalization can be applied to any input or hidden layer in a network.

Let $H$ be a minibatch of activations of the layer to normalize, arranged as a design matrix.

Normalize of $H$:

$\boldsymbol{H}^{\prime}=\frac{\boldsymbol{H}-\boldsymbol{\mu}}{\boldsymbol{\sigma}}$

$\mu$ - a vector containing the mean of each unit

$\sigma$ - a vector containing the standard deviatioin of each unit

$\boldsymbol{\mu}=\frac{1}{m} \sum_{i} \boldsymbol{H}_{i,:}$

$\boldsymbol{\sigma}=\sqrt{\delta+\frac{1}{m} \sum_{i}(\boldsymbol{H}-\boldsymbol{\mu})_{i}^{2}}$ $\delta$ is a small positive value (such as $10^{-8}$)

"We back-propagate through these operations for computing the mean and the standard deviation, and for applying them to normalize $H$. This means that the gradient will never propose an operation that acts simply to increase the standard deviation or mean of $h_{i}$; the normalization operations remove the effect of such an action and zero out its component in the gradient."

Batch normalization reparametrizes the model to make some units always be standardized by definition

Batch normalization acts to standardize only the mean and variance of each unit in order to stabilize learning, but allows the relationship between units and the non-linear statistics of a single unit to change.

In order to maintain the expressive power of the network, it is common to replace the batch of hidden unit activations $H$ with $\gamma \boldsymbol{H}^{\prime}+\boldsymbol{\beta}$ rather than simpy the normalized $\boldsymbol{H}^{\prime}$ 

"The mean is solely determined by $\beta$ using this approach", and the new parametrization is much easier to learn with gradient descent.

#### Coordinate Descent

Coordinate Descent - We minimize $f(x)$ with respect to a single variable $x_{i}$, then minimize it with respect to another variable $x_{j}$ and so, repeatedly cycling through all variables, we will arrive at a (local) minimum.

Block coordinate descent - minimize with respect to a subset of the variables simultaneously.

1. Different variables in the optimization problem can be clearly separated into groups that play relatively isolated roles,
2. Optimization with respect to one group of variables is significantly more efficient than optimization with respect to all of the variables.

Though coordinate decent is not a very good strategy when the value of one variable strongly influences the optimal value of another variable.