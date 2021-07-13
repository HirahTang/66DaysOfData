### 66DaysOfData - Day 11

# Optimization for Training Deep Models

## Deep Learning Chapter 8

The structure of this Chapter:

Optimization for training machine learning tasks VS. Pure optimization

​                                                    $\downarrow$

Concrete challenges that make optimization of neural networks difficult

​                                                    $\downarrow$

Define several algorithms (optimization algorithms & initializing parameters strategies)

​                                                    $\downarrow$

More advanced algorithms, adapt learning rates during training / leverage information contained in the second derivatives of the cost function

​                                                    $\downarrow$

Review and combing them into high-level procedures.

### How Learning Differs from Pure Opimization

For machine learning, we optimize the performance measure P, indirectly.

We reduce cost function $J(\theta)$ in the hope that doing so will improve P

The cost function can be written as an average over the training set.

$J(\boldsymbol{\theta})=\mathbb{E}_{(\boldsymbol{x}, \mathrm{y}) \sim \hat{p}_{\mathrm{data}}} L(f(\boldsymbol{x} ; \boldsymbol{\theta}), y)$

$L$ - per-sample loss function

$f(x;\theta)$ - predicted output for x

$\hat{p}_{data}$ - empirical distribution

y - target output

We would prefer to minimize the objective function where the expectation is taken across the data generating distribution $p_{data}$:

$J^{*}(\boldsymbol{\theta})=\mathbb{E}_{(\boldsymbol{x}, \mathrm{y}) \sim p_{\text {data }}} L(f(\boldsymbol{x} ; \boldsymbol{\theta}), y)$

#### Empirical Risk Minimization

$J^{*}(\boldsymbol{\theta})=\mathbb{E}_{(\boldsymbol{x}, \mathrm{y}) \sim p_{\text {data }}} L(f(\boldsymbol{x} ; \boldsymbol{\theta}), y)$ is known as **risk** (the expected generalization error)

We do not know $p_{data}(x, y)$, but we have a training set of samples.

The simplest way, we minimize the expected loss on the training set. It

 means replacing the true distribution $p(x,y)$ with the empirical distribution $\hat{p}(x,y)$ (The distribution in the training set).

$\mathbb{E}_{\boldsymbol{x}, \mathrm{y} \sim \hat{p}_{\mathrm{data}}(\boldsymbol{x}, y)}[L(f(\boldsymbol{x} ; \boldsymbol{\theta}), y)]=\frac{1}{m} \sum_{i=1}^{m} L\left(f\left(\boldsymbol{x}^{(i)} ; \boldsymbol{\theta}\right), y^{(i)}\right)$

m - the number of training examples.

This is **Empirical Risk Minimization**

Problems of Empirical Risk Minimization:

1. Prone to overfitting - especially for models with high capacity.
2. For gradient descent (the most effective modern optimization) - many useful loss functions (0-1 loss) have no useful derivatives

Thus we rarely use empirical risk minimization for deep learning.

#### Surrogate Loss Functions and Early Stopping

We optimise tbe surrogate loss function when the loss function we actually care about is intractable.

e.g. for 0-1 loss, we take the negative log-liklihood instead as the surrogate.

Optimization for training algorithms usually do not halt at a local minimum. As we use the true underlying loss function for validation, we introduce early stopping techniques. Sometimes early stopping calls the algorithms to stop as overfitting occurs, but the surrogate loss function can still have large derivatives.

#### Batch and Minibatch Algorithms

Using a subset of the terms of the full cost function

e.g. 

Maximum Likelihood estimation: 

$\boldsymbol{\theta}_{\mathrm{ML}}=\underset{\boldsymbol{\theta}}{\arg \max } \sum_{i=1}^{m} \log p_{\text {model }}\left(\boldsymbol{x}^{(i)}, y^{(i)} ; \boldsymbol{\theta}\right)$

maximising it is equivalent to maximising the expectation over the empirical distribution defined by training set:

$J(\boldsymbol{\theta})=\mathbb{E}_{\mathbf{x}, \mathrm{y} \sim \hat{p}_{\text {data }}} \log p_{\text {model }}(\boldsymbol{x}, y ; \boldsymbol{\theta})$

In practice, we compute the expectations by randomly sampling a small number of examples from the dataset, then taking the average over only those examples.

Reasons:

1.Standard error of the mean estimated from n samples: $\sigma / \sqrt{n}$

As we increase n, the standard error of the mean estimated only decrease in $\sqrt{n}$, but we can make the optimization converge much faster for smaller n.

2. The redundancy in the training set. We may fund large numbers of examples that all make very similar contributions to the gradient.

Using the entire training set for gradient descent - **batch gradient descent** / **deternministic gradient descent** 

Use only a single example at a time - stochastic / online methods

In between - mini batch / minibatch stochastic methods $\rightarrow$ stochastic methods

e.g. Staochastic Gradient Descent

