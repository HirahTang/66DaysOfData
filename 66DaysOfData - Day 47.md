### 66DaysOfData - Day 47

# Sequence Modeling: Recurrent and Recursive Nets

## Deep Learning Chapter 10

### Optimization for Long-Term Dependencies

Clipping Gradients

The update must be chosen to be small enough to avoid traversing too much upward curvature.

Gradient Descent with gradient clipping has a more moderate reaction to the cliff

<img src="/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-09-04 at 00.51.26.png" alt="Screenshot 2021-09-04 at 00.51.26" style="zoom: 50%;" />

**Clipping the gradient**

1. clip the parameter gradient from a minibatch element-wise just before the parameter update.
2. Clip the norm $\|\boldsymbol{g}\|$ of the gradient $g$ just before the parameter update.

![](https://cdn.mathpix.com/snip/images/EEMnd516n4lmCQWsYpu-_0yqv-XcKP3m3hYAkeq6zbU.original.fullsize.png)

$v$ - the norm threshold

$g$ - used to update parameters

"Traditional stochastic gradient descent uses an unbiased estimate of the gradient, while gradient descent with norm clipping introduces a heuristic bias that we know emirically to be useful."

### Regularizing to Encourage Information Flow

Another idea of addressig vanishing gradients and better capture long-term dependecies is to regularize or constrain the parameters so as to encourage "information flow."

We would like the gradient vector $\nabla_{\boldsymbol{h}^{(t)}} L$ being back-propagated to maintain its magnitude, even if the loss function only penalizes the output at the end of the sequence.

We want $\left(\nabla_{\boldsymbol{h}^{(t)}} L\right) \frac{\partial \boldsymbol{h}^{(t)}}{\partial \boldsymbol{h}^{(t-1)}}$ to be as large as $\nabla_{\boldsymbol{h}^{(t)}} L$

We propose the regularizer

$\Omega=\sum_{t}\left(\frac{||\left(\nabla_{\boldsymbol{h}^{(t)}} L\right) \frac{\partial \boldsymbol{h}^{(t)}}{\partial \boldsymbol{h}^{(t-1)}}||}{\| \nabla_{\boldsymbol{h}^{(t)}} L||}-1\right)^{2}$

There is also an approximation in which we consider the back-propagated vector  $\nabla_{\boldsymbol{h}^{(t)}} L$ as if they were constants.

It is not as effective as the LSTM for tasks where data is abundant (language modeling)

 