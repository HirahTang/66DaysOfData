### 66DaysOfData - Day 13

# Optimization for Training Deep Models

## Deep Learning Chapter 8

### Challenges in Neural Network Optimization

### Inexact Gradients

These issues mostly arise with the more advanced models.

#### Poor Correspondence between Local and Global Structure

<img src="/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-07-15 at 12.52.46.png" alt="Screenshot 2021-07-15 at 12.52.46" style="zoom:33%;" />

"Many existing research directions are aimed at fniding good initial points for problems that have difficult global structure"

### Basic Algorithms

#### Stochastic Gradient Descent

![Screenshot 2021-07-15 at 14.44.17](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-07-15 at 14.44.17.png)

In practice, we decrease the learning rate over time. For Batch gradient descent(deterministic gradient descent), the true gradient of the total cost functino becomes small and then 0, thus we can use a fixed learning rate.

In practice, it is common to decay the learning rate linearly until iteration $\tau$

$\epsilon_{k}=(1-\alpha) \epsilon_{0}+\alpha \epsilon_{\tau}$

with $\alpha=\frac{k}{\tau}$ , After iteration $\tau$, we leave $\epsilon$ constant

<img src="/Users/hirahtang/Library/Application Support/typora-user-images/image-20210715152458749.png" alt="image-20210715152458749" style="zoom:33%;" />

$\tau$ is set to the number of iterations required to make a few hundred passes through the training set. $\epsilon_{\tau}$ shoudl be set to roughly 1% the value of $\epsilon_{0}$ 

The setting of $\epsilon_{0}$ is the real problem.

"Typically, the optimal initial learning rate, in terms of total training time and the final cost value, is higher than the learning rate that yields the best performance after the first 100 iterations or so".

We monitor the first several iterations and use a learning rate that is higher than it, but not to high that may cause severe instability.

The computational time of SGD does not increase correspond to the increase of data size. SGD converges before proceeding the entire training set is possible.

The rest of this section is about the discussion of optimization speed.