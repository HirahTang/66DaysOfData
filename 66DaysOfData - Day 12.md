### 66DaysOfData - Day 12

# Optimization for Training Deep Models

## Deep Learning Chapter 8

#### Batch and MiniBatch algorithms 

Different kinds of algorithms use different kinds of information from the mini-batch in different ways.

Methods that compute updates based only on the gradient $g$ are more robust and can handle smaller batch size (100).

Second order methods, which also use Hessian Matrix $H$ and compute updates such as $H^{-1}g$ require larger batch sizes (10,000). Which is required to minimize fluctuations in the estimates of $H^{-1}g$.

##### Minibatch shall be selected randomly

##### Two subsequent gradient estimates to be independent to each other, thus examples should be independent from each other.

### Challenges in Neural Network Optimization

#### Ill-Conditioning

Ill-conditioning can manifest by causing SGD to get "stuck" in the sense that even very small steps increase the cost function.

A gradient descent step of $-\epsilon g$ will add $\frac{1}{2} \epsilon^{2} \boldsymbol{g}^{\top} \boldsymbol{H} \boldsymbol{g}-\epsilon \boldsymbol{g}^{\top} \boldsymbol{g}$ to the cost.

Ill-conditioning of the gradient becomes a problem when $\frac{1}{2} \epsilon^{2} \boldsymbol{g}^{\top} \boldsymbol{H g}$ > $\epsilon \boldsymbol{g}^{\top} \boldsymbol{g}$

#### Local Minima

Neural networks and any models with multiple equivalently parametrized latent variables all have multiple local minima because of the **model identifiability problem** 

Model with latent variables are often not identifiable because we obtain equivalent models by exchanging latent variables with each other.

We can swap the weight vectors (arrange the hidden units) - **weight space symmetry**

Local minima pose problems only when they have high cost in comparison to the global minimum.

#### Plateaus, Saddle Points and Other Flat Regions

Saddle point -  the Hessian matrix has both positive and negative eigenvalues.

Low dimensional space: local minima are common

HIgh-dimensional space: local minima are rare and saddle points are more common

For Hessian matrices of high dimensions, the chance of getting mixed value (positive and negative and the same time) eigenvalues is higher.

Local minima are much more likely to have low cost. Critical points with high cost are more likely to be saddle points.

Saddle points constitute a problem for Newton's method, but not for gradient descent. (It finds zero gradient points, which leads it to saddle points, possibly)

Plateaus, wide, flat regions of constant value. (the gradient and the Hessain are all zero) is still problematic.

#### Cliffs and Exploding Gradients

Cliffs - extremly steep regions. Result from the multiplication of several large weights together, happens to NN with many layers.

It can be avoided using the **gradient clipping**: The graident does not sepcify the optimal step size, but only the optimal direction. This technique intervenes when gradient descent is making a very large step.

Cliffs are most common in the cost functions for recurrent neural networks.

#### Long-Term Dependencies

When the graph becomes extremely deep.

The Vanishinng and exploding gradient problem refer to the fact that gradients through such a deep graph are also scaled according to $diag(\lambda)^{t}$ (t is the number of layer, and $\lambda$ is the eigenvalue of matrix $W$)

Deep feedforward networks do not use the same matrix $W$ at each time step (in contrary to recurrent networks), which makes it can largely avoid the vanishing and exploding gradient problem.

