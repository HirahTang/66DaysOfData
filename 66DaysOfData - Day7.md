### 66DaysOfData - Day7

# Regularization for Deep Learning

## Deep Learning Chapter 7

### 7.5 Noise Robustness

1. Adding to hidden units

"It is important to remember that noise injection can be much more powerful than simply shrinking the parameters, especially when the noise is added to the hidden units."

Noise applied to the hidden units - the dropout algorithm

2. adding noise to weights (used primarily in Recurrent Neural Networks)

"This can be interpreted as a stochastic implementation of Bayesian inference over the weights. The Bayesian treatment of learning would consider the model weights to be uncertain and representable via a probability distribution that reflects this uncertainty."?

Noise applied to the weights is quivalent to norm regularization.

#### Injecting Noise at the Output Targets

Label smoothing regularizes a model based on a softmax with k output values by replacing the hard 0 and 1 classification targets with targets of $\frac{\epsilon}{k-1}$ and $1-\epsilon$ 

### Semi-Supervised Learning

### Multi-Task Learning

Task-specific parameters (upper level networks)

Generic parameters, shared across all the tasks

![Screenshot 2021-07-07 at 12.02.01](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-07-07 at 12.02.01.png)

Assumption: Something shared across some of the tasks.

*among the factors that explain the variations observed in the dataassociated with the different tasks, some are shared across two or more tasks*

### Early Stopping

We return to the parameter setting at the point in time with the lowest validation set error. We return the parameters with the smallest validation error, rather the latest parameters.

It is an effective hyperparameter selection algorithm for selecting the appropriate training steps.

1.Train two time, the second is on all the training data (include validation data), with the training step from the first time.

2.Keep on training with validation set included, but keep the parameters obtained from the first round, instead of training from the scratch.

