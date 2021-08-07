### 66DaysOfData - Day 26

# Convolutional Networks

## Deep Learning Chapter 9

### Pooling

Typical layer of a convolutional network

 Convolutions in parrallel to produce a set of linear activations $\rightarrow$  Linear activation is run through a nonlinear activation function (ReLU) (detector stage) $\rightarrow$ A pooling function to modify the output of the layer further

"A pooling function replaces the output of the net at a certain location with a summary statistic of the nearby outputs"

Max pooling - reports the maximum output within a rectangular neighbourhood.

Average pooling, $L^{2}$ norm of a rectangular neighbourhood, a weighter average based on the distance from the center pixel.

Pooling makes the representations invariant to small translations of the input.

"Invariance to local translation can be a very useful property if we care more about whether some feature is present than exactly where it is"

But if we pool over the outputs of separately parmetrized convolutions, the features can learn with transformations to become invariant to.

A pooling unit that pools over multiple features that are learned with separate parameters can learn to be invariant to transformations of the input.

There can be fewer pooling units than detector units

Pooling is essential for handling inputs of varying size.

### Convolution and Pooling as an Infinitely Strong Prior

An infinitely strong prior places zero probability on smoe parameters and says that these parameter values are completely forbidden, regardless of how much support the data gives to those values.

The infinitely strong prior says that the weights for one hidden unit must be identical to the weights o its neighbour, but shifted in space, for a convolutional net to a fully connected net.

Thinking of a convolutional net as a fully connected net with an infinitely strong prior is computationally wasteful, but can give us some insights.

1.Convolution and pooling can cause underfitting. It's not accurate for pooling on all features if the task needs to preserve precise spatial information.

2. We should only compare convolutional models to other convolutional models in benchmarks of statistical learning performance.