### 66DaysOfData - Day 25

# Convolutional Networks

## Deep Learning Chapter 9

### Motivation

Convolution leverages three important ideas that can help improve a machine learning system:

1. Sparse Interactions
2. Parameter Sharing
3. Equivariant Representations

**Sparse Interactions** (sparse connectivity, sparse weights)

by making the kernel smaller than the input.

**Parameter sharing**: using the same parameter for more than one function in a model.

In a convolutional neural net, each number of the kernel is used at every position of the input.

It reduces the runtime of forward propagation of the model to $k$ parameters

The particular form of parameter sharing causes the layer to have a property called **equivariance** to translation.

**Equivariance**: If the input changes, the output changes in the same way.

$f(x)$ is equivariance to a function $g$ if $f(g(x)) = g(f(x))$

This is useful when we know that some function of a small number of neighboring pixels is useful when applied to multiple input locations.



