### 66DaysOfData - Day 24

# Covolutional Networks

## Deep Learning Chapter 9

CNNs are a specialized kind of neural networks for processing data that has a known, grid like topology.

Convolution is a specialized kind of linear operation.

"Convolutional networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers"

Structure of Chapter:

What convolution is

$\downarrow$

the motivation behind using convolution in a neural network.

$\downarrow$

Describe pooling operation

$\downarrow$

Describe several variants on the convolution function that are widely used in practice for neural networks

$\downarrow$

How convolution may be applied to many kinds of data, with different numbers of dimensions

$\downarrow$

How to make convolution more efficient

$\downarrow$

Neuroscience principles in Deep Learning, and the role of CNN in the history of deep learning

### The Convolution Operation

Example of Convolution Operation:

$x(t)$ for the position of spacecraft at time $t$

We take the average of $x(t)$ to reduce noises at a given moment. Moreover, we take the weighted average to give higher weight to recent measurements.

weight function $w(a)$, a is the age of a measurement.

so the final estimate of position $s(t)$ is

$s(t)=\int x(a) w(t-a) d a$

The operation is convolution operation (in asterisk):

$s(t)=(x * w)(t)$

 $w$ - a valid probablity density function, be 0 for negative arguments.

$x(t)$ - input, $w$ - kernel

$s(t)$ - feature map

Discrete form:

$s(t)=(x * w)(t)=\sum_{a=-\infty}^{\infty} x(a) w(t-a)$

For input of more than 1 dimension (take 2 dimensions as an example):

Two-dimensional image $I$ - input

Two-dimensional kernel $K$

$S(i, j)=(I * K)(i, j)=\sum_{m} \sum_{n} I(m, n) K(i-m, j-n)$

**cross-correlation**? & Kernel flipping

"Many machine learning libraries implement cross-correlation but call it convolution"

"Any neural network algorithm that works with matrix mulitplication and does not depend on specific properties of the matrix structure should work with convolution, without requiring any further changes to the neural network."

![Screenshot 2021-08-03 at 13.17.53](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-03 at 13.17.53.png)

### Motivation

Convolution leverages three important ideas that can help improve a machine learning system:

1. Sparse Interactions
2. Parameter Sharing
3. Equivariant Representations

