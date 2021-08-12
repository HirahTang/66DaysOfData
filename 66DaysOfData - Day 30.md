### 66DaysOfData - Day 30

# Convolutional Networks

## Deep Learning Chapter 9

### Variants of the Basic Convolution Function

Backpropagation of Convolutional Networks

The three operations - convolution, backdrop from output to weights, and backprop from output to inputs

The concept of autoencoder 

Bias for Conolutional networks

### Structured Outputs

"Convolutional networks can be used to output a high-dimensional, structured object" - A tensor, typically.

e.g. A tensor $S$, $S_{i,j,k}$ is the probablity that pixel (j,k) of the input to the network belongs to class $i$. This allows the model to label every pixel in an image and draw precise masks that follow the outlines of individual objects (target detect).

1. One can avoid pooling altogether if need to produce an output map of the similer size as the input.
2. Or simply emit a lower-resolution grid of labels.

**Recurrent Convolutional network**

Rather than outputting $\hat{Y}$ in a single shot, the recurrent network iteratively refines its estimate $\hat{Y}$ by using the previous estimate of $\hat{Y}$. The same parameters are used for each updated estimate.

### Data Types

One advantage to convolutioanl networks is that they can also process inputs with varying spatial extents.

The kernel is simply applied a different number of times depending on the size of the inut, and the ouptut of the convolution operation scales accordingly. 