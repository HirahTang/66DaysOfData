### 66DaysOfData - Day 15

# Optimization for Training Deep Models

## Deep Learning Chapter 8

### Parameter Initialization Strategies

Training algorithms for deep learning models is a sufficiently difficult task that most algorithms are strongly affected by the choice of initialization.

Initial parameter need to "break symmetry" between differOnt units.

If two hidden units with the same activation function are connected to the same inputs, then these units must have different initial parameters.

"Typically, we set the bias for each unit to heuristically chosen constants. As well as to extra parameters (such as paraemters encoding the conditional variance of a predictino)"

The perspective of regularization and optimization give different insights into how we should initialize a network. One aim the weight to be smaller, while the other suggests that the weights should be large enough to propagate information successfully.

**Heuristic for choosing the initial scale of the weights**

Initialize the weights of a fully connected layer with $m$ inputs and $n$ outputs by sampling each weight from $U\left(-\frac{1}{\sqrt{m}}, \frac{1}{\sqrt{m}}\right)$ 

Normalized initialization is also suggested

$\mathrm{W}_{i, j} \sim U\left(-\sqrt{\frac{6}{m+n}}, \sqrt{\frac{6}{m+n}}\right)$

Random orthogonal matrices.

