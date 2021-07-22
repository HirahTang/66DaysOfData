### 66DaysOfData - Day 16

# Optimization for Training Deep Models

## Deep Learning Chapter 8

### Parameter Initialization Strategies

In practice, we treat the scale of the weights as hyperparameter whose optimal value lies somewhere roughly near but not exactly equal to the theoretical predictions

*Sparse initialization*

One can manually search for the best initial scales.

Situations where we set biases to non-zero values:

??

### Algorithms with Adaptive Learning Rates

The momentum algorithm can mitigate the problem of choosing learning rate, at the expense of introducing another hyperparameter.

It can make sense to use a separate learning rate for each parameter, and automatically adapt these learning rates throughout the course of learning.

#### AdaGrad

Adapts the learning rates of all model parameters by scaling them inversely proportional to the square root of the sum of all their historical squared values.

Parameters with large partial derivative of the loss have more rapid decrease in their learning rate.

For Convex Optimization, it works well.

For training deep neural network models: The accumulation of squared gradients from the beginning of training can result in a premature and excessive decrease in the effective learning rate.

![Screenshot 2021-07-19 at 22.03.35](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-07-19 at 22.03.35.png)

