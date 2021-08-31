### 66DaysOfData - Day 40

# Sequence Modeling: Recurrent and Recursive Nets

## Deep Learning Chapter 10

### Modeling Sequences Conditioned on Context with RNNs

Add additional input x to the model

The extra input x is added at each time step:

![Screenshot 2021-08-24 at 18.59.16](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-24 at 18.59.16.png)

An RNN that maps a fixed-length vector $x$ into a distribution over sequences $Y$.

"We can think of the choice of $x$ as determining the value of $\boldsymbol{x}^{\top} \boldsymbol{R}$ that is effectively a new bias parameter used for each of the hidden units."

We add connections from the output at time $t$ To the hidden unit at time $t+1$, to remove the conditional independence assumption.

![Screenshot 2021-08-25 at 16.30.04](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-25 at 16.30.04.png)

The model can then represent arbitrary probability distributions over the $y$​ sequence.

## Bidirectional RNNs

For applications we want to output a prediciton of $y^{(t)}$ which may depend on the whole input sequence.

Bidirectional Recurrent Neural Networks were invented to address that need.

![Screenshot 2021-08-25 at 16.42.28](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-25 at 16.42.28.png)

The $h$​ Recurrence propagates information forward in time (towards the right) while the $g$ recurrence propagates information backward in time (towards the left)

The idea can be extended to 2-d input (images), by having four RNNs, each one going in one of the four directions: up, down, left, right.