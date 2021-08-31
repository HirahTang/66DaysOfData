### 66DaysOfData - Day 39

# Sequence Modeling: Recurrent and Recursive Nets

## Deep Learning Chapter 10

### Recurrent Networks as Directed Graphical Models

"Mean squared error is the cross-entropy loss associated with an output distribution that is a unit Gaussian, for example, just as with a feedforward network"

One way to interpret an RNN as a graphical model is to view the RNN as defining a graphical model whose structure is the complete graph, able to represent direct dependencies between any pair of y values.

When recurrent networks reduced to a directed graph, optimizing the parameters may be difficult.

Assumptions:

Parameter sharing: the same parameters can be used for different time steps.

The conditional probability distribution over the variables at time $t+1$ given the variables at time $t$ is stationary.

The mechanism of RNN to determine the length of the sequence.

1. Add a special symbol corresponding to the end of a sequence.
2. Introduce an extra Bernoulli output to the model that represents the decision to either continue generation or halt generation at each time step.
3. Add an extra output to the model that predicts the integer $\tau$ Itself. 

### Modeling Sequences Conditioned on Context with RNNs

Provide an extra input to an RNN:

1. as an extra input at each time step, or
2. as the initial state $h^{(0)}$, or
3. both

