### 66DaysOfData - Day 36

# Sequence Modeling: Recurrent and Recursive Nets

## Deep Learning Chapter 10

### Unfolding Computational Graphs

The classical form of a dynamical system:

$\boldsymbol{s}^{(t)}=f\left(\boldsymbol{s}^{(t-1)} ; \boldsymbol{\theta}\right)$

$s^{(t)}$ is the state of the system

The unfolded computational graph

![Screenshot 2021-08-19 at 13.22.39](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-19 at 13.22.39.png)

For a dynamical system driven by an external signal $x^{(t)}$

$\boldsymbol{s}^{(t)}=f\left(\boldsymbol{s}^{(t-1)}, \boldsymbol{x}^{(t)} ; \boldsymbol{\theta}\right)$

Many recurrent neural networks use the function

![Screenshot 2021-08-19 at 13.39.28](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-19 at 13.39.28.png)

to define their hidden units of the network

### Recurrent Neural Networks

The two main ideas for designing a wide variety of recurrent neural networks:

Graph unrolling & parameter sharing ideas

1."Recurrent networks that produce an output at each time step and have recurrent connections between hidden units"

![Screenshot 2021-08-19 at 15.03.28](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-19 at 15.03.28.png)

2."Recurrent networks that produce an output at each time step and have recurrent connections only from the output at one time step to the hidden units at the next time step"

![Screenshot 2021-08-19 at 15.09.50](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-19 at 15.09.50.png)

3."Recurrent networks with recurrent connections between hidden units, that read an entire sequence and then produce a single output"

![Screenshot 2021-08-19 at 15.14.46](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-19 at 15.14.46.png)

Computing of loss for RNNs

**back-propagation through time** (BPTT)