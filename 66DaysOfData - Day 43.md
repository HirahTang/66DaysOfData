### 66DaysOfData - Day 43

# Sequence Modeling: Recurrent and Recursive Nets

## Deep Learning Chapter 10

### Recursive Neural Networks

Recursive neural networks has a different kind of computational graph, which is structured as a deep tree. 

<img src="/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-30 at 16.54.28.png" alt="Screenshot 2021-08-30 at 16.54.28" style="zoom:40%;" />

A variable size sequence can be mapped to a fixed size representation (o) with a fixed set of parameters

Advantage: for a sequence of the same length $\tau$, the depth can be reduced from $\tau$ To $O(\log \tau)$ 

For choice of the best structure of the tree:

1. A tree structure which does not depend on the data (balanced binary tree)
2. external methods can suggest the appropriate tree structure - the structure can be fixed to the parse tree of the sentence in NLP tasks
3. Ideally, the learner itself would discover and infer the tree structure.

Other variants of the recursive net idea are possible

### The Challenge of Long-Term Dependencies

Gradients propagated over many stages tend to either vanish or explode.

For long-term dependencies, the long-term interactions are given to exponentially smaller weights.

The compositino of the same function one per time step can result in extremely nonlinear behavior.

The problem is particular to recurrent networks.

In order to store memories in a way that is robust to small perturbations, the RNN must enter a region of parameter space where gradients vanish. Specifically, whenever the model is able to represent long term dependencies, the gradient of a long term interaction has exponentially smaller magnitude than the gradient of a short term interaction.

The coming sections discuss various approaches that have been proposed to reduce the difficulty of learning long-term dependencies.
