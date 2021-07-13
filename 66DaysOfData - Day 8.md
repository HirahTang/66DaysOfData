### 66DaysOfData - Day 8

# Regularization For Deep Learning

## Deep Learning Chapter 7

### Parameter Tying and Parameter Sharing

force sets of parameters to be equal - parameter sharing

Advantage: only a subset of the parameters need to be stored in memory.

used in CNN, parameter shared across multiple image locations.

### Sparse Representations

Discussed multiple approaches to reach representational sparsity.

* $L^1$ penalty
* Penalty derived from a Student-t prior on the representation
* KL divergence penalty

Hard constraint on the activation values:

orthogonal matching pursuit

### Bagging and Other Ensemble Methods

Ensemble Methods: 

"In the case where the errors are perfectly correlated and c = v, the mean squared error reduces to v, so the model averaging does not help at all. In the case where the errors are perfectly uncorrelated and c = 0, the expected squared error of the ensemble is only $\frac{1}{k} v$. This means that the expected squared error of the ensemble decreases linearly with the ensemble size. In other words, on average, the ensemble
will perform at least as well as any of its members, and if the members make independent errors, the ensemble will perform significantly better than its members."

Different in random initialisation, random selection of mini batches, differences in hyprtparamters or different outcomes of non-deterinistic implementations of neural networks are often enough to cause differnet members of the ensemble to make partially independent errors.

### Dropout

Advantages: 

1. Computatioinally Cheap.
2. Does not significantly limit the type of model or training procedure that can be used.

But we require larger model and more data.