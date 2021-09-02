### 66DaysOfData - Day 44

# Sequence Modeling: Recurrent and Recursive Nets

## Deep Learning Chapter 10

### Echo State Networks

ESNs and liquid state machines: Set the recurrent weights such that the recurrent hidden units do a good job of capturing the history of past inputs, and learn only the outputs weights.

**Reservoir Computing**

the hidden units form of reservoir to temporal features which may capture different aspects of the history of inputs.

They are similar to kernel machines: the arbitraru length sequence (input) $\rightarrow$ a fixed length vector (the recurrent state $h^{(t)}$) so that a linear predictor can be applied to solve the problem of interest.

Reservoir Computing views the recurrent net as a dynamical system, and set the input and recurrent weights such that the dynamical system is near the edge of stability.

The eigenvalue spectrum of the Jacobians $\boldsymbol{J}^{(t)}=\frac{\partial s^{(t)}}{\partial s^{(t-1)}}$

 **spectral radius of $J^{(t)}$** - the maximum of the absolte values of its eigenvalues

When we backpropagate a gradient vector through time

one step - $Jg$

n steps - $J^{n}g$

When we back-propagate a perturbed version of $g$ : $\boldsymbol{g}+\delta \boldsymbol{v}$ 

one step : $\boldsymbol{J}(\boldsymbol{g}+\delta \boldsymbol{v})$

n steps: $\boldsymbol{J}^{n}(\boldsymbol{g}+\delta \boldsymbol{v})$

The two back-propagation diverge by $\delta J^{n}v$ after n steps of back-propagation.

$v$ is a unit eigenvector of $J$ with eigenvalue $\lambda$, multiplicatio by the Jacob

ian simply scales the difference at each step.

Two executions of back-propagation are separated by a distance of $\delta|\lambda|^{n}$

When $v$ corresponds to the largest value of $|\lambda|$  , the perturbation achieves the widest possible separation of an initial perturbation of size $\delta$ 

The strategy of echo state networks is simply to fix the weights to have some spectral radius.