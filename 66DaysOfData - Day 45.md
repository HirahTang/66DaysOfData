### 66DaysOfData - Day 45

# Sequence Modeling: Recurrent and Recursive Nets

## Deep Learning Chapter 10

### Leaky Units and Other Strategies for Multiple Time Scales

Design a model that operates at multiple time scales, some parts of the model operate a fine-grained time scales and can handle small details, while other parts operate at coarse time scales and transfer information from the distant past to the present more efficiently.

Addition of skip connections across time, "leaky units" that integrate signals with different time constants, and the removal of some of the connections used to model fine-grained time scales

#### Adding Skip Connections through Time

Add direct connections from variables in the distant past to variables in the present.

introduce recurrent connections with a time-delay of $d$, so that the gradient diminish exponentially as a functino of $\frac{\tau}{d}$ 

#### Leaky Units and a Spectrum of Different Time Scales

Have units with linear self-connections and a weight near one on these connections.

We accumulate a running average $\mu^{(t)}$ of some value $v^{(t)}$ by applying the update $\mu^{(t)} \leftarrow \alpha \mu^{(t-1)}+(1-\alpha) v^{(t)}$

the $\alpha$ is an example of a linear self-connectino from $\mu^{(t-1)}$ to $\mu^{(t)}$ 

$\alpha$ close to 1, the information about the past for a long time remembered

$\alpha$ close to 0, the information about the past is rapidly discarded.

Hidden units with linear self-connections are called leaky units

The linear self-connections approach allows this effect to be adapted more smoothly and flexibly by adjusting the real-valued $\alpha$ rather than by adjusting the integer-valued skip length.

#### Removing Connections

Organizing the state of the RNN at multiple time-scales, with information flowing more easilt through long distances at the slower time scales.

It involves actively removing length-one connections and replacing them with longer connections. 

Units receiving such new connections may learn to operate on a long time scale but may also choose to focus on their other short-term connections.

### The Long Short-Term Memory and Other Gated RNNs

The most effective sequence models used in practical applicatoins are called gated RNNs.

Networks based on the gated recurrent units

Gated RNNs are based on the idea of creating paths through time that have derivatives that neither vanish nor explode. Gated RNNs generalise this to connection weights that may change at each time step.

We want a mechanism to forget the old state by setting the unit to zero, and we want the neural network to learn to decied when to do it.

