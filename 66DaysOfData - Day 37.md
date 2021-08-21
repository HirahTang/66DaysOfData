### 66DaysOfData - Day 37

# Sequence Modeling: Recurrent and Recursive Nets

## Deep Learning Chapter 10

### Teacher Forcing and Networks with Output Recurrence

For the network with recurrent connections only from the output at one time step to the hidden units at the next time step, the structure is strictly less powerful. However, for loss function based on comparing the prediciton at time $t$ to the training target at time $t$, all the time steps are decoupled, which make the training be parallelized with the gradient for each step $t$ computed in isolation.

**teacher forcing**

"Models that have recurrent connections from their outputs leading back into the model may be triained with teacher forcing"

Teacher forcing is during thaining the model receives the ground truth output $y^{(t)}$ as input at time $t+1$. 

The conditional maximum likelihood criterion is

$\begin{aligned} & \log p\left(\boldsymbol{y}^{(1)}, \boldsymbol{y}^{(2)} \mid \boldsymbol{x}^{(1)}, \boldsymbol{x}^{(2)}\right) \\=& \log p\left(\boldsymbol{y}^{(2)} \mid \boldsymbol{y}^{(1)}, \boldsymbol{x}^{(1)}, \boldsymbol{x}^{(2)}\right)+\log p\left(\boldsymbol{y}^{(1)} \mid \boldsymbol{x}^{(1)}, \boldsymbol{x}^{(2)}\right) \end{aligned}$

Teacher forcing is motivated by allowing us to avoid back-propagation through time in models that lack hidden-to-hidden connections.

The disadvantages:

 If the network is going to be later used in an open-loop mode (the network outputs fed back as input). 

The inputs during training and test time can be quite different.

Solution:

Train with both teacher-forced inputs and with free-running inputs.

### Computing the Gradient in a Recurrent Neural Network

No specialized algorithm are necessary

