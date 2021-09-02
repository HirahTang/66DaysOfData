### 66DaysOfData - Day 46

# Sequence Modeling: Recurrent and Recursive Nets

## Deep Learning Chapter 10

### LSTM

Introducing self-loops to produce paths where the gradient can flow for long durations is a core contribution of the initial long short-term memory (LSTM) model.

Make the weight on this self-loop conditioned on the context, rather than fixed.

By making the weight of this self-loop gated (controlled by another hidden unit), the time scale of integratnoi can be changed dynamically. 

For LSTM, the time scale of integration can change based on the input sequence. 

<img src="/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-09-02 at 15.14.57.png" alt="Screenshot 2021-09-02 at 15.14.57" style="zoom:50%;" />

Block diagram of the LSTM recurrent network "cell", replacing the usual hidden units of ordinary recurrent networks.

Input feature is computed with a regular artificial neuron unit.

Its value can accumulate into the state once the sigmoid input gate allows it.

A linear self-loop whose weight is controlled by the forget gate. 

The output of the cell can be shut off by the output gate.

LSTM recurrent networks have "LSTM cells" that have an internal recurrence (a self-loop)

Each cell has the same inputs and outpus as an ordinary recurrent network, aprt the outer recurrence of the RNN

It contains a system of gating units that controls the flow of information.

The state unit $s_{i}^{(t)}$ has a linear self-loop similar to the leaky units. The self-loop weight is controlled by a forget gate unit $f_{i}^{(t)}$, 

 $f_{i}^{(t)}=\sigma\left(b_{i}^{f}+\sum_{j} U_{i, j}^{f} x_{j}^{(t)}+\sum_{j} W_{i, j}^{f} h_{j}^{(t-1)}\right)$

$x^{(t)}$ - the current input vector

$h^{(t)}$ - the current hidden layer vector

$b^{f}$ - bias for the forget gates

$U^{f}$ - input weights for the forget gates

$W^{f}$ - recurrent weights for the forget gates

The update of the LSTM cell internet state:

$s_{i}^{(t)}=f_{i}^{(t)} s_{i}^{(t-1)}+g_{i}^{(t)} \sigma\left(b_{i}+\sum_{j} U_{i, j} x_{j}^{(t)}+\sum_{j} W_{i, j} h_{j}^{(t-1)}\right)$

$b$ - bias, $U$ - input weights, $W$ - recurrent weights

The external input gate unit $g_{i}^{(t)}$ is computed similarly but with its own parameters:

$g_{i}^{(t)}=\sigma\left(b_{i}^{g}+\sum_{j} U_{i, j}^{g} x_{j}^{(t)}+\sum_{j} W_{i, j}^{g} h_{j}^{(t-1)}\right)$

The output $h_{i}^{(t)}$ can also be shut off, via the output gate $q_{i}^{(t)}$

$h_{i}^{(t)}=\tanh \left(s_{i}^{(t)}\right) q_{i}^{(t)}$

$q_{i}^{(t)}=\sigma\left(b_{i}^{o}+\sum_{j} U_{i, j}^{o} x_{j}^{(t)}+\sum_{j} W_{i, j}^{o} h_{j}^{(t-1)}\right)$

$b^{o}$  - bias, $U^{o}$ - input weights, $W^{o}$ - recurrent weights

The cell state $s_{i}^{(t)}$ can also be used as an extra input (with weights) into the three gates of the i-th unit.

### Other Gated RNNs

Gated Recurrent units (GRUs)

A single gating unit simultaneously controls the forgetting factor and the decision to update the state unit.

The update equations:

$h_{i}^{(t)}=u_{i}^{(t-1)} h_{i}^{(t-1)}+\left(1-u_{i}^{(t-1)}\right) \sigma\left(b_{i}+\sum_{j} U_{i, j} x_{j}^{(t-1)}+\sum_{j} W_{i, j} r_{j}^{(t-1)} h_{j}^{(t-1)}\right)$

$u$ - update gate

$u_{i}^{(t)}=\sigma\left(b_{i}^{u}+\sum_{j} U_{i, j}^{u} x_{j}^{(t)}+\sum_{j} W_{i, j}^{u} h_{j}^{(t)}\right)$

$r$ - reset gate

$r_{i}^{(t)}=\sigma\left(b_{i}^{r}+\sum_{j} U_{i, j}^{r} x_{j}^{(t)}+\sum_{j} W_{i, j}^{r} h_{j}^{(t)}\right)$

The update gates like conditional leaky integrators that can linearly gate any dimension, then to decide whether to copy it or ignore it.

The reset gates control which parts of the state get used to compute the next target state.



