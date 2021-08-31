### 66DaysOfData - Day 42

# Sequence Modeling: Recurrent and Recursive Nets

## Deep Learning Chapter 10

### Encoder-Decoder Sequence-to-Sequence Architectures

RNN can map an input sequence to an output sequence of the same length.

RNN can also be trained to map an input sequence to an output sequence which is not necessarily of the same length.

Input to the RNN - the "context"

We produce a representation of the context - C, which would be a vector or sequence of vectors that summarise the input sequence $\boldsymbol{X}=\left(\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{\left(n_{x}\right)}\right)$

![Screenshot 2021-08-28 at 16.57.14](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-28 at 16.57.14.png)

Encoder (reader, input RNN) processes the input sequence.

Decoder (writer, output RNN) is conditioned on the fixed-length vector to generate the output sequence $\boldsymbol{Y}=\left(\boldsymbol{y}^{(1)}, \ldots, \boldsymbol{y}^{\left(n_{y}\right)}\right)$

For a sequence-to-sequence architecture, the two RNNs are trained jointly to maximise the average of $\log P\left(\boldsymbol{y}^{(1)}, \ldots, \boldsymbol{y}^{\left(n_{y}\right)} \mid \boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{\left(n_{x}\right)}\right)$ over all the pairs of $x$ and $y$ sequences in the training set.

### Deep Recurrent Networks

![Screenshot 2021-08-29 at 22.03.56](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-29 at 22.03.56.png)

1. from the input to the hidden state
2. from the previous hidden state to the next hidden state
3. from the hidden state to the output

Experimental evidence strongly suggests to introduce depth in each of these operations

<img src="/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-29 at 22.24.29.png" alt="Screenshot 2021-08-29 at 22.24.29" style="zoom:40%;" />

The hidden recurrent state can be broken down into groups organized hierarchically. Transform the raw input into a representation that is more appropriate, at the higher levels of the hidden state.

<img src="/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-29 at 22.28.52.png" alt="Screenshot 2021-08-29 at 22.28.52" style="zoom:50%;" />

Deeper computation can be introduced in the input-to-hidde, hidden-to-hidden and hidden-to-output parts. There can be seperate MLP for each of the three blocks above. But this makes the training much more difficult.



