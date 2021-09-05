### 66DaysOfData - Day 48

# Sequence Modeling: Recurrent and Recursive Nets

## Deep Learning Chapter 10

### Explicit Memory

Memory networks introduced to resolve the problem of traditional neural networks lack in the ability of store working memories.

**Neural Turing Machine**

able to learn from and write arbitrary content to memory cells with the use of a content-based soft attention mechanism.

Memory cell differs LSTMs and GRUs in the network ouptuts an internal state that chooses which cell to read from or write to.

The mechanism of Neural Turing Machine (NTU) read & write:

NTMs read to or write from many memory cells simoutaneously.

Read: Take a weighted average of many cells

Write: They modify multiple cells by different amounts.

The memory cells are typically augmented to contain a vector, rather than the single scaler stored by an LSTM or GRU memory cell. 

2 reasons to increas the size of the memory cell

1. We have increased the cost of accessing a memory cell
2. They allow for content based addressing, where the weight used to read to or write from a cell is a function of that cell.



<img src="/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-09-06 at 00.03.33.png" alt="Screenshot 2021-09-06 at 00.03.33" style="zoom:40%;" />



Whether it is soft or stochastic and hard, the mechanism for choosing an address is in its form identical to the attention mechanism.