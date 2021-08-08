### 66DaysOfData - Day 27

# Convolutional Networks

## Deep Learning Chapter 9

### Variants of the Basic Convolution Function

Multi-channel convolution

Assume a 4-D kernel tensor $K$ with element $K_{i,j,k,l}$ giving the connection strength between a unit in channel $i$ of the output and a unit in channel $j$ of the input. With an offset of $k$ rows and $l$ columns between the output unit and the input unit.

Input - $V$ with element $V_{i, j,k}$ giving the value of the input unit within channel $i$ at row $j$ and column $k$

Output - $Z$ with the same format as $V$

Z is produced by convolving $K$ across $V$ without flipping $K$ then,

$Z_{i, j, k}=\sum_{l, m, n} V_{l, j+m-1, k+n-1} K_{i, l, m, n}$

If we want to sample only every $s$ pixels in each direction in the output.

$Z_{i, j, k}=c(\mathbf{K}, \mathbf{V}, s)_{i, j, k}=\sum_{l, m, n}\left[V_{l,(j-1) \times s+m,(k-1) \times s+n} K_{i, l, m, n}\right]$

$s$ - stride

Zero-padding

without which the representation shrinks by one pixel less then the kernel width at each layer.

