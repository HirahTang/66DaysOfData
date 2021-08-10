### 66DaysOfData - Day 29

# Convolutional Networks

## Deep Learning Chapter 9

**Unshared Convolution** - local connections without shared convolution

Local connections, convolution, and full connections

![Screenshot 2021-08-10 at 14.22.55](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-10 at 14.22.55.png)

The difference of local connections and convolutions is in convolutions share parameters. Though their connectivities are the same.

Locally connected layers are useful for cases which we know that each feature should be a function of a small part of space, but there is no reason to think that the same feature should occur across all of space.

**Connectivity further restriced Locally connected layers** e.g. to constrain each output channel $i$ to be a function of only a subset of the input channels $l$.

![Screenshot 2021-08-10 at 22.20.27](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-10 at 22.20.27.png)

**Tiled convolution**

Offers a compromise between a convolutional layer and a locally connected layer. We learn a set of kernels that we rotate through as we move through space.

Immediately neighboring locations have different filters,  the memory requirements for storing the parameters increase only by a factor of the size of this set of kernels.

Compute the gradient with respect to the kernel during learning.

