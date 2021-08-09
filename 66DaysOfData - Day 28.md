### 66DaysOfData - Day 28

# Convolutional Networks

## Deep Learning Chapter 9

3 special cases of the zero-padding:

1. The extreme case in which no zero-padding is used whatsoever, and the convolution kernel is only allowed to visit positions where the entire kernel is contained entirely within the image. - **valid convolution** - The size of the output shrinks at each layer, for image width $m$ and kernel has width $k$, the output will be of width $m-k+1$, which limits the number of convolutional layers.
2. Just enough zero-padding is added to keep the size of the output equal to the size of the input. - **same convolution** - The input pixels near the border influence fewer output pixels than the input pixels near the centre. This can make the border pixels somewhat underrepresented in the model.
3. **full convolution** - Enough zeroes are added for every pixel to be visited  $k$ times in each direction, resulting in an output image of width $m+k-1$. The output pixe;s near the border are a function of fewer pixels than the output pixels near the center. This can make it difficult to learn a single kernel that performs well at all positions in the convolutional feature map.

The optimal amount of zero padding lies between "valid" and "same" convolution.