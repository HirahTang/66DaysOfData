### 66DaysOfData - Day6

# Numerical Computation

## Deep Learning Chapter 4

### The final section of the Chapter 4 - 4.5 Example: Linear Least Squares

# Regularization for Deep Learning

## Deep Learning Chapter 7

### 7.3 Regularization and Under-Constrained Problems

$\boldsymbol{X}^{\top} \boldsymbol{X}$ is not invertible when $\boldsymbol{X}^{\top} \boldsymbol{X}$ is singular.

Singular: ***A matrix whose determinant is 0 and thus is non-invertible is known as a singular matrix.***

The reasons of Matrix be singular:

1. Data generating distribution truly has no variance in some direction.
2. No variance is ovserved in some diretion because there are fewer examples than input features.

Regularization correspond to inverting $\boldsymbol{X}^{\top} \boldsymbol{X}+\alpha \boldsymbol{I}$, which is invertible whatsoever.

Regularization are able to gurantee the convergence of iterative methods aplied to underdetermined problems.

In section 2.9, we solved underdetermined linear equations using the Moore-Penrose pseudo inverse?

### 7.4 Dataset Augmentation

Create new fake data.

Easy for classification, but not as easy for densty estimation task.

Dataset augmentation is effective to object recognition.

"Translating the training images a few pixels in each direction?"

rotating or scaling images are proved to be effective for image augmentation tasks.

"Injecting noise in the input to a neural network is also a form of data augmentation"

"Neural networks prove not to be very robust to noise, however (Tang and Eliasmith, 2010). One way to improve the robustness of neural networks is simply to train them with random noise applied to their inputs."

### 7.5 Noise Robustness

