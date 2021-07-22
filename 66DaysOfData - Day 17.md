### 66DaysOfData - Day 17

# Optimization for Training Deep Models

## Deep Learning Chapter 8

### Algorithm with Adaptive Learning Rates

#### AdaGrad

The parameters with relatively large accumulate gradients will have lower learning rates (large accumulate gradients means at the final stage of optimization, or the learning rate is too large makes the optimization process fluctuating, cumulate the square of learning rate so that $r$ takes the absolute value of gradients)

#### RMSProp

RMSProp modifies AdaGrad to perform better in the non-convex setting by changing the gradient accumulation into an exponentially weighter moving average.

For non-convex function, AdaGrad may pass through many different structures and eventually arrive at a region that is a locally convex bowl. Moreover, the learning rate can be too small before arriving at a convex structure after a long journey. RMSProp can converge rapidly after finding a convex bowl.

![Screenshot 2021-07-22 at 17.12.37](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-07-22 at 17.12.37.png)

the use of the moving average introduces a new hyperparameter $\rho$, which controls the length scale of the moving average.

#### ![Screenshot 2021-07-22 at 17.17.00](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-07-22 at 17.17.00.png)

RMSProp with Nesterov momentum. 

#### Adam

Adam - Adaptive moments

It is perhaps best seen as a variant on the combination of RMSProp and momentum with a few important distinctions.

![Screenshot 2021-07-22 at 17.26.25](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-07-22 at 17.26.25.png)

In Adam, momentum is incorporated directly as an estimate of the first order moment (with exponential weighting) of the gradient. (Through applying momentum to the rescaled gradients)

Adam includes bias corrections to the estimates of both the first-order moments (the momentum term) and the (uncentered) second-order moments to account for their initialization at the origin.

Adam is generally regarded as being fairly robust to the choice of hyperparameters.

#### Choosing the Right Optimization Algorithm

No consensus on this point.

Some research results suggest the family of algorithms with adaptive learning rates (RMSProp, AdaDelta) performed faily robustly.

SGD, SGD with momentum, RMSProp, RMSProp with momentum, AdaDelta and Adam are optimization algorithms actively in use. The choice of use largely depend on user's familiarity (hyperparameter tuning) with the algorithm.