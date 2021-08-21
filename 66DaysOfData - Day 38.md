### 66DaysOfData - Day 38

# Sequence Modeling: Recurrent and Recursive Nets

## Deep Learning Chapter 10

### Computing the Gradient in a Recurrent Neural Network

Given

![Screenshot 2021-08-21 at 15.06.36](/Users/hirahtang/Library/Application Support/typora-user-images/Screenshot 2021-08-21 at 15.06.36.png)

$\begin{aligned} \boldsymbol{a}^{(t)} &=\boldsymbol{b}+\boldsymbol{W} \boldsymbol{h}^{(t-1)}+\boldsymbol{U} \boldsymbol{x}^{(t)} \\ \boldsymbol{h}^{(t)} &=\tanh \left(\boldsymbol{a}^{(t)}\right) \\ \boldsymbol{o}^{(t)} &=\boldsymbol{c}+\boldsymbol{V} \boldsymbol{h}^{(t)} \\ \hat{\boldsymbol{y}}^{(t)} &=\operatorname{softmax}\left(\boldsymbol{o}^{(t)}\right) \end{aligned}$​

For each node $N$ we need to compute the gradient $\nabla_{\mathbf{N}} L$​ recursively, based on the gradient computed at nodes that follow in it in the graph. 

Start with the nodes immediately preceding the final loss

$\frac{\partial L}{\partial L^{(t)}}=1$

Based on the formulas above

The gradient $\nabla_{\boldsymbol{o}^{(t)}} L$ on the output at time step $t$:

$\left(\nabla_{\boldsymbol{o}^{t)}} L\right)_{i}=\frac{\partial L}{\partial o_{i}^{(t)}}=\frac{\partial L}{\partial L^{(t)}} \frac{\partial L^{(t)}}{\partial o_{i}^{(t)}}=\hat{y}_{i}^{(t)}-\mathbf{1}_{i, y^{(t)}}$

For the end of the sequence, the final time step $\tau$, 

$\nabla_{\boldsymbol{h}^{(\tau)}} L=\boldsymbol{V}^{\top} \nabla_{\boldsymbol{o}^{(\tau)}} L$

for any given time $t$​, the gradient is from the next time step and the current output node.

$\begin{aligned} \nabla_{\boldsymbol{h}^{(t)}} L &=\left(\frac{\partial \boldsymbol{h}^{(t+1)}}{\partial \boldsymbol{h}^{(t)}}\right)^{\top}\left(\nabla_{\boldsymbol{h}^{(t+1)}} L\right)+\left(\frac{\partial \boldsymbol{o}^{(t)}}{\partial \boldsymbol{h}^{(t)}}\right)^{\top}\left(\nabla_{\boldsymbol{o}^{(t)}} L\right) \\ &=\boldsymbol{W}^{\top}\left(\nabla_{\boldsymbol{h}^{(t+1)}} L\right) \operatorname{diag}\left(1-\left(\boldsymbol{h}^{(t+1)}\right)^{2}\right)+\boldsymbol{V}^{\top}\left(\nabla_{\boldsymbol{\alpha}^{(t)}} L\right) \end{aligned}$

According to the formulas ahead, we could thus give the gradients for all the remaining parameters:

$\nabla_{c} L=\sum_{t}\left(\frac{\partial \boldsymbol{o}^{(t)}}{\partial \boldsymbol{c}}\right)^{\top} \nabla_{\boldsymbol{o}^{(t)}} L=\sum_{t} \nabla_{\boldsymbol{o}^{(t)}} L$

$\nabla_{b} L=\sum_{t}\left(\frac{\partial \boldsymbol{h}^{(t)}}{\partial \boldsymbol{b}^{(t)}}\right)^{\top} \nabla_{\boldsymbol{h}^{(t)}} L=\sum_{t} \operatorname{diag}\left(1-\left(\boldsymbol{h}^{(t)}\right)^{2}\right) \nabla_{\boldsymbol{h}^{(t)}} L$

$\nabla_{\boldsymbol{V}} L=\sum_{t} \sum_{i}\left(\frac{\partial L}{\partial o_{i}^{(t)}}\right) \nabla_{\boldsymbol{V}} \boldsymbol{o}_{i}^{(t)}=\sum_{t}\left(\nabla_{\boldsymbol{o}^{(t)}} L\right) \boldsymbol{h}^{(t)^{\top}}$

$\begin{aligned} \nabla_{\boldsymbol{W}} L &=\sum_{t} \sum_{i}\left(\frac{\partial L}{\partial h_{i}^{(t)}}\right) \nabla_{\boldsymbol{W}^{(t)}} h_{i}^{(t)} \\ &=\sum_{t} \operatorname{diag}\left(1-\left(\boldsymbol{h}^{(t)}\right)^{2}\right)\left(\nabla_{\boldsymbol{h}^{(t)}} L\right) \boldsymbol{h}^{(t-1)^{\top}} \end{aligned}$

$\begin{aligned} \nabla_{\boldsymbol{U}} L &=\sum_{t} \sum_{i}\left(\frac{\partial L}{\partial h_{i}^{(t)}}\right) \nabla_{\boldsymbol{U}^{(t)}} h_{i}^{(t)} \\ &=\sum_{t} \operatorname{diag}\left(1-\left(\boldsymbol{h}^{(t)}\right)^{2}\right)\left(\nabla_{\boldsymbol{h}^{(t)}} L\right) \boldsymbol{x}^{(t)^{\top}} \end{aligned}$





