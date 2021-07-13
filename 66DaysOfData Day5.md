### 66DaysOfData Day5

# Numerical Computation

## Deep Learning Chapter 4

### Constrained Optimization

Sometimes we wish to find the maximal or minimal value of $f(x)$ for values of x in some set. -**Constrained optimization** 

Approaches

1. Modify gradient descent taking the constraint into account. We can only search over step sizes $\epsilon$ that yield new x points that are feasible. / or project each point on the line back into the constraint region. 

2. Design a different unconstrained optimization problem, whose solution can be converted into a solution to the original constrained optimization problem.

   

   Karush-Kuhn-Tucker (KKT) provedes a very general solution to constrained optimization

   

   We define the generalized Lagrangian function 

   

   