# Logistic_regression
Stochastic Gradient Descentによるロジスティック回帰の実装
## Note
In logistic regression, the probability that y=0 is estimated by

![proba](./figures/eq3.png ) .

Likelihood _L_ is 

![likelihood](./figures/eq4.png ) ,

and log likelihood with L2 regularization _l_ is

![logl](./figures/eq5.png ) .

We want to find _**w**_ that maximize _l_. Gradient descent can be used.

![logl](./figures/eq7.png ) 

Partial Derivative of _l_ is calculated by the following equation. Data used to estimate partial derivative of _l_ is randomly selected in SGD.

![logl](./figures/eq6.png ) 
