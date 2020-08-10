# Logistic_regression
Stochastic Gradient Descentによるロジスティック回帰の実装

## Result
### Different batch size
![r1](./figures/epoch_error_batch.png )
![r2](./figures/iter_error_batch.png )
![r3](./figures/iter_error_batch_.png )

### Different learning rate
![r4](./figures/epoch_error_rate.png )
![r5](./figures/epoch_error_rate_lim.png )

## Note
In logistic regression, the probability that y=0 is estimated by

![proba](./figures/eq3.png ) .

Likelihood _L_ is 

![likelihood](./figures/eq4.png ) ,

and log likelihood with L2 regularization _l_ is

![logl](./figures/eq5.png ) .

We want to find _**w**_ that maximize _l_ (minimize const function). Gradient ascent (descent) can be used.

![logl](./figures/eq7.png ) 

Partial Derivative of _l_ is calculated by the following equation. Data used to estimate partial derivative of _l_ is randomly selected in SGD.

![logl](./figures/eq6.png ) 
