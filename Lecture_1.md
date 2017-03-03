# Lecture 1
## Lecture 1c

### Idealised neurons
####Linear neurons
```latex
y = b + \sum_i x_i w_i
```
where y is the output, b is the bias, x_i are the inputs and w_i are the weights

#### Binary threshold neurons
Computed a weighted sum of inputs and then send out a spike of activity if this sum exceeds a threshold. So we get an output of 1 if above the threshold input and 0 otherwise.

We can write this as
```latex
z = \sum_i x_i w_i
y = 
\begin{cases}
1 & \text{if } z \ge \theta
0 & \text{otherwise}
\end{cases}
```
or
```latex
z = b + \sum_i x_i w_i
y = 
\begin{cases}
1 & \text{if } z\ge 0
0 & \text{otherwise}
\end{cases}
```
which are equivalent with `$\theta = -b$`

#### Binary threshold neurons
This type of neuron combines the properties of linear neurons and binary threshold neurons. This can be written as
```latex
z = b + \sum_i x_i w_i
y = 
\begin{cases}
z & \text{if } z\ge 0
0 & \text{otherwise}
\end{cases}
```

#### Sigmoid neurons
These give a real valued output that is a smooth and bounded function of the total input. This normally uses the logistic function
```latex
z = b + \sum_i x_i w_i
y = \frac{1}{1+\exp^{-z}} 
```
The advantage of this is that it gives smooth derivatives which makes learning easier.

#### Stochastic binary neurons
Use the same equations as logistic neurons but rather than computing the output with these functions, they are computing the probability that they will output a spike. So the equations are
```latex
z = b + \sum_i x_i w_i
p(s=1) = \frac{1}{1+\exp^{-z}} 
```
and they are making a probabilistic decision using this probability, outputting either a 1 or a 0.

We can do a similar thing for rectified linear neurons, where the output from the rectified linear neuron is the Poisson rate for producing spikes.