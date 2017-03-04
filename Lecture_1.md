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



## Lecture 1d
### A simple example of machine learning
We are making a simple neural network to recognise handwritten shapes. We have two types of neurons

1. A top layer of output neurons that recognise the classes of shapes.
1. A bottom layer of input neurons that represent the intensity of the pixels.

When we show the neural network a particular shape, we want the output neuron for that particular shape to get active. When an input neuron active it essentially votes for particular shapes. Each input neuron can vote for multiple shapes and vote with different intensities and the shape with the most votes wins.

Learning is done in this example by incrementing the weights of the active pixels to the class for that number. In order to stop the weights from just growing without bound, we will also decrement the weights for the active pixel to whatever number the pixel guesses. 

So we are essentially training it to do the right thing rather than the thing it currently has a tendency to do. If it makes the correct guess, then the increments will cancel off the decrements and the weights will not change which is what we want.

If we do this and train our neural network we find that the weights end up becoming like little templates for the digits. They aren't exactly templates though, as for example the number 7 and 9 both don't have the lower stroke as this doesn't help with differentiating between a 7 and a 9, instead you have to look for the loop.

Since this network is so simple, it can't learn a good way of differentiating shapes. Instead it is just using a template for the numbers and choosing based on which number has the most pixels in the template.