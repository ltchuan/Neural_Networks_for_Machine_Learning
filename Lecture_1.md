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



## Lecture 1e
### Types of learning

Three types of learning discussed in this course are

1. Supervised learning
    * Learn to predict an output based on a given input.
1. Unsupervised learning
    * Select actions that maximise the rewards and the rewards might only happen occasionally.
1. Reinforcement learning
    * Trying to discover a good internal representation of the input.

#### Supervised learning
There are two types of supervised learning. Regression where you are trying to predict a real number or a vector of real numbers and classification where you are trying to predict a class label.

We start by choosing a model-class `$y = f(\vec{x}; \vec{W})$` that is a whole set of models that we consider candidates. So we have a function f has some parameters W and maps each input vector x to an output vector y and we adjust the parameters W to make the mapping fit the supervised training data.

To measure fit we often use square difference `$\frac{1}{2}(y - t)^2$` where the half is there to cancel off the two when we differentiate.


#### Reinforcement learning
In this type of learning, the output is an action or sequence of actions and you have decide on these actions based on occasional rewards. The goal is to maximise the expected sum of the future rewards. 

A discount factor for delayed rewards is typically used so that we don't have to look too far into the future.

This type of learning is difficult as the rewards are delayed and so it is hard to know which action was right and a scalar reward that happens occasionally doesn't supply much information.

Therefore, typically you cannot learn millions of parameters using reinforcement learning. Instead you are only trying to learn dozens of parameters or maybe a thousand.


#### Unsupervised learning
It's hard to say what the aim of unsupervised learning is. One aim is to develop a representation of the input that is useful for later supervised or reinforcement learning. 

You might want to do this in two stages as we might not want to use the payouts from reinforcement learning in order to set the parameters of our visual system.

Other goals of unsupervised learning are to provide compact low-dimensional representations of the input. We can do this because even in high-dimensional inputs of say 1 million pixels, we typically don't have a million degrees of freedom. Instead we might only have a few hundred degrees of freedom.

This is equivalent to viewing out inputs as being on or close to a low dimensional manifold and so we are essentially reducing our inputs to the location that we are on the manifold (and remembering which manifold we are using).

One example of this is principal component analysis which is a linear method where our manifold is simply a plane.

Another goal is to provide an economical representation of the input in terms of learned features. E.g. we might have a binary feature or have a large number of real valued features but only allow a few of these features to be non-zero at once.

One other goal is to find clusters in the input. This can viewed as finding a very sparse feature (i.e. we insist that all features except one are zero).