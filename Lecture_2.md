# Lecture 2
## Lecture 2a
### Neural network architecture
The most common type of architecture is the feed forward neural network in which the inputs flow in one direction through hidden layers until they reach the outputs.

A more interesting type of neural network is a recurrent neural network in which information can flow round in cycles. These networks can remember information for a long time and can exhibit interesting oscillations. This makes them harder to train however as they are much more complicated.

Finally we have symmetrically connected neural networks where the weights are the same in both directions between two units.

#### Feed-forward neural networks
This is the commonest type of neural network in practical applications. We have the first layer as the input and the last layer as the output. We can also have one or more layers of hidden layers between these two layers. If there is more than one layer of hidden units we call them deep neural networks.

These networks can be viewed as computing a series of transformations between the input and output. So in each layer we get a new representation of the input in which some things may have become more similar and some things may have become less similar. In order to do this we need the activities of the neurons to be non-linear functions of the previous layer.


#### Recurrent networks
These are much more powerful. They can have directed cycles in their connections. The complicated dynamics this generates can make them very difficult to train. 

Recurrent neural networks with multiple hidden layers are actually just a special case of single layer ones with some of the hidden to hidden connections missing.


#### Symmetrically connected networks
Like recurrent networks but have same weight in both directions. Much easier to analyse than recurrent networks but also more restricted in what they can do. They obey an energy function and cannot model cycles.




## Lecture 2b
Standard paradigm for statistical pattern recognition

1. Convert raw input into a vector of features using handwritten programs.
1. Learn weights that use these features to get a single scalar decision quantity.
1. If this quantity is above a certain threshold, identify the input as that class.

This is the model that a standard Perceptron or alpha Perceptron uses.

The decision unit in the Perceptron is a binary threshold neuron.

For convenience, we can treat the bias as a weight with an input of 1 so that we can treat the bias just like another weight. Also remember that the threshold is just the negative of a bias.

The perceptron convergence procedure or training procedure

1. Add one to each input vector to turn the bias into a weight.
1. Pick the a training case.
    * If output is correct, don't change weights.
    * If incorrectly outputs a zero, add the input vector to the weight vector.
    * If incorrectly outputs a one, subtract the input vector from the weight vector.

This is guaranteed to get a set of weights that will get the right answer for all the training cases if it exists.
