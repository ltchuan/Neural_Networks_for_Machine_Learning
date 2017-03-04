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