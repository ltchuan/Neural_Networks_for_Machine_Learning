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



## Lecture 2c
### Geometric view of perceptrons
#### Weight-space
This space has one dimension for each weight so a point in this space represents a setting for all the weights. 

Training cases will be hyper-planes (through the origin if we have eliminated the threshold) in this space. Therefore the weights have to lie on one side of the hyper-plane to get the right answer for that training case.

Mathematically, the input vector (training case) will be a normal vector to the hyper-plane. Say that the correct output is 1. Any weight point that gives the correct answer will be on the same side as the input vector points. 

We can see that this is correct as if we take the dot product between the weight vector (centred at the origin) and input vector, we would get a positive value if the angle between these two vectors is less than 90 degrees. 

If we combine multiple input vectors, we will either get a hyper-cone of feasible solutions where the weights give the correct answers for all the input vectors or find that it is impossible to correctly classify all the input vectors.

We can also see that the problem is convex from this as if we have two different weight points that both lie in the hyper-cone of feasible solutions, any point between these two points will also lie in the hyper-cone. 




## Lecture 2d
### Proof that the perceptron learning procedure works
We wish to prove the the perceptron learning procedure will eventually get us into the hyper-cone of feasible solutions. 

Let us assume that there is in fact a feasible solution to the training cases. Consider a point that lies in the hyper-cone of feasible solutions and our current weight point that incorrectly classifies at least one of the training cases. 

We can work out the squared distance between these two points as `$d_a^2 + d_b^2$` where `$d_a$` is the distance along the input vector direction and `$d_b$` is the distance perpendicular to this.

Using the learning procedure, the distance `$d_a$` will get smaller and the distance `$d_b$` won't change.

Ideally, we would want the procedure to be moving the current weight point so that it is closer to all feasible weight point. However, this is not the case, as if we consider a feasible weight point that is just right near the plane for a training case, our procedure might 'overshoot' this weight point and hence actually get further away.

This is happening because our input vector is larger longer than the perpendicular distance between this feasible weight vector and the training case hyper-plane.

This suggests a way to fix this, we can define a generously feasible weight point as one that gets the solution right for all training cases by at least the length of the input vector for all cases.

Now we can see that whenever we get an incorrectly solution, the learning procedure will always reduce the squared distance to all the generously feasible weight points by at least the squared length of the input vector.

From this we can prove convergence. Every time we make a mistake, we are reducing the distance to every point the generously feasible region. Hence, as long as none of our input vectors are infinitesimally small, we will get to this feasible region after a finite number of steps as long as this region exits.




## Lecture 2e
### Limitations of perceptrons
The limitations stem from the kinds of features used. The limitations of perceptrons emphasises that the difficulty of learning is learning the right features.

If you can choose as many features are you want and were using binary input vectors, you could just link a separate feature unit to each input and make any possible discrimination between the inputs. However, this wouldn't generalise as if you add new inputs you would need new feature units and you won't know what weights to put on these.

Once you have chosen the features, perceptrons have very strong limitations on what they can learn to do.


#### Discriminating if two binary digits are the same
An example of what perceptrons can't do. Say we wish to teach a perceptron to classify whether two binary digits are the same. So we should have `$(0,0)\rightarrow 1$`, `$(0,1)\rightarrow 0$`, `$(1,0)\rightarrow 0$` and `$(1,1)\rightarrow 1$`. However, a perceptron cannot learn this.

To show this, let's right the constraints. We have `$0 \ge \theta$`, `$w_2 < \theta$`, `$w_1 < \theta$` and `$w_1 + w_2 \ge \theta$`, where `$\theta$` is the threshold. 

If we add together the first and the last we get `$w_1 + w_2 \ge 2\theta$` and if we add together the middle two, we get `$w_1 + w_2 < 2\theta$`. These two constraints are obviously contradictory.

Another way to see that this is impossible is geometrically. Consider a data-space in which points denote inputs and our weights now correspond to hyper-planes (i.e. the opposite of what we were doing with weight space). (Self note: is this the dual of weight space?).

The hyper-plane will be normal to the weight vector and would be offset from the origin by a distance equal to the threshold. Plotting this, we get
```
F----------T
|          |
|          |
|          |
T----------F
```
where T indicates the places where we should get true or 1 and F indicates the places where we should get false. Obviously, there is no way we can draw a line through this such that we separate the T's on one side and the F's on the other. 

This is called a set of training cases that is not linearly separable.


#### Discriminating simple patterns under translation with wrap-around
Another example is if we wish to recognise patterns even if they are translated. Can our binary threshold unit discriminate between two different patterns with the same number of pixels? This is impossible if our patterns wrap-around when translated.

Say we have this pattern, pattern A
```
□□■□□■■□□■□□□□□□
□□□□■□□■■□□■□□□□
□□■□□□□□□□□■□□■■
```
and pattern B
```
□□■■□□□■■□□□□□□□
□□□□■■□□□■■□□□□□
□□■■□□□□□□□□□■■□
```
Note that both patterns have 4 pixels. A binary threshold unit cannot learn to discriminate between these two patterns.

To show this, suppose we have training cases where pattern A is in all possible positions. Since pattern A has 4 pixels, for a given pixel location, there will be 4 translations of pattern A that have that pixel location active.

If we consider all these patterns together, the total input that the decision unit receives will be 4 times the sum of all the weights.

Similarly for pattern B, lets also consider all possible translations. Once again, each pixel location will be activated by 4 different translations of pattern B. So once again the total input received by the decision unit over all these patterns will be four times the sum of all the weights.

To discriminate between these two patterns, we have to have weights that every case of pattern A will provide more input to the decision unit than every case of pattern B. But this is impossible as if we sum over all the cases of pattern A, we get the same amount of input as if we sum over all the cases of pattern B.

This is a particular case of Minsky and Papert's group invariance theorem. This theorem says that a perceptron cannot learn to recognise a pattern under transformation if the transformations form a group.

If we wish to make a perceptron that can deal with these transformations, we would have to hand-code features to recognise these transformations and so we are essentially hand-coding the pattern recognition, rather than learning it.

In order to overcome this, we have to make a neural network that can learn the feature detectors rather than just learning the weights for the feature detectors. (Guess this means use hidden layers which act as feature detectors?).

If we add more layers to hidden units, this doesn't help as the result is still linear. 

We can make the perceptrons much more powerful by putting in essentially hand coded hidden units, although these aren't actually hidden units since we have hand coded them (i.e. it is not enough to just have fixed output non-linearities).

To truly overcome these limitations, we need adaptive, non-linear hidden units but if we do this, we need a way to train them. This is essentially adapting all the weights rather than just the last layer like in a perceptron. 

Learning weights going into hidden units is equivalent to learning features and this is hard to do as there is nothing directly telling us what feature the hidden unit should be recognising. So the real problem is how do we figure out how to train these hidden units to turn them into the kinds of feature detectors we need to solve a problem.