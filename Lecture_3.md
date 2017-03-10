# Lecture 3
## Lecture 3a
### Learning algorithm for a linear neuron
This learning algorithm is similar to the one for a perceptron but instead of the weights always getting closer to a good set of weights, the outputs are always getting closer to the target outputs.

The perceptron learning algorithm works by always converging towards our generously feasible set of weights. This cannot be extend to more complex networks as the solution may not be convex and hence averaging two good solutions might give a bad solution.

So for multi-layer networks, we don't use the perceptron learning procedure. The multi-layer perceptron name is a bit of a misnomer.

Our proof of convergence will therefore need to be different, instead of getting closer to a good set of weights, we will show our output values get closer to the target values as this can be true even for non-convex problems.

Note that this is not true for perceptron learning as the outputs can get further from the target outputs even though we are getting closer to a good set of weights.

We will look at the simplest example, a linear neuron with a squared error measure.

A linear neuron (covered previously) is given by
```latex
y = \sum_i w_i x_i
```
The aim of our learning will be to minimise the error summed over all our training cases. We will use the squared difference.

We can solve analytically for this simple example. However, we will use an iterative method so that we can generalise it to multi-layer, non-linear networks.

Toy example
* We have fish, chips and ketchup and will buy a combination of each every day.
* We only get told the total price.
* Figure out the price of each type of food.

In the iterative approach, we will start with random guesses for the prices and then adjust them to fit the observed total price.

So we have `$\vec{w} = (w_\text{fish}, w_\text{chips}, w_\text{ketchup})$` and `$\vec{x} = (x_\text{fish}, x_\text{chips}, x_\text{ketchup})$`. So `$\text{price} = \vec{w}^T \vec{x}$`.

Our true answer is `$\vec{w} = (150, 50, 100)$` and for our first day we buy `$\vec{x} = (2, 5, 3)$` so the total is 850.

We will guess that all the prices are 50, giving a total of 500. 

We use the delta-rule for learning which is
```latex
\delta w_i = \epsilon x_i (t - y)
```
where `$\epsilon$` is the learning rate.

If we use `$\epsilon = \frac{1}{35}$` (for easy math), our weight changes are +20, +50 and +30. Giving us new weights `$\vec{w} = (70, 100, 80)$`. Our weights for chips got worse as there is no guarantee our individual weights will improve but the total improved.

To derive the delta rule. Our error measure is
```latex
E = \frac{1}{2} \sum_{n\in\text{training}} (t^n + y^n)^2
```
where `$t^n$` is the total price for training case `$n$`.

Differentiating with respect to the weights
```latex
\frac{\partial E}{\partial w_i} = \frac{1}{2} \sum_n \frac{\partial y^n}{\partial w_i} \frac{\mathrm{d} E^n}{\mathrm{d} y^n}\\
\frac{\partial E}{\partial w_i} = - \sum_n x^n_i (t^n - y^n)
```

The delta rule just changes the weights in proportion to this derivative
```latex
\delta w_i = -\epsilon \frac{\partial E}{\partial w_i} = \epsilon\sum_n x^n_i (t^n - y^n)
```

This iterative procedure will get as close to the best answer (smallest possible error) as the learning rate will allow us.

The learning can also be quite slow if two of the input dimensions are quite correlated. E.g. if in our earlier example, all our training cases had the same number of portions of ketchup and chips we wouldn't be able to work out their individual weights. And if they are almost always the same, it can be slow to learn their weights.

Choosing the learning rate can be annoying as too big and the system will be unstable and if its too small, it will take a long time to converge.




## Lecture 3b
### Error surface for a linear neuron
To get a geometric understanding of the learning process, we look at the extended weight space. This is a space where we have the weights on a 'horizontal axis' and have one 'vertical axis' that measures the error.

For a linear neuron with squared error, our surface in this space will be a quadratic bowl. That is vertical cross-sections will be parabolas and horizontal cross-sections will be ellipses.

In this space, we can visualise what the delta rule does. For batch learning, it does steepest descent on the error surface.

For the simplest online learning, where we change the weights after each training case rather than for all the training cases at once. 

Geometrically we will have constraint hyper-planes which are where we will get the correct answer for that particular training case. Each step will move us perpendicularly towards the relevant hyper-plane. So our learning algorithm will zig-zag around the direction of steepest descent (assuming we just step through the training cases one by one in order).

Using this geometric image, we can also see when the learning will be slow. If our lines that correspond to the training cases are almost parallel, we will get an ellipse that is very elongated. 

In this case, our steepest descent direction can be pointing very far in non-elongated direction (where we don't want to move far) and very little in the elongated direction (where we have to move far to get to the bottom of the bowl).





## Lecture 3c
### Extending learning rule to a single logistic neuron
Logistic neurons are given by
```latex
z = b + \sum_i x_i w_i \\
y = \frac{1}{1+\exp^{-z}} 
```
where `$z$` is called the logit. It is smooth and so has continous derivatives which makes learning easier.

We need to get the derivatives of this which are given by
```latex
\frac{\partial z}{\partial w_i} = x_i \\
\frac{\partial z}{\partial x_i} = w_i \\
\frac{\partial y}{\partial z} = y(1-y)
```
hence
```latex
\frac{\partial y}{\partial w_i} = x_i y(1-y) \\
\frac{\partial E}{\partial w_i} = - \sum_n x^n_i (t^n - y^n)y^n(1-y^n)
```

This is very similar to the delta-rule, the only difference is the extra `$y^n(1-y^n)$` term.




## Lecture 3c
### Backpropagation algorithm
We use this algorithm to learn multiple layers of features (i.e. learning with hidden layers). 

As a reminder, we want hidden layers because networks without them are generally quite limited. We could hand-coded features (like in a perceptron) but then our results would mostly depend on how well this hand-coded bit does. So essentially, by adding hidden layers, we are automating the process of designing these features.

#### Alternative less efficient methods
One obvious way (that doesn't work as well as backpropagation) is to learn by perturbing the weights. You randomly perturb a weight and see if this improves performance. If it does, you keep this change. 

However, the problem with this is that it is very inefficient as you would have to test the change on multiple training sets. Backpropagation is more efficient than this by a factor of the number of weights, which could be a very large number.

Another issue with randomly changing weights is that near the end of learning, any large change of weights is likely to make things worse and so you would have to make only very small changes.

Alternatively we could perturb all the weights in parallel and try to correlate the performance gain with the weight changes. However this doesn't work very well either as it is difficult to isolate the effect of one weight change without doing very many trials.

We could randomly perturb the activities of hidden units rather than the weights. If we find a perturbed activity helps, we could then compute how to change the weights to give this change in activity. This is more efficient as there are less activities than weights but backpropagation is still better by a factor of the number of neurons.

#### Backpropagation
We don't know what the hidden units ought to be doing (hence their name), however, we can compute the change in error as we change a hidden activity on a particular training case.

So instead of using activities of the hidden units as our desired states, we use the error derivatives with respect to our activities. As our hidden units can be connected to many different output units, we need to combine these effects. 

We can compute the error derivatives for all the hidden units efficiently at the same time. Also, once we have the error derivatives of the hidden activities, it is easy to compute the error derivatives for the weights going into those hidden units.


#### Sketch of the backpropagation algorithm
First we define the error and get its derivative with respect to the outputs. In this case we will use the squared error 
```latex
E = \sum_{j\in\text{output}} (t_j - y_j)^2 \\
\frac{\partial E}{\partial y_j} = - (t_j - y_j)
```
where the subscript index denotes the layer that the variable is in (in this case `$j$` denotes the output layer).

We then use the error derivatives that we have computed for the output layer to compute the error derivatives for the layer below the output. This is the essence of backpropagation, we are using the error derivatives for one layer to calculate the error derivatives of the layer below it.

Let the subscript `$j$` denote one layer and the subscript `$i$` denote the layer below the `$j$` neurons. The output of the `$j$`th neuron is `$y_j$` and the output of the `$i$`th neuron is `$y_i$`. The total input received for a particular neuron is `$z$`, so the total input received for the `$j$`th neuron is `$z_j$`.

First we need to convert from the error derivative wrt the output `$y_j$` to one wrt the input `$z_j$`. We can do this via
```latex
\frac{\partial E}{\partial z_j} = \frac{\partial E}{\partial y_j}\frac{\partial y_j}{\partial z_j} \\
\frac{\partial E}{\partial z_j} = \frac{\partial E}{\partial y_j}[y_j(1-y_j)]
```

Next we convert from an error derivative wrt the input `$z_j$` to one wrt the output of the layer below it `$y_i$`. Recall that
```latex
z_j = b_j + \sum_i w_{ij} y_i
```
where `$w_ij$` denotes the weight from `$i$`th neuron to the `$j$` neuron. So
```latex
\frac{\partial E}{\partial y_i} = \sum_j \frac{\partial E}{\partial z_j}\frac{\partial z_j}{\partial y_i} \\
\frac{\partial E}{\partial y_/i} = \sum_j \frac{\partial E}{\partial z_j}[w_{ij}]
```

Hence
```latex
\frac{\partial E}{\partial y_i} = \sum_j \frac{\partial E}{\partial y_j}[y_j(1-y_j)][w_{ij}]
```

We can also compute the error derivatives wrt the weights on the connections
```latex
\frac{\partial E}{\partial w_ij} = \frac{\partial E}{\partial z_j}\frac{\partial z_j}{\partial w_ij} \\
\frac{\partial E}{\partial w_ij} = \frac{\partial E}{\partial z_j} [y_i]
```

So it is quite simple to get the error derivative wrt the weights for a given neuron. We simply multiply the quantity that we have computed on that neuron `$\frac{\partial E}{\partial z_j}$` with the activity from the neurons below.

So we started with the derivative of the error wrt to the output of one layer, `$\frac{\partial E}{\partial y_j}$` and ended up with the derivative of the error wrt to the output of layer below it `$\frac{\partial E}{\partial y_i}$`. So clearly we can do this for as many layers as there are and we can also compute the error derivative wrt the weights easily once we have computed the other derivatives.

This is the backpropagation algorithm, it is an algorithm to compute, efficiently, the error derivative with respect to the weight for every single weight in the network given a particular training case.





## Lecture 3e
### How to use the derivatives from the backpropagation algorithm
We have seen how the backpropagation algorithm allows us to compute the error derivatives with respect to all the weights in the network efficiently. This is what allows us to learn efficiently for these networks.

However, we still need to decide other details in order to completely specify the learning procedure. These issues can be summarised as
* Optimisation issues
    - How do we use these error derivatives on individual cases to get a good set of weights? (Lecture 6)
* Generalisation issues
    - How do we ensure that our learned weights are not over fitted and work well for cases not in our training set? (Lecture 7)

We will look at a very brief overview of these two sets of issues.

#### Optimisation issues
These issues arise from deciding about how we will use the weight derivatives.

##### The first question is how often do we update the weights? 

###### Online
We could update the weights after each training case. So we use backpropagation to compute the derivatives for a single training case and then you make a small change to the weights. 

This will zig zag around since each training case will give different error derivatives. But if we make the weight changes small enough, it will head in the right direction on average.

###### Full batch
Alternatively we can use full batch training where you do a full sweep over all the training cases to compute the total error derivatives. This is done by adding together the error derivatives from each individual training cases. We can then take a small step in that direction.

The problem with this is that we start off with a bad set of weights and we might have a very big training set. So we don't want to go to all that work to fix up some weights that we know are very bad. 

Really we only need to look at a few training cases before we get an idea of the direction these bad weights should go in. We only need to look at a large set of training cases when we are near the end of the training. So this gives us mini batch learning.

###### Mini batch
As explained above, initially we should only need a small batch and hence in mini batch learning we just take a small random sample of the training cases and go in that direction.

So we will do some zig zagging, but not as much as if we were doing online and we don't have the computational cost of computer the total derivatives from all the training data.

##### Next, we need to decide how much to update (discussed further in Lecture 6)
We could by hand pick some fixed learning rate and then change each weight by the derivative times this learning rate. But it seems more sensible to adapt the learning rate.

We could do this by considering how the error is changing. If the error is oscillating up and down, then we will reduce the learning rate. But if it is making steady progress, we might increase the learning weight. 

We could even have a separate learning rate for each connection in the network so that some weights learning more quickly than others.

Or going even further, we might decide we don't want to go in the direction of steepest descent at all. If you recall from the earlier example of linear neurons, when we had very correlated inputs, we had a very elongated elliptical error surface. Hence this meant that going in the direction of steepest descent was actually going at almost right angles to the direction we wanted to go in to get the minimum.

This is actually typical of most learning problems towards the end of learning. So there are actually better directions to go in than the direction of steepest descent but the problem is that these are hard to work out.


#### Generalisation issues

##### The errors
The second set of issues is to do with how well the network generalises to inputs that it didn't see in training. The problem is that the training data contains information about the mapping from input to output but it also contains two types of noise.

1. The target values may be unreliable (normally only a minor worry for neural networks).
1. There is sampling error

If we take any set of sample data (especially small ones), we will find that there will be accidental regularities in the data just because of how the particular training cases were chosen.
 
 For example, if you are showing someone some polygons, you might show them a square and a rectangle. These are polygons, but there's no way to realise that polygons can have something other than 4 sides and that the angles don't have to be right angles.

Alternatively you might show a triangle and a hexagon but you can't tell whether polygons are always convex and if their angles are always multiples of 60. 

So how ever you choose you samples, for a finite amount of samples, there are going to be accidental regularities. When we fit a model, there is no way that the model can tell the difference between these accidental regularities and the actual regularities of mapping inputs to outputs. 

So the model will fit both. And if its a very big model it will be very good at fitting this sampling error. This of course will cause it to generalise very badly.

This overfitting can be illustrated by a simple example. Say that we have 6 points with x and y values. We could fit a straight line (2 degrees of freedom) through these points or we could fit a 5th order polynomial (6 degrees of freedom) to these points and fit them exactly.

The complicated model obviously fits the data better but it is not economical. For convincing model, we want a simple model that fits the data surprisingly well. The polynomial does do this because it has 6 degrees of freedom and so wherever these data points are, it would be able to fit them exactly. It is not surprising that a complex model can fit a small number of data points well.


##### Ways to reduce overfitting
There a large number of different method to reduce overfitting for both neural networks as well as other models. They will be described in more detail in Lecture 7 but here is a brief survey.

* Weight-decay
    - This is where we try to keep many of the weights in the model small or zero and the idea of this is just to make the model simpler.
* Weight-sharing
    - Similar to above, you make the model simpler by requiring that many of the weights have the same value as the others. 
    - This value is still learnt but it has to be shared.
* Early stopping
    - You make yourself a fake test set and as you are training the net, you peak at what is happening on this fake test set.
    - Once the performance on the fake test set starts to get worse, you stop.
* Model averaging
    - You train lots of different neural nets and average together in hopes that this will reduce the errors you are making.
* Bayesian fitting of neural nets
    - Fancy form of model averaging.
* Dropout
    - You try to make your model more robust by randomly omitting hidden units when you are training it.
* Generative pre-training
    - Somewhat more complicated and will be described near the end of the course.
