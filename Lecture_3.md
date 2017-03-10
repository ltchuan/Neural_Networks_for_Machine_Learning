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