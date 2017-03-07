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