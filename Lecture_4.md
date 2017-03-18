# Lecture 4
## Lecture 4a
### Learning to predict the next word

In this section, we are looking at using backpropagation to learn a feature representation of the meaning of a word. 

Simple example of a family tree where we are using backpropagation to turn relational information into feature vectors that capture the meanings of words.

The family tree we are using is
```
            Christopher=Penelope               Andrew=Christine
                       |                             |
           -------------------------      -----------------------
           |                       |      |                     |
Margaret=Arthur                Victoria=James                Jennifer=Charles
                                       |
                               -----------------
                               |               |
                             Colin         Charlotte
```
and we have another Italian family tree with a similar structure
```
                Roberto=Maria                  Pierro=Francesca
                       |                             |
           --------------------------     -----------------------
           |                        |     |                     |
    Gina=Emilio                   Lucia=Marco                 Angela=Tomaso
                                       |
                               -----------------
                               |               |
                            Alfonso          Sophia
```

We are trying to teach our network to understand the information in this family trees.

The information in the family trees can be expressed as a set of propositions using 12 relationships
* son, daughter, nephew, niece, father, mother, uncle, aunt
* brother, sister, husband, wife

Using this we can write down triples like
* (colin, has-father, james)
* (colin, has-mother, victoria)
* (james, has-wife, victoria) - follows from two above
* (charlotte, has-brother, colin)
* (victoria, has-brother, arthur)
* (charlotte, has-uncle, arthur) - follows from two above

So learning the trees can be viewed as working out the regularities in a large set of these triples. One obvious way to express these is through simple rules such as (x, has-mother, y) & (y, has-husband z) => (x, has-father, z)

We could search for these rules but this involves searching through a combinatorially large discrete space of possibilities. We could instead use neural network that uses continuous space of weights to try and capture the information.

We are going to say it captures the information if it can predict the third element of the triple from the first two.

We are using a neural network with a structure like this
```
                        local encoding of person 2          (output)
                                    ^
                                    |  
                    distributed encoding of person 2
                                    ^
                                    |
units that learn to predict the features of the output from the features of the inputs
             ^                                                ^
             |                                                |
distributed encoding of person 1          distributed encoding of relationship
             ^                                                ^
             |                                                |
local encoding of person 1     (inputs)        local encoding of relationship
```
The architecture of this was designed by hand where the number of layers and bottle necks have been put into it to force it to learn interesting features.

We encode the information in a neural way. There are 24 possible people so there will be 24 neurons for local encoding of person 1 (so only 1 will be on per training case). Same thing for the 12 relationship units.

Similarly, for a relationship with a unique answer, we will have only 1 neuron at the top on. By using representation with only 1 neuron being on, we don't accidentally give the network any similarities between people. All pairs of people are equally dissimilar and we are not giving the network information about who's like who.

We take these 24 inputs and connected to 6 neurons and because of this, we can't dedicate 1 neuron per person. It has to re-represent the people as patterns of activity over these 6 neurons and we are hoping that when it learns these propositions, the way in which it encodes a person in this distributed encoding layer will reveal structuring of the task.

We will train it on 112 of these propositions many times. After training, we will look at these 6 units in that layer to see what they are doing.

(See slides for pictures)

If we do this, we find that one neuron learns to recognise the English names (very positive) from the Italian names (very negative) (since this will tell us whether the output is an English or Italian name).

Next it recognises the generation by making the first generation very positive and the last generation very negative. A third recognises the which branch we are in on the family tree (positive on the left and negative on the right).

So the neuron in the bottleneck have learn to represent features of the people that are useful for predicting the answer without us manually inputting these features.

Obviously these features are only useful if the higher level layers use similar presentations. E.g. Input person is generation 3 and relationship requires one generation up implies output person is of generation 2.

Another way to see the network works is to train it on most of the triples and test to see if it can predict the last few correctly. There's 112 triples, training it on 108 and testing it on the remaining 4, it go either 2 or 3 of those correct (don't forget there are 24 choices at the output so this is much better than chance).

If you train it on a much bigger dataset, it can generalise from a much smaller fraction of the data. If you have thousands of relationships, you only need to show a small percentage before it can start predicting other correctly.

This was a toy example that showed backpropagation can learn interesting features from the 1980s. Now we have much bigger computers and databases of millions of relational facts. Many of them of the form (A, R, B). We can then do as we did above and use A and R to predict B.

We could use the trained net to find very unlikely triples and these are good candidates for errors in the database. For example, of the database said Bach was born in 1902, the neural net might be able to recognise this is implausible as Bach is associated with a much older generation and everything else he is related to is much older than 1902.

Instead of using the first two terms to predict the third term, we could use all three terms as inputs and predict the probability that the fact is correct. We would need examples of a whole bunch of correct facts and ask it to give a high output for these. We would also need a good source of incorrect fact and ask it to give low output.