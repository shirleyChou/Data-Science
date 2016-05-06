# Lecture 2: Decision Tree learning
* The big picture
* Overfitting
* Random variables and probabilities

## Function approximation
### The Big picture 
##### Suppose we have 20 features(also boolean features) in X and each feature could only take on 0 and 1. And y is also 0 and 1.
* How many instances are there in the space?
    * 2 to the power of n (in our case: 2^20)
* How many Hs are there(how many functions are there u can define over 2 to 20)? 
    * The number of distinct trees or the number of distinct functions?
        * the number of distinct functions. **Multible trees can represent exactly the same functions.**
    * How many **teachable functions** can we label this many examples with positive and negative labels?
        * 2 to the 2 to the 20. Because we have 2 to 20 examples and each example can be label as positive or negetive.
    * How many of this functions can Decision Trees represent?
        * All of them. i.e. 2 to the 2 to the 20.

In the hypothesis set, each decision tree has it own set of examples that it label positive and negative. i.e. some label correctly while others incorrectly. 

The whole idea of the inductive inference(归纳推理) is to somehow guess the right function from the label example that we have in hand. So the issue that I want to focus on is what is the basics for genelizing from some subset of the possible examples that have been labelled for it but not all of them. That is usually the situation we are in.

Part of the picture points out that are some ambiguity here. We are uncertain which hypothesis should we choose. So. **How many examples would we have to label in order to thrink the set of hypothesis that are consistent with the label examples down to one decision tree?** We can't let the program to guess a right function until we see all of the examples.

![](https://github.com/shirleyChou/Data-Science/blob/master/MachineLearning/Courses/CS601/picts/BigPictureJPG.JPG?raw=true)

##### So what do we do in machine learning to build pratically useful algorithms?
To build an some of the additional assumptions. What is the assumptions we build for decision trees? (Shorter decision trees is preferred over long decision tree) Why that is a reasonable thing to do? 


### Occam's razor
It suggests when other thing equal we should prefer simplest hypothesis that fits the data.


## Overfitting in Decision Trees
We can sometimes get the algorithms that fit the training data very well. But doesn't do a very good job in furture training examples. It happens for several reasons. One can be just the technical coincidences in the small sample of the training examples that the algorithms happen to be given. Another it can happen becauses it has noises(wrong labelled) in the data.

### Overfitting
![](https://github.com/shirleyChou/Data-Science/blob/master/MachineLearning/Courses/CS601/picts/overfitting.JPG?raw=true)

### Overfitting in Decision Tree Learning
![](https://github.com/shirleyChou/Data-Science/blob/master/MachineLearning/Courses/CS601/picts/OverfittingInDT.JPG?raw=true)

The degree of overfitting = Accuracy on training data - Accuracy on test data (Gap between two lines). So in practice when we train the decision tree we have to aware of this problem and do something about it.

### Reduced-Error Pruning
##### How to deal with overfitting? how to you motify the algorithms?
Slightly more often way to do in decision tree learning in particular is what call **Reduced-Error Pruning**. The idea is to **split data into training and validation set**. This issue of overfitting is at the heart of both machine learning practices and theory. We would actually come up with some theoretical result that tell us how we can bound that expective degree of overfitting in terms of other propertes of learner.


## Formal Guarantees
### Two key questions in supervised learning, and machine learning generally
1. **An algorithm design question. How to optimize?** How do we automatically, in polynomial time, generate hypothesis who do well on the training set? This is a computational complexity question. i.e. automatically generate rules that do well on observed data.
2. **Confidence Bounds, Generalization**. How many training example do we need to see in order to be confident if we do well in the training set, we also do well in the future. This is a sample complexity question. i.e. confidence for rule effectiveness on future data. (Occam's bound, VC theory, etc)

The two core resources that we care about when we think about supervised learning are, number one computation, so that is a computer science question. And number two is data. In the supervised case, the label data. This are two key resources you need to keep in mind when you want to think about the guarantee of your algorithms.

**For decision trees, if we were able to find a small decision tree that explains data well, in polynomial time, then good generalization guarantees.** But the question of finding the smallest decision tree is a NP-hard problem.


### Top Down Decision Tree Algorithms
ID3 is a top down approach for decision learning. It split teh leaf that decreases the entropy the most. But why not split according to error rate -- this is what we care about after all? Because there are examples where we can get stuck in local minima!!

### Entropy as a Better Splitting Measure
If measure of progress is entropy, we can always guarantees success under some formal relationships between the class of splits and the target (the class of splits can weakly approximate the target function).

## What you should know
* Well posed function approximation problems:
    * Instance space, X
    * Sample of labeled training data {<x(i), y(i)>}
    * Hypothesis space, H = {f: X -> Y}
* Learning is a search/optimization problem over H
    * Various objective functions
        * minimize training error (0-1 loss)
        * among hypotheses that minimize training error, select smallest (?)
    * But inductive learning without some bias is futile !
* Decision tree learning
    * Greedy top-down learning of decision trees (ID3, C4.5, ...)
    * Overfitting and tree post-pruning
    * Extensions…