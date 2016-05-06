# Lecture 1: Intro to ML Decision Trees
* Machine learning examples
* Well defined machine learning problem
* Decision tree learning

## What is Machine Learning:
* Study of algorithms that
* improve their performance P 
* at some task T
* with experience E

## Machine Learning in Computer Science
Machine learning already the preferred approach to
* Speech recognition, Natural language processing
* Computer vision
* Medical outcomes analysis
* Robot control

## when to use machine learning
This are applications where we have a hard time writting a program by hand

## Machine learning's interesting overlap with all kind of other simiiar fields
![](https://github.com/shirleyChou/Data-Science/blob/master/MachineLearning/Courses/CS601/picts/overlap.png?raw=true)

## What You’ll Learn in This Course
* The primary Machine Learning algorithms
  * Logistic regression, Bayesian methods, HMM’s, SVM’s, reinforcement learning, decision tree learning, boosting, unsupervised clustering, ...
* How to use them on real data
  * text, image, structured data
  * your own project
* Underlying statistical and computational theory
* Enough to read and understand ML research papers

## Function Approximation and Decision tree learning
Decison Trees is the first algorithm that we gonna talk about and I like it because it works. It's commercially used and it's a real algorithm. And it illustrate the  whole bond of points that's going to revisit over and over through the semester in different ways abount how organize **the learning system**.

## Function Approximation
Much of machine learning(I would say 90% of the paper of machine learning) are really function approximation.

## Decision Tree
Decision Trees are a very expressive, hypothesis representation. That's a property in general we like to see for learning systems. We like to have learning system that at least in principle have the capability to represent any function that we may want to teach then.


### Function Approximtion of Decision Tree
![](https://github.com/shirleyChou/Data-Science/blob/master/MachineLearning/Courses/CS601/picts/functionApproximation.png?raw=true)

### How to decide the best attribute in each iteration?
Using entropy. Entropy measures the **impurity** of S(a sample of training examples).
![](https://github.com/shirleyChou/Data-Science/blob/master/MachineLearning/Courses/CS601/picts/entropy.png?raw=true)

### Entropy is a interesting quantity for several reasons.
![](https://github.com/shirleyChou/Data-Science/blob/master/MachineLearning/Courses/CS601/picts/entropyQuantity.png?raw=true)

### How entropy is used
For all features, see how much the entropy reduces(the impurity of S reduces), so it can give the most purer sub-samples.