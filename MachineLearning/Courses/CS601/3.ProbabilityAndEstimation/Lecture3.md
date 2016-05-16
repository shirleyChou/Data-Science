### The Axioms of Probability
* P(A or B) = P(A) + P(B) - P(A and B)
* P(¬A) + P(A) = 1
* P(A) = P(A ^ B) + P(A ^ ¬B)

### Multivated Random Variables
* Suppose A can take on more than 2 values(k values). Then A is a **random variable** with arity k if it can take on exactly one value out of {v1,v2, ..vk}
    * $ P(A = v_{i} ∧ A = v_{j}) = 0 $ if i ≠ j (相互独立)
    * $ P(A = v_{1} ∨ A = v_{2} ∨ A = v_{k}) = \sum_{j=1}^{i}P(A=v_{j}) = 1 $
    * $ P(B) = \sum_{j=1}^{k}P(B ∧ A= v_{j}) $ 


### Definition of Conditional Probability
Fraction of worlds in which F is true that also have H true
$$ P(H|F) = \frac{P(H ∧ F)}{P(F)} $$
The Chain Rule (which implies): $ P(H ∧ F) = P(H|F) P(F) $

### Independent Events
Two events A and B are independent if P(A ^ B) = P(A)*P(B)

### Bayes Rule
#### Intuition of Bayes' Rule
![](https://github.com/shirleyChou/Data-Science/blob/master/MachineLearning/Courses/CS601/picts/intuition.JPG?raw=true)

* P(A) is the **prior** distribution (先验分布)
* P(A|B) is the **posterior** distribution (后验分布)
$$ P(B|A) = \frac{P(A ∧ B)}{P(A)} = \frac{P(A|B)P(B)}{P(A)} $$



#### More General Forms of Bayes Rule
$$ 
P(A|B) = \frac{P(B|A)P(A)}{P(B|A)P(A) + P(B|¬A)P(¬A)}  \\
P(A|B ∧ X) = \frac{P(B|A ∧ X)P(A ∧ X)}{P(B ∧ X)}  \\
P(A=v_{i}|B) = \frac{P(B|A=v_{i})P(A=v_{i})}{\sum_{k=1}^{n_{A}}P(B|A=v_{k})P(A=v_{k})}
$$

#### Bayes Rule and function approximation
We would be able to train machine learning programs that represent instead of learning functions, like the decision tree did, instead of learning **deterministic function**(function from X to Y), we can learn the probability distribution(P(Y|X)) of y given x. 

And that's what to do with function approximation. Learn P of Y given x  going to be just as important for us as learning the deterministic function. 

## The Joint Distribution(联合分布)
For Discrete Random Variables, the joint distribution of x and y is Pr(X = x ∧ Y = y).
$$ Pr(X = x ∧ Y = y) = P(Y = y|X = x)P(X = x) = P(X = x|Y = y)P(Y = y) $$

#### Maximum likehood Estimation (最大似然估计)
