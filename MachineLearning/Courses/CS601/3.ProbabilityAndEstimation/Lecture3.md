## Probability
#### The Axioms of Probability
* P(A or B) = P(A) + P(B) - P(A and B)
* P(¬A) + P(A) = 1
* P(A) = P(A ^ B) + P(A ^ ¬B)

#### Multivated Random Variables
* Suppose A can take on more than 2 values(k values). Then A is a **random variable** with arity k if it can take on exactly one value out of {v1,v2, ..vk}
    * $ P(A = v_{i} ∧ A = v_{j}) = 0 $ if i ≠ j (相互独立)
    * $ P(A = v_{1} ∨ A = v_{2} ∨ A = v_{k}) = \sum_{j=1}^{i}P(A=v_{j}) = 1 $
    * $ P(B) = \sum_{j=1}^{k}P(B ∧ A= v_{j}) $ 


#### Definition of Conditional Probability
Fraction of worlds in which F is true that also have H true
$$ P(H|F) = \frac{P(H ∧ F)}{P(F)} $$
The Chain Rule (which implies): $ P(H ∧ F) = P(H|F) P(F) $

#### Independent Events
Two events A and B are independent if P(A ^ B) = P(A)*P(B)

## Bayes Rule
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

## Probabilistic Approaches to Function approximation
We are very interested in function approximation, that is coming up a learning algorithm  that can learn function from Y to X. Instead of learning **deterministic function**(F: X -> Y), like the decision tree which picking one value from the label(one value of Y), we can learn the probability distribution(P(Y|X)) of value of y condition on value of x. If I want a spam filter, instead of training to learn a spam filter from email to YES/NO spam, or I could say I just want to know the probability that any given email describe by any features we gonna use, have a spam equal is YES or spam equal is NO. When Y can pick on many values, we would be interested in full distribution probability of those possible value of y.


#### The Joint probability Distribution(联合概率分布)
The joint distribution is the assignment of probability to values for a collection of random variables. For Discrete Random Variables, the joint distribution of x and y is Pr(X = x ∧ Y = y).
$$ Pr(X = x ∧ Y = y) = P(Y = y|X = x)P(X = x) = P(X = x|Y = y)P(Y = y) $$

#### Limitations of Joint Distribution
##### The Main problem of Joint Dist are
* **Continuous value** variables like your salary.
* How about **too many rows** in the table(when value is continuous)
* It's not always easy to learn the probability distribution. Consider learning Joint Dist. with 100 attributes. Then with binomial distribution of each attributes, 2^(100) of rows in this table. And most of the fraction of rows with 0 training examples. It transcend to **data sparsity**. If we really want to estimate that joint Dist. accurately, we need multible people(samples) in each rows. And even with 100 boolean attributes we restrict to predicament.

So that's the **key idea** in the lecture, joint probability distribution are absolutely awesome, unfortunately it's not how we estimate them when we start scaling up to reasonable numbers of variables.

##### What to do?
1. Be smart about how we estimate probabilities from sparse data
    * maximum likelihood estimates
    * maximum a posteriori estimates

2. Be smart about how to represent joint distributions
    * Bayes networks, graphical models


## Estimating probability Distributions
### Principles for Estimating Probabilities
#### Principle1: Maximum likehood Estimation(MLE) - 最大似然估计
MLE: Choose parameters $ \theta $ that maximizes $ P(data|\theta) $.  
$ P(D|\theta) $ is called **data likelihood**. 
$$ \widehat{\theta} = arg \underset{\theta}{max}P(D | \theta) $$  
$ \ln P(D|\theta) $ is called **log likelihood**.
$$ \widehat{\theta} = arg \underset{\theta}{max}\ln P(D | \theta) $$

Since the value $ P(D|\theta) $  depends on $ \theta $, the way to maximize $ \ln P(D|\theta) $ is to set derivative to zero: $ \frac{\partial}{\partial\theta} \ln P(D|\theta) = 0 $  

Filp coins $ \alpha_{T} + \alpha_{H} $ times. Which:  
$ \alpha_{T} $: tails outcomes   
$ \alpha_{H} $: heads outcomes    
$ P(D|\theta) = \theta^{{\alpha}_{H}}(1-\theta)^{{\alpha}_{T}} $: the probability of $ \alpha_{H} $ heads and $ \alpha_{T} $ tails. 


$  \frac{\partial}{\partial\theta} \ln P(D|\theta) \\
= \frac{\partial}{\partial\theta} (\ln\theta^{{\alpha}_{H}}(1-\theta)^{{\alpha}_{T}}) \\
= \frac{\partial}{\partial\theta} (\alpha_{H}\ln(\theta) + \alpha_{T}\ln(1-\theta)) \\
=  \alpha_{H}\frac{\partial}{\partial\theta}\ln\theta + \alpha_{T}\frac{\partial}{\partial\theta}\ln(1-\theta) \\
= \frac{\alpha_{H}}{\theta} - \frac{\alpha_{T}}{1-\theta} = 0$

Which infer that:
$$ \widehat{\theta}_{MLE} = \frac{\alpha_{H}}{\alpha_{H} + \alpha_{T}}  $$

#### Principle2: Maximum Aposteriori Probability - 最大后验概率
MAP: Choose parameters $ \theta $ that maximizes $ P(\theta|data) $.  
Remember to use Bayes Rule:
$$ P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)} $$
In which $ P(D|\theta) $ can be calculated by MLE, $ P(D) $ not depend on $ \theta $. So:
$$ P(\theta|D) ∝ P(D|\theta)P(\theta) $$


