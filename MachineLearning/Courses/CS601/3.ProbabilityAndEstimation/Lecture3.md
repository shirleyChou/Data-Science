## Probability basics
#### The Axioms of Probability
* P(A or B) = P(A) + P(B) - P(A and B)
* P(¬A) + P(A) = 1
* P(A) = P(A ^ B) + P(A ^ ¬B)

#### Multivated Random Variables
* Suppose A can take on more than 2 values(k values). Then A is a **random variable** with arity k if it can take on exactly one value out of {v1,v2, ..vk}
    * $ P(A = v_{i} ∧ A = v_{j}) = 0 $ if i ≠ j (相互独立)
    * $ P(A = v_{1} ∨ A = v_{2} ∨ A = v_{k}) = \sum_{j=1}^{i}P(A=v_{j}) = 1 $
    * $ P(B) = \sum_{j=1}^{k}P(B ∧ A= v_{j}) $ 


#### Definition of Conditional Probability(条件概率)
Fraction of worlds in which F is true that also have H true
$$ P(H|F) = \frac{P(H ∧ F)}{P(F)} $$
The Chain Rule (which implies): $ P(H ∧ F) = P(H|F) P(F) $

#### Independent Events
Two events A and B are independent if P(A ^ B) = P(A)*P(B)


#### Conjugate prior(共轭先验)
In **Bayesian probability theory**, $ P(\theta) $ is the conjugate prior for likelihood function $ P(data|\theta) $ if the forms of  $ P(\theta) $ and $ P(\theta | data) $ are the same. 

#### Bernoulli distribution(伯努利分布)
伯努利试验(Bernoulli experiment)是在**同样的条件下**(每一次试验的结果不会受其它实验结果的影响)重复地、相互独立地进行的一种随机试验。其特点是该随机试验只有两种可能结果：发生或者不发生。然后我们假设该项试验独立重复地进行了n次，那么我们就称这一系列重复独立的随机试验为n重伯努利试验，或称为伯努利概型。
又名0-1分布。随机变量只有两类取值。两类取值的概率不等。    
$P(X=0) = \theta, \\ 
P(X=1) = 1 - \theta, \\
P(X=0) + P(X=1) = 1 
$


#### Binomial distribution(二项分布)
一般地，在n次独立重复的伯努利试验中，用$ \xi $表示事件A发生的次数，如果事件发生的概率是P,则不发生的概率 q=1-p，N次独立重复试验中发生K次的概率是
$P(\xi=K) = \binom{n}{k}p^{k}(1-p)^{n-k} $


#### Beta prior distribution(Beta分布)
$ P(\theta) = \frac{\theta^{\beta_{H}-1}(1-\theta)^{\beta_{T}-1}}{B(\beta_{H}, \beta_{T})} \sim Beta(\beta_{H}, \beta_{T}) $
![](https://github.com/shirleyChou/Data-Science/blob/master/MachineLearning/Courses/CS601/picts/prior.JPG?raw=true)

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
1. Be smart about how we estimate probabilities distributions from sparse data
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
$ \ln P(D|\theta) $ is called **log likelihood** (The calculus is easiler to work is we maximize the log of it. 而且加log不影响函数的单调性).
$$ \widehat{\theta} = arg \underset{\theta}{max}\ln P(D | \theta) $$

Since the value $ P(D|\theta) $  depends on $ \theta $, the way to maximize $ \ln P(D|\theta) $ is to set derivative to zero: $ \frac{\partial}{\partial\theta} \ln P(D|\theta) = 0 $  

##### Example:Flip coins with $ \alpha_{H} $ heads and $ \alpha_{T} $ tails
* flips are **independent, identically distributed** 1's and 0's(Bernoulli) 
* $ \alpha_{H} $ and $ \alpha_{T} $ are counts that sum these outcomes (Binomial). $ P(D|\theta) $ is the **likelihood function** which represent the probability of $ \alpha_{H} $ heads and $ \alpha_{T} $ tails. 
$$ P(D|\theta) = P(\alpha_{H}, \alpha_{T} | \theta) = \theta^{{\alpha}_{H}}(1-\theta)^{{\alpha}_{T}} 
$$

So what values of $ \theta $ can make the probability $ P(D|\theta) $ as biggest as possible? Set  $ \frac{\partial}{\partial\theta} \ln P(D|\theta) = 0 $  and figure out that:

$  \frac{\partial}{\partial\theta} \ln P(D|\theta) \\
= \frac{\partial}{\partial\theta} (\ln\theta^{{\alpha}_{H}}(1-\theta)^{{\alpha}_{T}}) \\
= \frac{\partial}{\partial\theta} (\alpha_{H}\ln(\theta) + \alpha_{T}\ln(1-\theta)) \\
=  \alpha_{H}\frac{\partial}{\partial\theta}\ln\theta + \alpha_{T}\frac{\partial}{\partial\theta}\ln(1-\theta) \\
= \frac{\alpha_{H}}{\theta} - \frac{\alpha_{T}}{1-\theta} = 0$

Which infer that:
$$ \widehat{\theta}_{MLE} = \frac{\alpha_{H}}{\alpha_{H} + \alpha_{T}}  $$

**Summary**
![](https://github.com/shirleyChou/Data-Science/blob/master/MachineLearning/Courses/CS601/picts/summary.JPG?raw=true)

#### Principle2: Maximum Aposteriori Probability - 最大后验概率
MAP: Choose parameters $ \theta $ that maximizes **Posterior** $ P(\theta|data) $.  
Remember to use Bayes Rule:
$$ P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)} $$
##### In which:
* Likelihood function: $ P(D|\theta) $, which can be calculated by MLE
* Prior: $ P(\theta) $, 
* $ P(D) $ not depends on $ \theta $
$$ P(\theta|D) \propto P(D|\theta)P(\theta) $$

##### When likelihood is Binomial
If likelihood is Binomial    
$$ P(D|\theta) = \theta^{{\alpha}_{H}}(1-\theta)^{{\alpha}_{T}} $$.     
And prior is **Beta prior distribution**
$$ P(\theta) = \frac{\theta^{\beta_{H}-1}(1-\theta)^{\beta_{T}-1}}{B(\beta_{H}, \beta_{T})} \sim Beta(\beta_{H}, \beta_{T}) $$
Then **posterior** is Beta distribution:
$$ P(\theta|D) = \frac{\theta^{\alpha_{H}+\beta_{H}-1}(1-\theta)^{\alpha_{T}+\beta_{T}-1}}{B(\beta_{H} + \alpha_{H}, \beta_{T} + \alpha_{T})} \sim Beta(\beta_{H} + \alpha_{H}, \beta_{T} + \alpha_{T}) $$

And **MAP estimate** is therefore
$$ \widehat{\theta}^{MAP} = \frac{(\alpha_{H}+\beta_{H}-1)}{(\alpha_{H}+\beta_{H}-1) + (\alpha_{T}+\beta_{T}-1)}  $$

##### When Likelihood is Multinomial(outcomes more than 2)
If likelihood is multinomial
$$\theta = (\theta_{1}, \theta_{2} ..., \theta_{k}), P(D|\theta) = \theta_{1}^{\alpha_{1}}\theta_{2}^{\alpha_{2}}...\theta_{k}^{\alpha_{k}} $$

And prior is **Dirichlet distribution**
$$ P(\theta) = \frac{\theta_{1}^{\beta_{1}-1}\theta_{2}^{\beta_{2}-1}...\theta_{k}^{\beta_{k}-1}}{B(\beta_{1},...,\beta_{k})} \sim Dirichlet(\beta_{1},...,\beta_{k}) $$

Then **posterior** is Dirichlet distribution
$$ P(\theta) \sim Dirichlet(\beta_{1} + \alpha_{1},...,\beta_{k} + \alpha_{k}) $$

And **MAP estimate** is therefore
$$ \widehat{\theta}_{i}^{MAP} = \frac{(\alpha_{i}+\beta_{i}-1)}{\sum_{j=1}^{k}(\alpha_{i}+\beta_{i}-1)}  $$

http://www.cs.cmu.edu/~tom/10701_sp11/slides/MLE_MAP_1-18-11-ann.pdf
http://www.52nlp.cn/lda-math-%E8%AE%A4%E8%AF%86betadirichlet%E5%88%86%E5%B8%832
http://maider.blog.sohu.com/306392863.html