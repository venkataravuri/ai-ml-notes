# Statistics & Proabability

**Table of Contents**
- [Statistics](#statistics)
- [Descriptive Statistics](#descriptive-statistics)
  - [Probability](#probability)
    - [Overview]()
    - [Probability Distribution]()
    - [Probability Questions](https://mecha-mind.medium.com/probability-questions-for-ml-interviews-692fadf0ac12)
 - [Inferential Statistics](#inferential-statistics)
   - [Central limit Theorem]() 
   - [Hypothesis Testing]() 
 
## Statistics

Statistics is a set of mathematical methods which enable us to answer important questions about data.

Statistics is an important prerequisite for machine learning, it helps to select, evaluate and interpret predictive models. Statistical methods form the foundation for regression algorithms and classification algorithms.

Statistics helps answer questions like:
- Which features are the most important?
- How should we design the experiment to develop our product strategy?
- What performance metrics should we measure?
- What is the most common and expected outcome?
- How do we differentiate between noise and valid data?

### Definitions

**Random sample** - A random sample is a collection of nn random variables X1,...,Xn​ that are independent and identically distributed with X.

**Sample mean** - The sample mean of a random sample is used to estimate the true mean μμ of a distribution, is often noted X‾X and is defined as follows:

Statistics is divided into two categories:
- **Descriptive Statistics** - Offers methods to summarise data by transforming raw observations into meaningful information that is easy to interpret and share. Raw observations are just data, descriptive statistics transform these observations into insights that make sense. 
- **Inferential Statistics** - Offers methods to study experiments done on small samples of data and chalk out the inferences to the entire population (entire domain).

### Descriptive Statistics

There are 3 main types of descriptive statistics:

- **Frequency distribution** summarize the frequency of every possible value of a variable in numbers or percentages. Frequency distribution table & Grouped frequency distribution table.
- **Measures of central tendency** estimate the center, or average, of a data set. The mean, median and mode are 3 ways of finding the average.
- **Measures of variability** give you a sense of how spread out the response values are. The range, standard deviation and variance each reflect different aspects of spread.

## Probability

## Overview

Probability is a measure that quantifies the likelihood that an event will occur.

e.g., we can quantify the probability of a customer purchasing a product, chances of click on an advertisment and customer chrun.

### Probability Vs. Odds Ratio Vs. Log Odds

Probability, odds ratios and log odds are all the same thing, just expressed in different ways. It’s similar to the idea of scientific notation: the number 1,000 can be written as 1.0*103 or even 1*10*10*10.

- **Probability** is the probability an event happens. For example, there might be an 80% chance of rain today.
- **Odds** (more technically the odds of success) is defined as probability of success/probability of failure. So the odds of a success (80% chance of rain) has an accompanying odds of failure (20% chance it doesn’t rain); as an equation (the “odds ratio“), that’s .8/.2 = 4.
- **Log odds** is the logarithm of the odds. Ln(4) = 1.38629436 ≅ 1.386.

### Likelihood Vs. Probability

The terms "likelihood" and "probability" refer to the likelihood of events occurring.

The term "probability" refers to the possibility of something happening.
The term "Likelihood" refers to the process of determining the best data distribution given a specific situation in the data.

Now suppose we flip the coin 100 times and it only lands on heads 17 times. We would say that the likelihood that the coin is fair is quite low. If the coin was actually fair, we would expect it to land on heads much more often as P(Heads) = 0.5 usually.

## Probability Distribution

Probability distributions describe all of the possible values that a random variable can take.

Probability distribution depends on a number of factors such as distribution's mean (average), standard deviation, skewness, and kurtosis.

A probability distribution is an idealized frequency distribution.

A frequency distribution describes a specific sample or dataset. It’s the number of times each possible value of a variable occurs in the dataset.

Using response variable’s probability distribution we can answer a lot of analytical questions. We tend to collect a huge amount of data, fit the complex models to find interesting insights. In some cases, collecting data itself is a costly process. At times we have data for only the response variable. Instead of spending a lot of time and effort on collecting information in such a situation, a simple approach like distribution analysis can provide us more insights into the problem.

Probability distributions are divided into two types: **discrete** and **continuous**.

#### Discrete Distributions

Discrete distributions have finitely many outcomes, equal outcomes are called Equiprobability. Events with only two possible outcomes [True/False], so any event with two outcomes can be transformed into a Bernoulli Distribution.

**probability mass function** is a function that gives the probability that a **discrete** random variable is exactly equal to some value. 

|Discrete Distributions|Description|Examples|Probability Mass Function|Graph|
|---|---|---|---|---|
|Bernoulli distribution|Bernoulli distribution is a discrete distribution having two possible outcomes labeled as n.|e.g., Exactly two mutually exclusive outcomes of a trial, like flipping a coin (heads/tails) and outcome of a match (win/loss). <br/><br/>In flipping a coin, there are two possibilities — Head or Tail. Head occurs with the probability p and tail occurs with probability 1-p.<br/><br/>Bernoulli distribution can be used to model single events like whether I get a job or not, will it rain today or not.|$$P(n) = p^n (1-p)^{(1-n)}$$|<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*u689-6PVdf_Sd_9u26Y6Iw.png" />|
|Binomial distribution|Binomial distribution gives the discrete probability distribution of obtaining exactly x successes out of n Bernoulli trials.||$$P(X=x) = (\frac{n!}{(n-x)!x!}) p^n (1-p)^{(1-n)}$$|<img src="" />|
|Geometric distribution||How many job interviews are needed before getting a first job offer <br/><br/>How many hits a cricket bat takes before it breaks?<br/><br/>In a manufacturing process how many good units are produced before the faulty unit.|$$P(X=x) = (1-p)^xp$$|<img src="" />|
|Poission distribution|Poisson distribution is used for modeling the random arrival — like the arrival of people at the queues or calls at the support desks.|<br />• The number of users who visited a website in an interval can be thought of as a Poisson process.|$$f(x) = \frac{\lambda^x e^{-y}}{x!}$$|<img src="" />|
|Exponential distribution|||$$f(x) = \lambda{e^{-{\lambda}x}}$$|<img src="" />|

Continuous probability distribution describes the probability of a continuous random variable taking on a particular value. These types of distributions are used to model things like the height of a person, the time it takes to complete a task, or the distance a car travels.

<img src="https://i.stack.imgur.com/2bRvg.png" width="70%" height="70%" />

Real-life uses of continuous probability distribution:
- **Internet download speed** can be modeled through a continuous probability distribution. The internet speed can also accelerate or slow down due to different factors. As the speed can assume different values in Mbps, the internet download speed in a city can be modeled by a normal distribution with a specific mean and standard deviation.
- **Lifetime of electronic devices** may vary from months to years, hence, making it a continuous variable. Their life is influenced by factors such as manufacturing quality and usage conditions. To present their lifetime and predict the ideal range within which they can last can be modeled as continuous probability distribution.
- **Daily returns on a stock** are influenced by factors such as market conditions and company performance. Often the return rate which assumes a quantitative value is unpredictable, and random and can take infinite numbers amongst the real values, hence, it is a continuous variable. Due to the fluctuations in stock prices, the continuous probability distribution is the best to map and predict any changes in stock returns. 
- **Wind speeds at a geographical location** are influenced by factors such as season and local weather patterns. As wind speed is a continuous variable, a continuous probability distribution can better help present and predict the range of wind speed in a location. 

There are two common ways to represent a probability distribution, the probability density function (PDF) and cumulative distribution function (CDF). I suspect you're wondering most about the former. For the latter, the distribution is plotted as cumulative from zero to one, so the y-axis is the sum of the distribution up to a given value of x.

**probability density function** represents the density of a **continuous** random variable lying between a specific range of values.

PDF is the derivative of CDF, i.e., the rate of CDF's change, just like speed is the derivative of moving distance

|Continuous Distributions|Description|Examples|Probability Density Function|Graph|
|---|---|---|---|---|
|Uniform Distribution|Uniform distribution specifies an equal probability across a given range of continuous values. In other words, it is a probability distribution with a constant probability.|e.g., Exactly two mutually exclusive outcomes of a trial, like flipping a coin (heads/tails) and outcome of a match (win/loss). <br/><br/>In flipping a coin, there are two possibilities — Head or Tail. Head occurs with the probability p and tail occurs with probability 1-p.<br/><br/>Bernoulli distribution can be used to model single events like whether I get a job or not, will it rain today or not.|$$P(n) = p^n (1-p)^{(1-n)}$$|<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*u689-6PVdf_Sd_9u26Y6Iw.png" />|
|Normal distribution (Or) Gaussian distribution (Or) Bell Curve|||$$$$|<img src="" />|

### Inferential Statistics



**Central Limit Theorem**

The central limit theorem states that if you take sufficiently large samples from a population, the samples’ means will be normally distributed, even if the population isn’t normally distributed.

The central limit theorem in statistics states that, given a sufficiently large sample size, the sampling distribution of the mean for a variable will approximate a normal distribution regardless of that variable’s distribution in the population. Central Limit Theorem helps you balance the time and cost of collecting all the data you need to draw conclusions about the population.

Studying the population is hard, it will be extremely hard to gather data for the entire population.

When we collect a sufficiently large sample of n independent observations from a population with mean μ and standard deviation σ, the sampling distribution the sample means will be nearly normal with mean = μ and standard error = σ/ √n

If these samples meet Central Limit Theorem’s criteria, you can assume the distribution of the sample means can be approximated to the Normal distribution. So now you can use all the statistical tools the Normal distribution provides.

From this point on, since you know the distribution at hand, you can **calculate probabilities and confidence intervals, and perform statistical tests**.

- [Central Limit Theorem: a real-life application](https://towardsdatascience.com/central-limit-theorem-a-real-life-application-f638657686e1)


### Hypothesis Testing

- __Hypothesis in Statistics__: Probabilistic explanation about the presence of a relationship between observations.
- __Hypothesis in Machine Learning__: Candidate model that approximates a target function for mapping examples of inputs to outputs.

A hypothesis in machine learning:

- __Covers the available evidence__: the training dataset.
- __Is falsifiable (kind-of)__: a test harness is devised beforehand and used to estimate performance and compare it to a baseline model to see if is skillful or not.
- __Can be used in new situations__: make predictions on new data.

**Hypothesis testing** is used to confirm your conclusion or hypothesis about the population parameter (which you know from EDA or your intuition).

Through hypothesis testing, you can determine whether there is enough evidence to conclude if the hypothesis about the population parameter is true or not.

Hypothesis Testing starts with the formulation of these two hypotheses:
- **Null hypothesis (H₀)**: The status quo
- **Alternate hypothesis (H₁)**: The challenge to the status quo

_Either reject or fail to reject the null hypothesis_

**Hypothesis testing example**
> You want to test whether there is a relationship between gender and height. Based on your knowledge of human physiology, you formulate a hypothesis that men are, on average, taller than women. To test this hypothesis, you restate it as:

> H0: Men are, on average, not taller than women.

> Ha: Men are, on average, taller than women.

Steps in Hypothesis testing:

1. Developing your initial _research hypothesis_ (the prediction that you want to investigate), it is important to restate it as a null (Ho) and alternate (Ha) hypothesis so that you can test it mathematically.
2. Collect data in a way designed to test the hypothesis.
3. Perform an appropriate statistical test such as t-Test
4. Decide whether to reject or fail to reject your null hypothesis.
5. Present the findings in your results and discussion section.

Statistical tests come in three forms:
- **Comparison tests** assess whether there are differences in means, medians or rankings of scores of two or more groups.
- **Correlation tests** determine the extent to which two variables are associated.
- **Regression tests** demonstrate whether changes in predictor variables cause changes in an outcome variable

## References
- [Conditional Probability Explained (with Formulas and Real-life Examples)](https://365datascience.com/tutorials/statistics-tutorials/conditional-probability/)

This is a univariate probability distribution, which is the probability distribution of a single random variable. This is in contrast to a bivariate or multivariate probability distribution, which defines the probability distribution of two or more random variables.

Check out this site to learn about 76 types of univariate distribution

- [Statistics cheatsheet](https://stanford.edu/~shervine/teaching/cme-106/cheatsheet-statistics#hypothesis-testing)
