# Classic Ml Models - Interview Notes

- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()

---

### Describe Logit and Probit models and explain the difference between them.

**Logit** model is also called _logistic regression_ model. The logit model is based on the **logistic** function (also called the **sigmoid** function).

**Probit** model is similar to logit model, but it is based on the probit function instead of the logistic function. In probit model, the cumulative distribution function Φ of standard normal distribution is used to model the relationship between the predictors and the probability of the event occurring.

The main difference between Logit and Probit models lies in the choice of the link function used to model the relationship between the predictor variables and the probability of the event occurring. In the case of the logit model, we use a** logistic or sigmoid** function while in case of probit models, the probit **link function Φ** used is a cumulative distribution function of the standard normal distribution. 

Logit models assume that the error term follows a logistic distribution, while Probit models assume that the error term follows a normal distribution.

### What are different type of recommendaiton systems? How Recommenders Work? What are different Deep Neural Network Models for Recommendation?

https://www.nvidia.com/en-us/glossary/recommendation-system/

### Briefly explain the K-Means clustering and how can we find the best value of K.

Refer to [Methods to Find the Best Value of K](docs/ml/ml-algorithms.md#methods-to-find-the-best-value-of-k)

### Explain the difference between KNN and k-means clustering.

- K-nearest neighbors represents a **unsupervised** **classification** algorithm used for classification.  It works by calculating the distance of 1 test observation from all the observation of the training dataset and then finding K nearest neighbors of it.
- k-means clustering is an **unsupervised** **clustering** algorithm that gathers and groups data into **k number of clusters**.

<img src="https://www.kdnuggets.com/wp-content/uploads/popular-knn-metrics-0.png" width="50%" height="50%" />
<img src="https://pythonprogramminglanguage.com/wp-content/uploads/2019/07/clustering.png" width="50%" height="50%" />

### For k-means or kNN, why do we use Euclidean distance over Manhattan distance?

### Explain the difference between the normal soft margin SVM and SVM with a linear kernel.

### What is the Bayes’ Theorem? Why do we use it?
### What are Naive Bayes classifiers? Why do we use them?

### You build a random forest model with 10,000 trees. Training error as at 0.00, but the validation error is 34.23. Explain what went wrong.

Your model is likely overfitted. A training error of 0.00 means that the classifier has mimicked training data patterns. This means that they aren’t available for our unseen data, returning a higher error.

When using random forest, this will occur if we use a large number of trees.

### Describe the motivation behind random forests.


###  You are told that your regression model is suffering from multicollinearity. How do verify this is true and build a better model?

You should create a correlation matrix to identify and remove variables with a correlation above 75%. Keep in mind that our threshold here is subjective.

You could also calculate VIF (variance inflation factor) to check for the presence of multicollinearity. A VIF value greater than or equal to 4 suggests that there is no multicollinearity. A value less than or equal to 10 tells us there are serious multicollinearity issues.

You can’t just remove variables, so you should use a penalized regression model or add random noise in the correlated variables, but this approach is less ideal.

### Difference between bagging and boosting.

**Ensemble** combines several models to create a strong learner, thus reducing the bias and/or variance of the individual models.

**Bagging** is one such ensemble model which creates different training subsets from the training data with replacement. In this way, the same algorithm with a similar set of hyperparameters is exposed to different subsets of the training data, resulting in a slight difference between the individual models. The predictions of these individual models are combined by taking the average of all the values for regression or a majority vote for a classification problem. Random forest is an example of the bagging method.

<img src="https://images.upgrad.com/7129ac5a-b368-44bb-927a-1b23858ff03b-bagging.png" width="60%" height="60%" />

**Boosting** is another popular approach to ensembling. This technique combines individual models into a strong learner by creating sequential models such that the final model has a higher accuracy than the individual models.

Individual models are called **weak learners**. 

<img src="https://images.upgrad.com/4c620d9a-d65a-4dee-8d60-8c894aa1f7e1-Boosting.png" width="60%" height="60%" />

**AdaBoost** - **Decision stump** is one such weak learner when talking about a shallow decision tree having a depth of only 1. Weak learners are combined sequentially such that each subsequent model corrects the mistakes of the previous model, resulting in a strong overall model that gives good predictions. [Upgrad AdaBoost Notebook](https://github.com/ContentUpgrad/Boosting/blob/main/Introduction%20to%20Boosting/Adaboost-Classifier-Updated.ipynb) & [AdaBoost Regression](https://github.com/ContentUpgrad/Boosting/blob/main/Introduction%20to%20Boosting/Adaboost_Regression.ipynb)

In **AdaBoost**, more weight is given to the datapoints which are misclassified/wrongly predicted earlier. **Gradient Boosting** performs the same by using gradients in the loss function.

**XGBoost** 

- **Parallel Computing**: When you run XGBoost, by default it would use all the cores of your laptop/machine enabling its capacity to do parallel computation.
- **Tree pruning using depth first approach**: XGBoost uses ‘max_depth’ parameter as specified instead of criterion first, and starts pruning trees backward. 
- **Missing Values**: XGBoost is designed to handle missing values internally. The missing values are treated in such a manner that any trend in missing values (if it exists)  is captured by the model.
- **Regularization**: The biggest advantage of XGBoost is that it uses regularisation in its objective function which helps to controls the overfitting and simplicity of the model,  leading to better performance.

XGBoost Hyperparameters - Learning Rate, Number of Trees and **Subsampling**, γ - Gamma is a parameter used to control the pruning of the tree.

**Subsampling** is training the model in each iteration on a fraction of data 

<img src="https://images.upgrad.com/22babbc4-5403-4af6-9ee4-ef3c07e8a46f-xgboost.gif" />

Reference: [Upgrad Notes](https://learn.upgrad.com/course/4701/segment/44323/263918/806828/4055282) 

### What are the differences and similarities between gradient boosting and random forest?

### Why does XGBoost perform better than SVM?

XGBoost is an ensemble method that uses many trees. This means it improves as it repeats itself.

SVM is a linear separator. So, if our data is not linearly separable, SVM requires a Kernel to get the data to a state where it can be separated. This can limit us, as there is not a perfect Kernel for every given dataset.

