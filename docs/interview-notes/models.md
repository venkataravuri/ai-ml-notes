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

### Explain the difference between Logistic Regression and Collaborative Filtering.

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
### What are the differences and similarities between gradient boosting and random forest?

### Why does XGBoost perform better than SVM?

XGBoost is an ensemble method that uses many trees. This means it improves as it repeats itself.

SVM is a linear separator. So, if our data is not linearly separable, SVM requires a Kernel to get the data to a state where it can be separated. This can limit us, as there is not a perfect Kernel for every given dataset.

###  You are told that your regression model is suffering from multicollinearity. How do verify this is true and build a better model?

You should create a correlation matrix to identify and remove variables with a correlation above 75%. Keep in mind that our threshold here is subjective.

You could also calculate VIF (variance inflation factor) to check for the presence of multicollinearity. A VIF value greater than or equal to 4 suggests that there is no multicollinearity. A value less than or equal to 10 tells us there are serious multicollinearity issues.

You can’t just remove variables, so you should use a penalized regression model or add random noise in the correlated variables, but this approach is less ideal.
