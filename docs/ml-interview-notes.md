# :robot: Machine Learning - :briefcase: Interview Notes :clipboard:


#### Explain the difference between supervised and unsupervised machine learning? What are the most common algorithms for supervised learning and unsupervised learning?

#### What is the difference between overfitting and underfitting?

**Overfitting** occurs when a model learns to perform exceptionally well on the training data but fails to perform well on new, unseen data.

- Overfitting occurs when a model learns the training data too well, **capturing even the noise or random fluctuations** present in the data.
- Model is too complex and adapts to the training data’s peculiarities rather than learning the underlying pattern.

**Underfitting** occurs when a model is too simple and cannot capture the underlying pattern in the data, resulting in poor performance on both the training and test data.

- Model doesn’t have enough capacity or complexity to learn the true relationship between the input features and the target variable.
  
[Reference](https://medium.com/@nerdjock/lesson-15-machine-learning-overfitting-underfitting-and-model-complexity-intuition-ba6874224a2c)

#### Explain overfitting and regularization

**Overfitting** occurs when a model learns to perform exceptionally well on the training data but fails to perform well on new, unseen data.

**Regularization** is a set of techniques designed to prevent overfitting and enhance the generalization ability of a model.

Regularization methods introduce additional constraints or penalties to the learning process to ensure that the model does not become overly complex and is better suited for making accurate predictions on new data.

- Adding a penalty term to the loss function to prevent the model from becoming too complex


#### Explain the difference between likelihood and probability.

#### What is instance normalisation?

#### What do L1 and L2 regularization mean and when would you use L1 vs. L2? Can you use both?

Lasso will be unable to make model if there is multicollinearity.

#### When there are highly correlated features in your dataset, how would the weights for L1 and L2 end up being?

L2 regularization can keep the parameter values from going too extreme. While L1 regularization can help remove unimportant features. 

#### Explain the bias-variance tradeoff.

Bias is the difference between y prediction & y actual.

**High Bias** means the model is unable to capture important features in our dataset.
**Low Bias** always make small assumptions when the model is too simple.

Bias Variance Complexity:
- Low bias & low variance is the ideal condition in theory only, but practically it is not achievable.
- Low bias & high variance is the reason of overfitting where the line touch all points which will lead us to poor performance on test data.
- High bias & low variance is the reason of underfitting which performs poor on both train & test data.
- High bias & high variance will also lead us to high error.

> When the Bias increases automatically variance decreases and vice versa also.

Total error = error due to Bias + error due to Variance

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*rW2XEw2lDyjSslo9F0O-WA.png" width="50%" height="50%" />

#### While analyzing your model’s performance, you noticed that your model has low bias and high variance. What measures will you use to prevent it (describe two of your preferred measures)?

#### How do you handle data imbalance issues?
#### Explain Gradient descent and Stochastic gradient descent. Which one would you prefer?
#### Can you explain logistic regression and derive gradient descent for Logistic regression
#### What do eigenvalues and eigenvectors mean in PCA
#### Explain different types of Optimizers — How is Adam optimizer different from Rmsprop?
#### What are the different types of activation functions and explain about vanishing gradient problem

#### What is the exploding gradient problem when using the backpropagation technique?

What is a Box-Cox transformation?
Water Tapping problem

#### Can you use MSE for evaluating your classification problem instead of Cross entropy

#### How does the loss curve for Cross entropy look?
#### What is the cross-entropy of loss?

#### What does the “minus” in cross-entropy mean?

#### Explain how Momentum differs from RMS prop optimizer?

####  What is a hyperparameter? How to find the best hyperparameters?

#### When is Ridge regression favorable over Lasso regression?


#### When is one hot encoding favored over label encoding?

    How to deal with outliers?
    How to deal with missing values?
    How to deal with an imbalanced dataset?

#### What is the curse of dimensionality? Why do we need to reduce it?

#### What is PCA, why is it helpful, and how does it work?

PCA stands for principal component analysis

- A dimensionality reduction technique
- ?

#### What is the goal of A/B testing?

#### Why is a validation set necessary?

A method for assessing the performance of a model on unseen data by partitioning the dataset into training and validation sets multiple times, and averaging the evaluation metric across all partitions.

#### Can K-fold cross-validation be used on Time Series data? Explain with suitable reasons in support of your answer.


#### Define precision, recall, and F1 and discuss the trade-off between them.
#### What is the purpose of a ROC curve?
#### Explain the ROC Curve and AUC.

#### Mention one disadvantage of Stochastic Gradient Descent.

#### What is Transfer Learning? Give an example.

#### What are the data preprocessing techniques to handle outliers? Mention 3 ways that you prefer, with proper explanation.
#### Mention three ways to handle missing or corrupted data in a dataset.?

#### Cosine similarity Vs. Jaccard similarity methods to compute the similarity scores


#### How will you implement dropout during forward and backward passes?

#### Briefly explain the K-Means clustering and how can we find the best value of K.
#### For k-means or kNN, why do we use Euclidean distance over Manhattan distance?

#### Explain the difference between KNN and k-means clustering.

#### Explain the difference between the normal soft margin SVM and SVM with a linear kernel.

#### What is the Bayes’ Theorem? Why do we use it?
#### What are Naive Bayes classifiers? Why do we use them?

#### You build a random forest model with 10,000 trees. Training error as at 0.00, but the validation error is 34.23. Explain what went wrong.

Your model is likely overfitted. A training error of 0.00 means that the classifier has mimicked training data patterns. This means that they aren’t available for our unseen data, returning a higher error.

When using random forest, this will occur if we use a large number of trees.

#### Describe the motivation behind random forests.
#### What are the differences and similarities between gradient boosting and random forest?

#### Why does XGBoost perform better than SVM?

XGBoost is an ensemble method that uses many trees. This means it improves as it repeats itself.

SVM is a linear separator. So, if our data is not linearly separable, SVM requires a Kernel to get the data to a state where it can be separated. This can limit us, as there is not a perfect Kernel for every given dataset.

####  You are told that your regression model is suffering from multicollinearity. How do verify this is true and build a better model?

You should create a correlation matrix to identify and remove variables with a correlation above 75%. Keep in mind that our threshold here is subjective.

You could also calculate VIF (variance inflation factor) to check for the presence of multicollinearity. A VIF value greater than or equal to 4 suggests that there is no multicollinearity. A value less than or equal to 10 tells us there are serious multicollinearity issues.

You can’t just remove variables, so you should use a penalized regression model or add random noise in the correlated variables, but this approach is less ideal.

#### You are given a data set with missing values that spread along 1 standard deviation from the median. What percentage of data would remain unaffected?

The data is spread across the median, so we can assume we’re working with normal distribution. This means that approximately 68% of the data lies at 1 standard deviation from the mean. So, around 32% of the data is unaffected.

#### How would you prevent a neural network from overfitting?

#### Explain how to apply drop-out. Does it differ for train and test?

#### Describe convolution types and the motivation behind them.

#### Why do we need positional encoding in transformers?
#### Describe several attention mechanisms, what are advantages and disadvantages?
#### What techniques for NLP data augmentation do you know?


## Interview Case Studies

###  Design a feed recommendation system
### Design Youtube(Google)
### Design Google contact ranking(Google)
### Design an item replacement recommendation(Instacart)
### Design an ML System to optimize coupon distribution with a set budget(Netflix)


#### What are the metrics for search ranking?


####  Imagine you’re building a system to recommend users items similar to those they’ve bought. How would you go about building this?

- Item-item similarity matrix: Create an item-item similarity matrix to measure the similarity between pairs of items. You can use cosine similarity or Jaccard similarity methods to compute the similarity scores.
- Item-based recommendation: Once the item-item similarity matrix is built, recommend items to users based on the items they have bought. For each item a user has purchased, find the most similar items and recommend those to the user.
- User-item interaction matrix: Alternatively, you can build a user-item interaction matrix that reflects the relationship between users and items. The matrix can contain information such as whether a user has bought an item, viewed it, or added it to their cart.
- User-based recommendation: Based on the user-item interaction matrix, you can recommend items to users by finding similar users and recommending items that similar users have purchased.
- Hybrid recommendation: Combine item- and user-based recommendations to create a hybrid recommendation system. This can provide better recommendations by considering both the items a user has bought and the behavior of similar users.
- Model evaluation: Evaluate the performance of the recommendation system using metrics such as accuracy, precision, recall, and F1-score. Iteratively improve the system by trying different algorithms, adjusting parameters, and incorporating user feedback.

- Content-based recommendations: Using the features of the items the user has interacted with in the past, such as genre, keywords, or other metadata, to make recommendations.
- Popularity-based recommendations: Recommend the most popular items in the product catalog, regardless of the user’s preferences.
- Hybrid approaches: Combine the outputs of multiple recommendation models to provide recommendations.

For that recommender, how would you handle a new user who hasn’t made any past purchases?

For a new user who hasn’t made any past purchases, there are several ways to handle them in a recommender system:

- Cold start: One approach is to collect more information about the new user, such as demographic information, browsing history, or preferences. This information can then be used to generate recommendations.
- Popularity-based recommendations: Another approach is to make recommendations based on popular items in the system, as this information is easily accessible and doesn’t require user-specific data.
- Clustering: Another approach is to cluster similar users based on demographic information or browsing history and use the cluster to make recommendations.
- Matrix Factorization: This approach decomposes the user-item matrix into two matrices, one representing users and the other items. This can be used to make recommendations based on similar users or items.

### References 
- [Source-1](https://medium.com/bitgrit-data-science-publication/11-machine-learning-interview-questions-77650cb89918)
- [Source-2](https://medium.com/@reachpriyaa/how-to-crack-machine-learning-interviews-at-faang-78a2882a05c5)
- [Source-3](https://sukanyabag.medium.com/top-15-important-machine-learning-interview-questions-32e6093c70e2)
- [Source-4](https://medium.com/@365datascience/10-machine-learning-interview-questions-and-answers-you-need-to-know-c9c78823954a)
- [Source-5](https://medium.com/swlh/cheat-sheets-for-machine-learning-interview-topics-51c2bc2bab4f)
