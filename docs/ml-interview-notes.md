# Machine Learning Interview Notes


#### Explain the difference between supervised and unsupervised machine learning? What are the most common algorithms for supervised learning and unsupervised learning?
#### Explain overfitting and regularization
#### Explain the difference between likelihood and probability.
#### What is the difference between overfitting and underfitting?

#### What do L1 and L2 regularization mean and when would you use L1 vs. L2? Can you use both?
#### When there are highly correlated features in your dataset, how would the weights for L1 and L2 end up being?

#### Explain the bias-variance tradeoff.
#### While analyzing your model’s performance, you noticed that your model has low bias and high variance. What measures will you use to prevent it (describe two of your preferred measures)?

#### How do you handle data imbalance issues?
#### Explain Gradient descent and Stochastic gradient descent. Which one would you prefer?
#### Can you explain logistic regression and derive gradient descent for Logistic regression
#### What do eigenvalues and eigenvectors mean in PCA
#### Explain different types of Optimizers — How is Adam optimizer different from Rmsprop?
#### What are the different types of activation functions and explain about vanishing gradient problem>

#### Can you use MSE for evaluating your classification problem instead of Cross entropy

#### How does the loss curve for Cross entropy look?
#### What is the cross-entropy of loss?

#### What does the “minus” in cross-entropy mean?

#### Explain how Momentum differs from RMS prop optimizer?

####  What is a hyperparameter? How to find the best hyperparameters?

#### When is Ridge regression favorable over Lasso regression?


#### When is one hot encoding favored over label encoding?



#### What is the curse of dimensionality? Why do we need to reduce it?

#### What is PCA, why is it helpful, and how does it work?

PCA stands for principal component analysis

- A dimensionality reduction technique
- ?

#### What is the goal of A/B testing?

#### Why is a validation set necessary?
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
#### Explain the difference between KNN and k-means clustering.
#### What is the Bayes’ Theorem? Why do we use it?
#### What are Naive Bayes classifiers? Why do we use them?



#### Describe the motivation behind random forests.
#### What are the differences and similarities between gradient boosting and random forest?

## Interview Case Studies

###  Design a feed recommendation system
### Design Youtube(Google)
### Design Google contact ranking(Google)
### Design an item replacement recommendation(Instacart)
### Design an ML System to optimize coupon distribution with a set budget(Netflix)


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
- 
