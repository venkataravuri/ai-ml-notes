# Machine Learning Algorithms & Models

## Supervised Learning

|Regression|Classification|
|---|---|
|• [Linear Regression]()<br/>• [Decison Trees, Random Forest]()<br/>• [Gradient Boosting - XGBoost]()<br/>• [Neural Networks]()|• [Logistic Regression]()<br/>• [Bayes Theorem & Navie Bayes]()<br/>• [Support Vector Classifier]()<br/>• [K-Nearest Neighbor](#k-nearest-neighbor)<br/>• [Random Forest](#random-forest)<br/>• [Gradient Boosting - XGBoost]()<br/>• [Neural Networks]()|
  
## Unsupervised Learning
- [K-Means Clustering]()
- [Neural Networks]()

---

## Linear Regression

**Bias** - the constant b of a linear function ```y = ax + b```
- It allows you to move the line up and down to fit the prediction with the data better.
- Without b, the line always goes through the origin (0, 0) and you may get a poorer fit.
- A bias value allows you to shift the activation function to the left or right, which may be critical for successful learning.
  
## Logisitic Regression

Logistic regression, by default, is limited to two-class classification problems. It can also be applied to multiclass problems.

### Bayes Theorem & Navie Bayes

https://machinelearningmastery.com/intuition-for-bayes-theorem-with-worked-examples/

### K-Nearest Neighbor

K-nearest neighbors (KNN) is a type of supervised learning algorithm used for both regression and classification. 
KNN tries to predict the correct class for the test data by calculating the distance between the test data and all the training points.
  - Then select the K number of points which is closet to the test data.
  - The KNN algorithm calculates the probability of the test data belonging to the classes of ‘K’ training data and class holds the highest probability will be selected.
  - In the case of regression, the value is the mean of the ‘K’ selected training points.

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/0*34SajbTO2C5Lvigs.png" height="50%" width="50%" />
<img src="https://miro.medium.com/v2/resize:fit:810/format:webp/0*KxkMe86skK9QRcJu.jpg" height="75%" width="75%" />

'K' value indicates the count of the nearest neighbors. We have to compute distances between test points and trained labels points. Updating distance metrics with every iteration is computationally expensive, and that’s why KNN is a lazy learning algorithm.

There are various methods for calculating this distance, of which the most commonly known methods are,
- Euclidian, Manhattan (for continuous)
- Hamming distance (for categorical).

|K-Nearest Neighbor Video Tutorial|
|---|
|[![K-Nearest Neighbor Video Tutorial](https://img.youtube.com/vi/HVXime0nQeI/mqdefault.jpg)](https://youtu.be/HVXime0nQeI)|

## K-Means Clustering

Clustering is the task of _grouping a set of objects_, such that objects in the same group (cluster) are more similar to each other than to those in other groups (clusters).

K-means is a centroid-based clustering technique that partitions the dataset into k distinct clusters, where each data point belongs to the cluster with the nearest center.

K-means clustering is a widely-used unsupervised machine learning technique for data segmentation.

K-means can be employed for,
- Customer segmentation
- Image compression
- Document clustering
- and anomaly detection.

Centroid-Based Clustering

In centroid-based clustering, each cluster is represented by a central vector, called the cluster center or centroid, which is not necessarily a member of the dataset. The cluster centroid is typically defined as the mean of the points that belong to that cluster.

Our goal in centroid-based clustering is to divide the data points into k clusters in such a way that the points are as close as possible to the centroids of the clusters they belong to.

|K-Means Clustering Video Tutorial|Clustering with DBSCAN Video Tutorial|
|---|---|
|[![K-Means Clustering Video Tutorial](https://img.youtube.com/vi/4b5d3muPQmA/mqdefault.jpg)](https://youtu.be/4b5d3muPQmA)|[![Clustering with DBSCAN Video Tutorial](https://img.youtube.com/vi/RDZUdRSDOok/mqdefault.jpg)](https://youtu.be/RDZUdRSDOok)|


### Random Forest

Random Forest approach is an ensemble learning method based on many decision trees.

https://datagy.io/sklearn-random-forests/

- A forest is made up of trees and these trees are randomly build.
- Random Forests are often used for feature selection in a data science workflow.
- In random forest, decision trees are trained independent to each other.

- The idea behind is a random forest is the automated handling of creating more decision trees. Each tree receives a vote in terms of how to classify. Some of these votes will be wildly overfitted and inaccurate. However, by creating a hundred trees the classification returned by the most trees is very likely to be the most accurate.

Classes, functions, and methods:

```
from sklearn.ensemble import RandomForestClassifier: random forest classifier from sklearn ensemble class.
```

```
plt.plot(x, y): draw line plot for the values of y against x values.
```

### Gradient Boosting (XGBoost)

Unlike Random Forest where each decision tree trains independently, in the Gradient Boosting Trees, the models are combined sequentially where each model takes the prediction errors made my the previous model and then tries to improve the prediction. This process continues to n number of iterations and in the end all the predictions get combined to make final prediction.

XGBoost is one of the libraries which implements the gradient boosting technique. To make use of the library, we need to install with pip install xgboost. To train and evaluate the model, we need to wrap our train and validation data into a special data structure from XGBoost which is called DMatrix. This data structure is optimized to train xgboost models faster.

xgb.train(): method to train xgboost model.
xgb_params: key-value pairs of hyperparameters to train xgboost model.
watchlist: list to store training and validation accuracy to evaluate the performance of the model after each training iteration. The list takes tuple of train and validation set from DMatrix wrapper, for example, watchlist = [(dtrain, 'train'), (dval, 'val')].

## Support Vector Machines

- In 2D data, a Support Vector Classifier is a line. 
- In 3D, it is a plane.
- In 4 or more dimensions, Support Vector Classifier is a hyperplane. 

Technically all SVCs are a hyperplane, but it is easier to call them planes in the case of 2D.

|||
|---|---|
|<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*-WBufd0WALi9tbHsp_Ns4w.jpeg" width="100%" height="100%"/>|<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*ktT3-kieBj70EF4bezYPoA.png" width="80%" height="80%"/>|

Important concepts in SVM which will be used frequently are as follows.

- **Hyperplane** − It is a decision plane or space which is divided between a set of objects having different classes.
- **Support Vectors** − Datapoints that are closest to the hyperplane are called support vectors. The separating line will be defined with the help of these data points.
- **Kernel** — A kernel is a function used in SVM for helping to solve problems. They provide shortcuts to avoid complex calculations.
- **Margin** − It may be defined as the gap between two lines on the closet data points of different classes. A large margin is considered a good margin and a small margin is considered as a bad margin.

Support Vector Machines use **Kernal** Functions to find Support Vector Classifiers in higher dimensions. A **kernel function** is a function that takes two input data points in the original input space and calculates the inner product of their corresponding feature vectors in a transformed (higher-dimensional) feature space.

|Polynomial Kernel|Radial Kernel (RBF)|
|---|---|
|The polynomial kernel is used to transform the input data from a lower-dimensional space to a higher-dimensional space where it is easier to separate the classes using a linear decision boundary.|Radial Kernel finds Support Vector Classifiers in infinite dimensions. It assigns a higher weight to points closer to the test point and a lower weight to points farther away (like nearest neighbors). Observations that further away have relatively little influence on the classification of a data point.|

<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*T5KBJYoB32xkvAsrNapz6A.png" />

In general, SVMs are suitable for classification tasks where the number of features is relatively small compared to the number of samples, and where there is a clear margin of separation between the different classes. SVMs can also handle high-dimensional data and nonlinear relationships between the features and the target variable. However, SVMs may not be suitable for very large datasets, as they can be computationally intensive and require a lot of memory.
