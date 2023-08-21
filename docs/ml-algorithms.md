# ML Models

- [K-Nearest Neighbor](#k-nearest-neighbor)
- [Random Forest](#random-forest)

### Logisitic Regression

Logistic regression, by default, is limited to two-class classification problems. It can also be applied to multiclass problems.

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

### Random Forest

Random Forest approach is an ensemble learning method based on many decision trees.

To illustrate the process of building a Random Forest classifier, consider a two-dimensional dataset with N cases (rows) that has M variables (columns).
The Random Forest algorithm will build independent decision trees but only using a random subset of the data.
For each tree, a random subset of n cases is sampled from all available N cases; the cases not used in the tree construction are called the Out Of Bag (OOB) cases. 
In addition, at each node (decision point) of a tree, a random number of m variables is used from all available M variables.

- A forest is made up of trees and these trees are randomly build.
- Random Forests are often used for feature selection in a data science workflow.
- In random forest, decision trees are trained independent to each other.

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


##### References

- https://github.com/kabiromohd/machine-learning-zoomcamp/blob/master/06-trees/07-boosting.md
- https://medium.com/swlh/k-nearest-neighbor-ca2593d7a3c4

