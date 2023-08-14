

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
