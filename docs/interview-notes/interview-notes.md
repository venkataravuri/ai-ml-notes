# :robot: Machine Learning - :briefcase: Interview Notes :clipboard:

### Probability Interview Questions

- [28 Probability Questions for ML Interviews](https://mecha-mind.medium.com/probability-questions-for-ml-interviews-692fadf0ac12)
- [Probability and Statistics for Software Engineering Problems](https://mecha-mind.medium.com/probability-and-statistics-for-software-engineers-1c67c96a81e3)

### Explain the difference between supervised and unsupervised machine learning? What are the most common algorithms for supervised learning and unsupervised learning?

In **supervised learning**, the algorithm is trained on a labelled dataset, meaning the desired output is provided, while in **unsupervised learning**, the algorithm must find patterns and relationships in an unlabeled dataset.

**Supervised Learning Model Types**

- In regression models, the output is continuous
- In classification models, the output is discrete.

**Models**

- Linear Regression
- Linear Support Vector Machines (SVMs)
- Naive Bayes


**Unsupervised learning**

Principal Component Analysis (PCA)
Clustering - Clustering is an unsupervised technique that involves the grouping, or clustering, of data points.
  - K-Means Clustering

### What is the difference between overfitting and underfitting?

**Overfitting** occurs when a model learns to perform exceptionally well on the training data but fails to perform well on new, unseen data.

- Overfitting occurs when a model learns the training data too well, **capturing even the noise or random fluctuations** present in the data.
- Model is too complex and adapts to the training data’s peculiarities rather than learning the underlying pattern.

**Underfitting** occurs when a model is too simple and cannot capture the underlying pattern in the data, resulting in poor performance on both the training and test data.

- Model doesn’t have enough capacity or complexity to learn the true relationship between the input features and the target variable.

||Underfitting|Just right|Overfitting|
|---|---|---|---|
|**Symptoms**|• High training error<br/>• Training error close to test error<br/>• High bias|• Training error slightly lower than test error|• Very low training error<br/>• Training error much lower than test error<br/>• High variance|
|**Regression illustration**|<img src="https://stanford.edu/%7Eshervine/teaching/cs-229/illustrations/regression-underfit.png" width="50%" height="50%" />|<img src="https://stanford.edu/%7Eshervine/teaching/cs-229/illustrations/regression-just-right.png" width="50%" height="50%" />|<img src="https://stanford.edu/%7Eshervine/teaching/cs-229/illustrations/regression-overfit.png" width="50%" height="50%" />|
|**Classification illustration**|<img src="https://stanford.edu/%7Eshervine/teaching/cs-229/illustrations/classification-underfit.png" width="50%" height="50%" />|<img src="https://stanford.edu/%7Eshervine/teaching/cs-229/illustrations/classification-just-right.png" width="50%" height="50%" />|<img src="https://stanford.edu/%7Eshervine/teaching/cs-229/illustrations/classification-overfit.png" width="50%" height="50%" />|
|**Deep learning illustration**|<img src="https://stanford.edu/%7Eshervine/teaching/cs-229/illustrations/deep-learning-underfit-en.png" width="50%" height="50%" />|<img src="https://stanford.edu/%7Eshervine/teaching/cs-229/illustrations/deep-learning-just-right-en.png" width="50%" height="50%" />|<img src="https://stanford.edu/%7Eshervine/teaching/cs-229/illustrations/deep-learning-overfit-en.png" width="50%" height="50%" />|
|**Possible remedies**|• Complexify model<br/>• Add more features<br/>• Train longer||• Perform regularization<br/>• Get more data|

[Reference-1](https://stanford.edu/%7Eshervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks)
[Reference-2](https://medium.com/@nerdjock/lesson-15-machine-learning-overfitting-underfitting-and-model-complexity-intuition-ba6874224a2c)

### Explain regularization. When is 'Ridge regression' favorable over 'Lasso regression'?

Refer to [Regularization Notes](https://github.com/venkataravuri/ai-ml/blob/master/docs/machine-learning.md#regularization)

### What do L1 and L2 regularization mean and when would you use L1 vs. L2? Can you use both?

Lasso will be unable to make model if there is multicollinearity.

###  When there are highly correlated features in your dataset, how would the weights for L1 and L2 end up being?

L2 regularization can keep the parameter values from going too extreme. While L1 regularization can help remove unimportant features. 

|Lasso|Rasso|Elasitc Net|
|---|---|---|
|• Shrinks coefficients to 0<br/>• Good for variable selection|Makes coefficients smaller|Tradeoff between variable selection and small coefficients|

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*-aJ9MXozOKv6joiX.png" width="70%" height="70%" />

### Explain the difference between likelihood and probability.

**Probability** is used to find the chances of occurrence of a particular event whereas **likelihood** is used to maximize the chances of occurrence of a particular event.

|Probability|Likelihood|
|---|---|
|<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*s7gd7r0HIDutc_h6iE62gA.png" width="110%" height="110%" />|<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*TtpklBikcior40IGSs59ig.png" width="80%" height="80%" />|
|Probability is simply how likely something is to happen. It is attached to the possible results.|Likelihood talks about the model parameters or the evidence. In likelihood, the data or the outcome is known and the model parameters or the distribution have to be found.|
|Probabilities are the areas under a fixed distribution. Mathematically denoted by: p( data \| distribution )|Likelihoods are the y-axis values for fixed data points with different distributions. Mathematically denoted by: L( distribution \| data )|
|p(x1 ≤ x ≤ x2 \| μ, σ)|L(mean=μ, sd=σ \| X=x0) = y0|

[Reference](https://medium.com/@banerjeesoumya15/probability-vs-likelihood-d2b412b0f43a)

### What is instance normalisation?

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*N0uMSJp3_0X-De3qjvfdEg.png" width="70%" height="70%" />

Batch Normalisation normalises the feature map across the entire batch of fixed size using the mean and variance of the entire batch. **Instance normalisation** (a.k.a. Contrast Normalisation) on the other hand normalises each channel for each sample.

Instance norm normalises the contrast of the input image making the stylised image independent of the input image contrast, thus improving the output quality

### Explain the bias-variance tradeoff.

Bias is a phenomenon that skews the result of a model in favour or against an idea.

Technically, bias is error between average model prediction and the ground truth. Moreover, it describes _how well the model matches the training dataset_.

- A model with **high bias** would not match the dataset closely.
- A **low bias** model will closely matches the training dataset.

**Variance** is the variability in model prediction when using different portions of the training dataset.

A model with **high bias** tries to oversimplify the model whereas a model with **high variance** fails to generalize on unseen data. Upon reducing the bias, the model becomes susceptible to high variance and vice versa. Hence, a trade-off or balance between these two measures is what defines a good predictive model.

|Characterstics of high bias model|Characterstics of high variance model|
|---|---|
|• Failure to capture proper data trends<br/>• Potential towards underfitting<br/>• More generalized / overly simplified<br/>• High error rate|• Noice in dataset<br/>• Potential towards overfitting<br/>• Complex models<br/>• Trying to put all data points as close as possible.

Bias Variance Complexity:
- Low bias & low variance is the ideal condition in theory only, but practically it is not achievable.
- Low bias & high variance is the reason of overfitting where the line touch all points which will lead us to poor performance on test data.
- High bias & low variance is the reason of underfitting which performs poor on both train & test data.
- High bias & high variance will also lead us to high error.

> A model which has _low bias_ and _high variance_ is said to be _overfitting_ which is the case where the model performs really well on the training set but fails to do so on the unseen set of instances, resulting in high values of error. One way to tackle overfitting is _Regularization_.

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*rW2XEw2lDyjSslo9F0O-WA.png" width="50%" height="50%" />

- Models with high bias will have low variance.
- Model with high variance will have low bias.

<img src="https://miro.medium.com/v2/resize:fit:2964/1*cz9IXO7jEtzSZqYUBC4yVw.png" width="70%" height="70%"/>

### While analyzing your model’s performance, you noticed that your model has low bias and high variance. What measures will you use to prevent it (describe two of your preferred measures)?

### How do you handle data imbalance issues?

Refer to,
- [Class Imbalance data in Machine Learning Notes](https://github.com/venkataravuri/ai-ml/blob/master/docs/machine-learning.md#class-imbalance-data-in-machine-learning)
- [Techniques for handling imbalanced data Notes](https://github.com/venkataravuri/ai-ml/blob/master/docs/machine-learning.md#techniques-for-handling-imbalanced-data)

### How to deal with outliers? What are the data preprocessing techniques to handle outliers? Mention 3 ways that you prefer, with proper explanation.


#### Methods for Detection of Outliers

We can use visualization,

|||
|---|---|
|Box Plot|It captures the summary of the data effectively and also provides insight about 25h, 50th and 75th percentile, median as well as outliers|
|Scatter Plot|It is used when we want to determine the relationship between the 2 variables and can be used to detect any outlier(s)|
|Inter-Quartile Range||
|Z score method|It tells us how far away a data point is from the mean.|
|DBSCAN (Density Based Spatial Clustering of Applications with Noise)|is focused on finding neighbors by density (MinPts) on an ‘n-dimensional sphere’ with radius ɛ. A cluster can be defined as the maximal set of ‘density connected points’ in the feature space.|

#### Methods for Handling the Outliers

|||
|--|--|
|Deleting observations|Delete outlier values if it is due to data entry error, data processing error or outlier observations are very small in numbers.|
|Transforming and binning values|Transforming variables can also eliminate outliers. Natural log of a value reduces the variation caused by extreme values. Binning is also a form of variable transformation. Decision Tree algorithm allows to deal with outliers well due to binning of variable. We can also use the process of assigning weights to different observations.|
|Imputing|We can also impute outliers by using mean, median, mode imputation methods. Before imputing values, we should analyze if it is natural outlier or artificial. If it is artificial, we can go with imputing values. We can also use statistical model to predict values of outlier observation and after that we can impute it with predicted values.|

[Source](https://medium.com/analytics-vidhya/how-to-remove-outliers-for-machine-learning-24620c4657e8)

### How to deal with missing values? Mention three ways to handle missing or corrupted data in a dataset.?

Missing data can be handled by imputing values using techniques such as mean imputation or regression imputation. Outlier values can be detected and removed using methods such as Z-score or interquartile range (IQR) based outlier detection.

### Explain Gradient descent and Stochastic gradient descent. Which one would you prefer?
### Mention one disadvantage of Stochastic Gradient Descent.

Refer to,
- [Gradient Descent](https://github.com/venkataravuri/ai-ml/blob/master/docs/machine-learning.md#gradient-descent)
- [Stochastic Gradient Descent](https://github.com/venkataravuri/ai-ml/blob/master/docs/machine-learning.md#stochastic-gradient-descent)

### Can you explain logistic regression and derive gradient descent for Logistic regression

https://medium.com/intro-to-artificial-intelligence/logistic-regression-using-gradient-descent-bf8cbe749ceb
https://medium.com/intro-to-artificial-intelligence/multiple-linear-regression-with-gradient-descent-e37d94e60ec5
https://medium.com/intro-to-artificial-intelligence/linear-regression-using-gradient-descent-753c58a2b0c

### Explain different types of Optimizers? How is 'Adam' optimizer different from 'RMSprop'? Explain how Momentum differs from RMS prop optimizer?

|Stochastic gradient descent (SGD)|Adam, adaptive moment estimation|RMSprop - root mean squared propagation|Adagrad|
|---|---|---|---|
|Simple and widely used optimizer that updates the model parameters based on the gradient of the loss function with respect to the parameters.|uses moving averages of the gradients to automatically tune the learning rate, which can make it more efficient and easier to use than SGD|an optimizer that divides the learning rate by an exponentially decaying average of squared gradients.|an optimizer that adapts the learning rate for each parameter based on the past gradients for that parameter.|
|sensitive to the learning rate and may require careful tuning.| It uses moving averages of the gradients to automatically tune the learning rate, which can make it more efficient and easier to use than SGD.|This can make it more stable and efficient than SGD, but it may require careful tuning of the decay rate.| This can make it effective for training with sparse gradients, but it may require careful tuning of the initial learning rate.|

### When is One Hot encoding favored over label encoding?

Refer to [Feature Engineering Notes]()

### Define precision, recall, and F1 and discuss the trade-off between them.

Refer to [Evaluation Metrics Notes](https://github.com/venkataravuri/ai-ml/blob/master/docs/machine-learning.md#evaluation-metrics)
 
### Explain the ROC Curve and AUC. What is the purpose of a ROC curve?

Refer to [ROC Notes](https://github.com/venkataravuri/ai-ml/blob/master/docs/machine-learning.md#roc)

### What is a Gradient?

The Gradient is nothing but a _derivative of loss function_ with respect to the weights. It is used to updates the weights to minimize the loss function during the back propagation in neural networks.

### What are the different types of Activation Functions? Explain vanishing gradient problem? What is the exploding gradient problem when using the backpropagation technique?

Refer to [Activation Functions Notes](https://github.com/venkataravuri/ai-ml/edit/master/docs/machine-learning.md#activation-functions)



### What is the cross-entropy of loss? How does the loss curve for Cross entropy look? What does the “minus” in cross-entropy mean?

Refer to [Loss Functions Notes](https://github.com/venkataravuri/ai-ml/edit/master/docs/machine-learning.md#loss-functions)

### Can you use MSE for evaluating your classification problem instead of Cross entropy?

MSE doesn’t punish misclassifications enough but is the right loss for regression, where the distance between two values that can be predicted is small.

For classification, cross-entropy tends to be more suitable than MSE 

the cross-entropy arises as the natural cost function to use if you have a sigmoid or softmax nonlinearity in the output layer of your network, and you want to maximize the likelihood of classifying the input data correctly.

###  What is a hyperparameter? How to find the best hyperparameters?

Hyperparameters are variables of which values are set by the ML engineer or any other person before training the model. These values are not automatically learned from the data.

Grid Search is a powerful tool for hyperparameter tuning, In Random Search CV, the user defines a distribution of values for each hyperparameter of the model. The algorithm then randomly samples hyperparameters from these distributions to create a set of hyperparameter combinations. For example, if there are three hyperparameters with ranges of [0.1, 1.0], [10, 100], and [1, 10], the algorithm might randomly sample values of 0.4, 75, and 5, respectively, to create a hyperparameter combination.

Random search is more efficient than grid search when the number of hyperparameters is large because it does not require evaluating all possible combinations of hyperparameter values.

learning rate, momentum, dropout, etc

Learning Rate

The learning rate determines the step size at which the model adjusts its weights during each iteration of training. A high learning rate might cause the model to overshoot the optimal weights, while a low learning rate might result in slow convergence. It’s essential to find a balance. 

Batch Size

The batch size, which determines the number of training examples used in each gradient descent iteration, is a critical hyperparameter in deep learning. 

Number of Epochs

An epoch represents a full pass through the entire training dataset. Too few epochs might lead to underfitting, while too many can lead to overfitting. Finding the right number of epochs involves monitoring validation performance. 

### What is the curse of dimensionality? Why do we need to reduce it? What is PCA, why is it helpful, and how does it work? What do eigenvalues and eigenvectors mean in PCA?

PCA stands for principal component analysis

- A dimensionality reduction technique
- ?

### What is the goal of A/B testing?

### Why is a validation set necessary?

A method for assessing the performance of a model on unseen data by partitioning the dataset into training and validation sets multiple times, and averaging the evaluation metric across all partitions.

### Can K-fold cross-validation be used on Time Series data? Explain with suitable reasons in support of your answer.

Time series cross-validation

There is a task called time series forecasting, which often arises in the form of “What will happen to the indicators of our product in the nearest day/month/year?”.

Cross-validation of models for such a task is complicated by the fact that the data should not overlap in time: the training data should come before the validation data, and the validation data should come before the test data. Taking into account these features, the folds in cross-validation for time series are arranged along the time axis as shown in the following image:

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*jRZTjEK21X1iSSsisie-5Q.png" weight="50%" height="50%" />

Cross-validation is a method used for evaluating the performance of a model, It helps to compare different models and select the best one for a specific task. 

- Hold-out - The hold-out method is a simple division of data into train and test sets.
- Stratification - Simple random splitting of a dataset into training and test sets (as shown in the examples above) can lead to a situation where the distributions of the training and test sets are not the same as the original dataset. The class distribution in this dataset is uniform:
  - 33.3% Setosa
  - 33.3% Versicolor
  - 33.3% Virginica
- k-Fold - The k-Fold method is often referred to when talking about cross-validation.
- Stratified k-Fold - The stratified k-Fold method is a k-Fold method that uses stratification when dividing into folds: each fold contains approximately the same class ratio as the entire original set.

K-fold cross-validation is a technique used in machine learning to assess the performance and generalization ability of a model. It helps us understand how well our model will perform on unseen data.

how K-fold cross-validation works:
1. Splitting the dataset: First, you divide your dataset into K equal-sized subsets or folds. For example, let’s use K=5, so you’ll have 5 subsets, each containing 200 images.
2. Training and testing: Now, you iterate through each fold, treating it as a testing set, while the remaining K-1 folds serve as the training set. 

### What is Transfer Learning? Give an example.

Transfer learning is a technique that involves training a model on one task, and then transferring the learned knowledge to a different, but related task. In other words, the model uses its existing knowledge to learn new tasks more efficiently.

In computer vision, transfer learning is used to fine-tune pretrained convolutional neural network (CNN) architectures like VGGNet (Simonyan & Zisserman, 2014), ResNet (He et al., 2015), the recently launched version of YOLO(v8) or Google’s* Inception Module for image classification tasks with an improved accuracy over basic CNN architectures built from scratch.

In NLP, it’s generally used to leverage pre-trained models with word embedding (vector representations of words) such as GloVe or Word2Vec.

In the realm of natural language processing (NLP), transfer learning has become increasingly popular in recent years due to the advent of large language models (LLMs), such as GPT-3 and BERT.

One of the key steps in transfer learning is the ability to freeze the layers of the pre-trained model so that only some portions of the network are updated during training. Freezing is crucial when you want to maintain the features that the pre-trained model has already learned.

Load a Pretrained Model

We’ll use the pre-trained ResNet-18 model for this example:

#### Load the pre-trained model
```
resnet18 = models.resnet18(pretrained=True)
```

Freezing Layers
To freeze layers, we set the requires_grad attribute to False. This prevents PyTorch from calculating the gradients for these layers during backpropagation.

#### Freeze all layers
```
for param in resnet18.parameters():
    param.requires_grad = False
```

Unfreezing Some Layers

Typically, for achieving the best results, we fine-tune some fo the later layers in the network. We can do this as follows:

#### Unfreeze last layer
```
for param in resnet18.fc.parameters():
    param.requires_grad = True
```

Modifying the Network Architecture

We’ll replace the last fully-connected layer to adapt the model to a new problem with a different number of output classes (let’s say 10 classes). Also, this allows us to use this pretrained network for other applications other than classification, for example segmentation. For segmentation, we replace the final layer with a convolutional layer instead. For this example, we continue with a classification task with 10 classes.

#### Replace last layer
```
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 10)
```

### Cosine similarity Vs. Jaccard similarity methods to compute the similarity scores

### Briefly explain the K-Means clustering and how can we find the best value of K.
### For k-means or kNN, why do we use Euclidean distance over Manhattan distance?

### Explain the difference between KNN and k-means clustering.

- K-nearest neighbors represents a **unsupervised** **classification** algorithm used for classification.  It works by calculating the distance of 1 test observation from all the observation of the training dataset and then finding K nearest neighbors of it.
- k-means clustering is an **unsupervised** **clustering** algorithm that gathers and groups data into **k number of clusters**.

<img src="https://www.kdnuggets.com/wp-content/uploads/popular-knn-metrics-0.png" width="50%" height="50%" />
<img src="https://pythonprogramminglanguage.com/wp-content/uploads/2019/07/clustering.png" width="50%" height="50%" />

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

### You are given a data set with missing values that spread along 1 standard deviation from the median. What percentage of data would remain unaffected?

The data is spread across the median, so we can assume we’re working with normal distribution. This means that approximately 68% of the data lies at 1 standard deviation from the mean. So, around 32% of the data is unaffected.

The dropout will randomly mute some neurons in the neural network. At each training stage, individual nodes are either dropped out of the net 

### How would you prevent a neural network from overfitting? How does Dropout prevent overfitting? Does it differ for train and test? How will you implement dropout during forward and backward passes?

Dropout is a technique to regularize in neural networks. When we drop certain nodes out, these units are not considered during a particular forward or backward pass in a network.

Dropout forces a neural network to learn more robust features that are useful in conjunction with many different random subsets of the other neurons. It will make the weights spread over the input features instead of focusing on just some features.

Dropout is used in the training phase to reduce the chance of overfitting. As you mention this layer deactivates certain neurons. The model will become more insensitive to weights of other nodes. Basically with the dropout layer the trained model will be the average of many thinned models. 

dropout is disabled in test phase.

### Describe convolution types and the motivation behind them.

Refer to, [Convolution Operations](https://github.com/venkataravuri/ai-ml/blob/master/docs/neural-networks-deeplearning.md#convolutional-neural-networks-cnn)

### Why do we need positional encoding in transformers?
### Describe several attention mechanisms, what are advantages and disadvantages?
### What techniques for NLP data augmentation do you know?

Data augmentation techniques are used to generate additional, synthetic data using the data you have. 

NLP data augmentation methods provided in the following projects:

- Back translation. 
- EDA (Easy Data Augmentation).
- NLP Albumentation.
- NLP Aug

**Back translation** - translate the text data to some language and then translate it back to the original language
<img src="https://lh6.googleusercontent.com/x3ZAhTDLT1QVSD8gCdaBVMquM2dcYA15A-orfzXyTzhTP8m0ZKLXz_2NrJdWlTgWKRS7BimExM8RO9Ce_uVVVdRR29vGeP0VZdncDZY0GTwkctocQyYg7HK9VL5ay3QC4JhbSXBK" />

**EDA** consists of four simple operations that do a surprisingly good job of preventing overfitting and helping train more robust models.
|Synonym Replacement|Random Insertion|Random Swap|Random Deletion|
|---|---|---|---|
|Randomly choose n words from the sentence that are not stop words. Replace each of these words with one of its synonyms chosen at random.|Find a random synonym of a random word in the sentence that is not a stop word. Insert that synonym into a random position in the sentence. Do this n times.|Randomly choose two words in the sentence and swap their positions. Do this n times.|Randomly remove each word in the sentence with probability p.|
|This **article** will focus on summarizing data augmentation **techniques** in NLP.<br/>This **write-up** will focus on summarizing data augmentation **methods** in NLP.|This **article** will focus on summarizing data augmentation **techniques** in NLP.<br/>This **article** will focus on write-up summarizing data augmentation techniques in NLP **methods**.|This **article** will focus on summarizing data augmentation **techniques** in NLP.<br/>This **techniques** will focus on summarizing data augmentation **article** in NLP.|This **article** will focus on summarizing data augmentation **techniques** in NLP.<br/>This **article** focus on summarizing data augmentation in NLP.|

**NLP Albumentation**

- Shuffle Sentences Transform: In this transformation, if the given text sample contains multiple sentences these sentences are shuffled to create a new sample. 

For example:
```text = ‘<Sentence1>. <Sentence2>. <Sentence4>. <Sentence4>. <Sentence5>. <Sentence5>.’```

Is transformed to:
```text = ‘<Sentence2>. <Sentence3>. <Sentence1>. <Sentence5>. <Sentence5>. <Sentence4>.’```

**NLPAug** Python Package helps you with augmenting NLP for your machine learning projects. NLPAug provides all the methods discussed above.

In computer vision applications data augmentations are done almost everywhere to get larger training data and make the model generalize better. 

The main methods used involve:
- cropping, 
- flipping, 
- zooming, 
- rotation, 
- noise injection, 
- and many others.  

In **computer vision**, these transformations are **done on the go using data generators**.

[Source](https://neptune.ai/blog/data-augmentation-nlp)

### References 
- [Source-1](https://medium.com/bitgrit-data-science-publication/11-machine-learning-interview-questions-77650cb89918)
- [Source-2](https://medium.com/@reachpriyaa/how-to-crack-machine-learning-interviews-at-faang-78a2882a05c5)
- [Source-3](https://sukanyabag.medium.com/top-15-important-machine-learning-interview-questions-32e6093c70e2)
- [Source-4](https://medium.com/@365datascience/10-machine-learning-interview-questions-and-answers-you-need-to-know-c9c78823954a)
- [Source-5](https://medium.com/swlh/cheat-sheets-for-machine-learning-interview-topics-51c2bc2bab4f)
