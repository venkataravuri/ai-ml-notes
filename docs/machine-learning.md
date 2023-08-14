# Machine Learning
- [Concepts - Statistics & Linear Algebra]()
- [Activation Functions](#activation-functions)
- [Loss Functions](#loss-functions)
- [Evaulation Metrics](#evaluation-metrics)

## Feature Engineering

|Rating|Type|Topic
------------: | ------------- | -------------
|:star:|:newspaper:|[Dealing with missing values](https://www.kaggle.com/alexisbcook/missing-values)

## Concepts - Statistics & Linear Algebra

#### What are Logits in machine learning?

Logits interpreted to be the unnormalised (or not-yet normalised) predictions (or outputs) of a model. These can give results, but we don't normally stop with logits, because interpreting their raw values is not easy.

[Logits Explanation](https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean)|

In a classification problem, the model’s output is usually a vector of probability for each category. Often, this vector is usually expected to be “logits,” i.e., real numbers to be transformed to probability using the softmax function, or the output of a softmax activation function.

#### Why we want to wrap everything with a logarithm?

1. It’s related to the concept in information theory where you need to use log(x) bits to capture x amount of information.
2. Computers are capable of almost anything, except exact numeric representation.

#### What is distiction between Gradient and Derivative?

A gradient is a vector that goes in the direction of a function’s sharpest ascend whereas a derivative quantifies the rate of shift of a function at a certain location.

- The derivative of a function is the change of the function for a given input.
- The gradient is simply a derivative vector for a multivariate function.
 - Although both ideas include calculating slopes, derivatives emphasize one variable while gradients take into account a few variables at once. [Source](https://allthedifferences.com/exploring-the-distinction-gradient-vs-derivative/)

#### What is the Difference Between Gradient and Partial Derivative?

A gradient represents the vector pointing in the direction of the steepest ascent of an equation and encompasses partial derivatives about all variables, whereas a partial derivative reflects the rate of shift of a function about one particular variable while keeping other variables at a single value.

### Regularization

Regularization helps in preventing the over-fitting of the model and the learning process becomes more efficient.

Regularization techniques,

early stopping, dropout, weight initialization techniques, and batch normalization. 

#### What is Normalization? How it is done in Neural Networks?

Normalization is a data pre-processing tool used to bring the numerical data to a common scale without distorting its shape.

#### Normalization Vs. Standardization

<img height="50%" width="50%" src="assets/Normalization-Standardization.png">

[Source](https://www.youtube.com/watch?v=of4-jeKtyB4)

#### Batch Normalization

Batch Normalization is a supervised learning technique that converts interlayer outputs into of a neural network into a standard format, called normalizing. 

This approach leads to faster learning rates since normalization ensures there’s no activation value that’s too high or too low, as well as allowing each layer to learn independently of the others.

For each layer in the neural network, batch normalization normalizes the activations by adjusting them to have a standardized mean and variance.

In a deep learning network, batch normalization affects the output of the previous activation layer by subtracting the batch mean, and then dividing by the batch’s standard deviation.

#### What is difference between Cosine Similarity & Ecludian Distance?

https://medium.com/@sasi24/cosine-similarity-vs-euclidean-distance-e5d9a9375fc8
https://cmry.github.io/notes/euclidean-v-cosine


#### Gradient Descent Vs. Stochastic Gradient Descent Vs. Batch Gradient Descent Vs. Mini-batch Gradient Descent

[Source](https://datascience.stackexchange.com/questions/53870/how-do-gd-batch-gd-sgd-and-mini-batch-sgd-differ)

Gradient Descent

Gradient Descent is an optimization method used to optimize the parameters of a model using the gradient of an objective function ( loss function in NN ). It optimizes the parameters until the value of the loss function is the minimum ( of we've reached the minima of the loss function ). It is often referred to as back propagation in terms of Neural Networks.

Batch Gradient Descent:

The samples from the whole dataset are used to optimize the parameters i.e to compute the gradients for a single update. For a dataset of 100 samples, updates occur only once.

Stochastic Gradient Descent:

Stochastic GD computes the gradients for each and every sample in the dataset and hence makes an update for every sample in the dataset. For a dataset of 100 samples, updates occur 100 times.

Mini Batch Gradient Descent:

Instead of a single sample ( Stochastic GD ) or the whole dataset ( Batch GD ), we take small batches or chunks of the dataset and update the parameters accordingly. For a dataset of 100 samples, if the batch size is 5 meaning we have 20 batches. Hence, updates occur 20 times.


## Evaluation Metrics

A confusion matrix is a table used to evaluate the performance of a classification model by comparing its predictions to the actual ground truth labels. It provides a summary of the model’s true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions for each class in a multi-class classification problem or for the positive class in a binary classification problem.

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*yshTLPeqgL9Nm5vTfGpMFQ.jpeg" width="50%" height="50%" />

**Accuracy** measures the overall correctness of the model’s predictions.

It is calculated as the ratio of the correctly predicted instances (TP + TN) to the total number of instances in the dataset. High accuracy is desirable, but it can be _**misleading when dealing with imbalanced datasets**_.

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

In **imbalanced datasets**, accuracy may not be a reliable measure, as a high accuracy score could be driven by the model’s ability to predict the majority class accurately while performing poorly on the minority class.

**Precision** measures the proportion of true positive predictions among the instances predicted as positive. It is useful when the cost of false positives is high, and you want to minimize the number of false positives.

```
Precision = TP / (TP + FP)
```

- Use precision when the cost of false positives is high and you want to minimize false alarms or false positive predictions. (minimizing false positives)
- For example, in fraud detection, precision is crucial because it indicates how many flagged cases are actually true frauds, reducing the need for manual investigation of false positives.

**Recall** (Sensitivity or True Positive Rate) measures the proportion of true positive predictions among all instances that are actually positive.

```
Recall = TP / (TP + FN)
```

- Use recall when the cost of false negatives is high, and you want to ensure that you capture as many positive instances as possible, even at the expense of some false positives. (minimizing false negatives)
- For example, in medical diagnosis, a high recall rate is crucial because it means correctly identifying individuals with a disease, even if it leads to some false alarms.

**F1 score** is the harmonic mean of precision and recall, providing a balanced metric for situations where both precision and recall are important.

```
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
```

- F1 score strikes a balance between precision and recall in the presence of an imbalanced dataset.
- For example, in sentiment analysis of customer reviews, F1 score is a suitable metric when dealing with imbalanced sentiment classes. It helps strike a balance between correctly identifying positive and negative sentiment reviews, taking into account both precision and recall.


AUC-ROC (Area Under the Receiver Operating Characteristic) curve is a graphical representation that showcases the relationship between the true positive rate (TPR)(sensitivity) and the false positive rate (FPR) as the classification threshold varies.

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*9JT5mrixTcelpsk4yyTjgQ.jpeg" width="50%" height="50%" />

```
TPR = TP / (TP+FN); FPR = FP / (FP+TN)
```

The AUC ranges from 0 to 1, where:

- AUC = 0.5 implies that the model’s performance is no better than random guessing.
- AUC > 0.5 and < 1 implies better-than-random performance, where higher values indicate better discrimination between classes.
- AUC = 1 implies that the model is a perfect classifier, meaning it achieves a TPR of 1 for all FPR values. This suggests that the model can completely separate the two classes without any errors.

- While precision, recall, and F1 score are threshold-specific, the AUC-ROC curve considers multiple thresholds simultaneously.

|Category|Task|Metric|Metric Summary|Reference|
|-----|-----|------|------|-----|
|-|Binary Classification|Confusion Matrix, Accuracy, Precision Recall and F1 Score|Shouldn’t use accuracy on imbalanced problems. Its easy to get a high accuracy score by simply classifying all observations as the majority class.|[Confusion Matrix, Accuracy, Precision, Recall, F1 Score](https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd)[Beyond Accuracy: Recall, Precision, F1-Score, ROC-AUC](https://medium.com/@priyankads/beyond-accuracy-recall-precision-f1-score-roc-auc-6ef2ce097966)|
|LLM / NLP|Text Summary & Translation|ROGUE|Used for evaluating test summarization and machine translation. Metric compares an automatically produced summary or translation against human-produced summary or translation. It measures how many of the n-grams in the references are in the predicted candidate.|[An intro to ROUGE, and how to use it to evaluate summaries](https://www.freecodecamp.org/news/what-is-rouge-and-how-it-works-for-evaluation-of-summaries-e059fb8ac840/)|
|LLM / NLP|Text Summary & Translation|Perplexity|Intuitively, perplexity means to be surprised. We measure how much the model is surprised by seeing new data. The lower the perplexity, the better the training is. Perplexity is calculated as exponent of the loss obtained from the model. Perplexity is usually used only to determine how well a model has learned the **training set**. Other metrics like BLEU, ROUGE etc., are used on the **test set** to measure test performance.|[Perplexity in Language Models](https://chiaracampagnola.io/2020/05/17/perplexity-in-language-models/)[Perplexity of Language Models](https://medium.com/@priyankads/perplexity-of-language-models-41160427ed72)|
|LLM / NLP|Text Summary & Translation|GLUE benchmark|GLUE benchmark that measures the general language understanding ability.|[Perplexity in Language Models](https://chiaracampagnola.io/2020/05/17/perplexity-in-language-models/)[Perplexity of Language Models](https://medium.com/@priyankads/perplexity-of-language-models-41160427ed72)|

## Activation Functions

Activation functions transforms the weighted sum of a neuron so that the output is non-linear.
Activation function decides whether a neuron should be activated or not.

<img src="https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/627d12431fbd5e61913b7423_60be4975a399c635d06ea853_hero_image_activation_func_dark.png" height="35%" width="35%">

#### [Cheatsheet](https://miro.medium.com/v2/resize:fit:720/format:webp/1*o7sNtf4Cmou-3eSW35Sx4g.png)

### Sigmoid or Logistic Activtion Function

It is generally used in logistic regression and binary classification models in the output layer.

The output of sigmoid activation function lies between 0 and 1, making it perfect to model probability. Hence it is used to convert the real-valued output of a linear layer to a probability.

The function is differentiable but saturates quickly because of the boundedness leading to a vanishing gradient when used in a deep neural network. Contributes to the vanishing gradient problem.

https://machinelearningmastery.com/using-activation-functions-in-neural-networks/

https://www.v7labs.com/blog/neural-networks-activation-functions

### Softmax

Extension of sigmoid activation function taking advantage of range of the output between 0 and 1. This is mainly used in the output layer of a multiclass, multinomial classification problem with a useful property of sum of the output probabilities adding up to 1.

Softmax is used for multi-classification, the probabilities sum will be 1.

### Tanh or hyperbolic tangent Activation Function
tanh is also like logistic sigmoid but better. The range of the tanh function is from (-1 to 1). tanh is also sigmoidal (s - shaped).

### ReLU (Rectified Linear Unit) Activation Function

The ReLU is the most used activation function in the world right now.Since, it is used in almost all the convolutional neural networks or deep learning.


### Swish
### GeLU
https://towardsdatascience.com/fantastic-activation-functions-and-when-to-use-them-481fe2bb2bde
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

## Loss Functions

In neural networks, loss functions help optimize the performance of the model. They are usually used to measure some penalty that the model incurs on its predictions, such as the deviation of the prediction away from the ground truth label.

All machine learning models are one optimization problem or another, the loss is the objective function to minimize. In neural networks, the optimization is done with gradient descent and backpropagation. [Source](https://machinelearningmastery.com/loss-functions-in-pytorch-models/)
### Loss functions for Regression
- Mean Absolute Error (MAE)
- Mean Square Error (MSE)
For details refer [this](https://machinelearningmastery.com/loss-functions-in-pytorch-models/)

### Loss functions for classification

### Entropy
**Entropy** measures the degree of randomness.
https://www.javatpoint.com/entropy-in-machine-learning

#### Cross Entropy Loss
https://datajello.com/cross-entropy-and-negative-log-likelihood/

Cross refers to the fact that it needs to relate two distributions. It’s called the cross entropy of distribution q relative to a distribution p.
- p is the true distribution of X (this is the label of the y value in a ML problem)
- q is the estimated (observed) distribution of X (this is the predicted value of y-hat value in a ML problem)

##### Log Loss - Binary Cross-Entropy Loss


References

- https://sharkyun.medium.com/complete-guide-to-confusion-matrix-accuracy-precision-recall-and-f1-score-easy-to-understand-8772c2403df3

