# Neural Networks / Deeplearning

A Neural Network introductory videos with nice intutions. A 4-part series that explaines neural networks very intutively.

- :tv: [What is a neural network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- :tv: [Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3)
- :tv: [What is backpropagation really doing? | Chapter 3, Deep learning](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4)
- :tv: [Backpropagation calculus | Chapter 4, Deep learning](https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=5)


## Neural Networks

### Deep Learning

|Rating|Type|Topic
------------: | ------------- | -------------
|:star:|:tv:|[MIT Deep Learning Course - Lex Fridman](https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf)

#### CNN

|Rating|Type|Topic
------------: | ------------- | -------------
|:star:|:tv:|[Nice intution about CNN, Convolutional Neural Networks](https://www.youtube.com/watch?v=xg2ajb3csgk&list=PLXAoLgwZtKcgGE2-Wy23EUE4Q03s-YVwF&index=3)|

#### RNN

|Rating|Type|Topic
------------: | ------------- | -------------
|:star:|:tv:|[A friendly introduction to Recurrent Neural Networks](https://www.youtube.com/watch?v=UNmqTiOnRfg)

### Activation Functions

Activation functions transforms the weighted sum of a neuron so that the output is non-linear

#### Sigmoid

The most common sigmoid function used in machine learning is Logistic Function, as the formula below.
https://miro.medium.com/v2/resize:fit:382/format:webp/1*VE7kb7J2lEo5zsUGyHgMWQ.png

The formula is simple, but it is quite useful because it offers us some nice properties:
* It maps the feature space into probability functions
* It uses exponential
* It is differentiable


Sigmoid is used for binary classification

#### Softmax

The Softmax function is a generalized form of the logistic function

the softmax function also has the following advantages so that people are widely using it in multi-class classification problems:

* It maps the feature space into probability functions
* It uses exponential
* It is differentiable

Softmax is used for multi-classification
The probabilities sum will be 1



References: 
* https://towardsdatascience.com/understanding-sigmoid-logistic-softmax-functions-and-cross-entropy-loss-log-loss-dbbbe0a17efb

### Loss Vs. Optimizer

Think of loss function has what to minimize and optimizer how to minimize the loss.

The *loss* is way of measuring difference between target label(s) and prediction label(s).
* Loss could be "mean squared error", "mean absolute error loss also known as L2 Loss", "Cross Entropy" ... and in order to reduce it, weights and biases are updated after each epoch. Optimizer is used to calculate and update them.

The optimization strategies aim at minimizing the cost function.

#### Loss Function Vs. Cost Function
Cost function and loss function are synonymous and used interchangeably, they are different.

A loss function is for a single training example. It is also sometimes called an error function. A cost function, on the other hand, is the average loss over the entire training dataset.

the log magnifies the mistake in the classification, so the misclassification will be penalized much more heavily compared to any linear loss functions. The closer the predicted value is to the opposite of the true value, the higher the loss will be, which will eventually become infinity. That’s exactly what we want a loss function to be.