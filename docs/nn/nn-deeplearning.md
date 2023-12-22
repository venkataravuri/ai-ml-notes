# Neural Networks & Deep Learning

### Artificial Neuron / Perceptron

<img src="https://miro.medium.com/v2/resize:fit:640/1*sPg-0hha7o3iNPjY4n-vow.jpeg" width="50%" height="50%" />

#### Forward Pass

Forward pass computes network output and “error”

Input data -> Neural Network -> Prediction

#### Backpropagation (Backward pass)

- Backward pass to compute gradients
- A fraction of the weight’s gradient is subtracted from the weight. (based on learning rate)

Neural Network <- Measure of Error

Adujst to reduce error

Learning is an Optimization Problem

-Update the weights and biases to decrease loss function

A Neural Network introductory videos with nice intutions. A 4-part series that explaines neural networks very intutively.

- :tv: [What is a neural network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- :tv: [Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3)
- :tv: [What is backpropagation really doing? | Chapter 3, Deep learning](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4)
- :tv: [Backpropagation calculus | Chapter 4, Deep learning](https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=5)


## Neural Networks

## Deep Learning

Deep learning is a type of neural networks with multiple layers which can handle higher-level computation tasks, such as natural language processing, fraud detection, autonomous vehicle driving, and image recognition.

Deep learning models and their neural networks include the following:
- Convolution neural network (CNN).
- Recurrent neural network (RNN).
- Generative adversarial network (GAN).
- Autoencoder.
- Generative pre-trained transformer (GPT).

Deep Learning modeling is still an art, in other words, intelligent trial and error. There is no universal answer for NNs.

### Convolutional Neural Networks (CNN)

#### CNN Layer Types

There are many types of layers used to build Convolutional Neural Networks, but the ones you are most likely to encounter include:

* Convolutional (CONV)
* Activation (ACT or RELU, where we use the same or the actual activation function)
* Pooling (POOL)
* Fully connected (FC)
* Batch normalization (BN)
* Dropout (DO)

Stacking a series of these layers in a specific manner yields a CNN. 

A simple text diagram to describe a CNN: ```INPUT => CONV => RELU => FC => SOFTMAX```

### Convolution Operation

The convolutional operation is implemented by making The kernel slides across the image and produces an output Value at each position.

Convolution is using a ‘kernel’ to extract certain ‘features’ from an input image.

A kernel is a matrix, which is slid across the image and multiplied with the input such that the output is enhanced in a certain desirable manner.

Also we convolve different Kernels and as a result obtain Different feature maps or channels. to extract latent features.

<img src="https://miro.medium.com/v2/resize:fit:780/1*Eai425FYQQSNOaahTXqtgg.gif" height="50%" width="50%" />

the kernel used above is useful for sharpening the image.

Convolutional filters, also called kernels are designed to detect specific patterns or features in the input data.

in image processing, filters might be designed to detect edges, corners, or textures. In deep learning, the weights of these filters are learned automatically through training on large datasets.

**Variants of The Convolution Operation**

|Valid Convolution|Strided Convolution|Dilated Convolution|Depth wise Convolution|
|---|---|---|---|
|<img src="https://miro.medium.com/v2/resize:fit:578/format:webp/1*8QgzufBR-FofT8OAjnKSow.png" height="70%" weight="70%" />|<img src="https://miro.medium.com/v2/resize:fit:436/format:webp/1*9h-pnJxNKwRi9ft9ljQatg.png" height="70%" weight="70%" />|<img src="https://miro.medium.com/v2/resize:fit:646/format:webp/1*eUjPo__YgjupAKV5rg4MSw.png" height="70%" weight="70%" />|<img src="https://miro.medium.com/v2/resize:fit:668/format:webp/1*S_pnYr5LMrWk4oXEqpj6bA.png" height="70%" weight="70%" />|<img src="" height="70%" weight="70%" />|
|Doesn’t used any padding|kernel slides along the image with a step > 1|kernel is spread out, step > 1 between kernel elements|each output channel is connected only to one input channel|

Take an input FloatTensor with torch.Size = [10, 3, 28, 28] in NCHW order,
and apply nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))

in channels cin = 3
out channels cout = 16
number of filters f = 16
size of filters k = 5
stride s = 1
padding p = 0
height in h = 28
width in w = 28

The formula used to calculate the output shape is:

output_shape = ((input_height - kernel_size + 2 * padding) / stride) + 1
= ((28 - 5 + 2 * 1)/(1) + 1)
= 24

Output shape: 24 x 24 x 16

Conv2D calculator: https://abdumhmd.github.io/files/conv2d.html

http://layer-calc.com/

https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573

### Pooling
A pooling function replaces the output of the net at a certain location with a summary statistic of the nearby outputs. 
For example, 
- **Max pooling** operation reports the maximum output within a rectangular neighborhood. 
- Others,
  - Average of a rectangular neighborhood
  - L2 norm of a rectangular neighborhood
  - A weighted average based on the distance from the central pixel.

In all cases, pooling helps to make the representation become approximately invariant to small translations of the input. Invariance to translation means that if we translate the input by a small amount, the values of most of the pooled outputs do not change.

Understanding the **Receptive Field** of Convolutional Layer

For large inputs, we need many layers to understand the whole input. We can downsample the features by using stride, kernel_size and max_pooling. They increase the receptive field. The receptive field essentially expresses how much information a later layer contains of the first input layer. Consider the example of a 1D array of length 7, where we apply a 1D kernel of size 3. On the left we see that the 1D array length decreases from 7 to 5 in the second layer due to the convolutional operation.

[Source](https://pyimagesearch.com/2021/05/14/convolutional-neural-networks-cnns-and-layer-types/)

### NN Vs. RNN

Fundamental difference in the preservation of the network state, which is absent in the feed-forward network.

#### Generative Adversarial Network (GAN)

GANs are a way to make generative model by having two neural network models compete each other.

- [Face Generation Using Generative Adversarial Networks (GAN)](https://medium.com/nerd-for-tech/face-generation-using-generative-adversarial-networks-gan-6d279c2d5759)

- [Deep Convolutional Generative Adversarial Networks(DCGANs)](https://medium.datadriveninvestor.com/deep-convolutional-generative-adversarial-networks-dcgans-3176238b5a3d)

#### Diffusions



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

<img src="https://miro.medium.com/v2/resize:fit:382/format:webp/1*VE7kb7J2lEo5zsUGyHgMWQ.png" />

It has following properties:
* It maps the feature space into probability functions
* It uses exponential
* It is differentiable

Sigmoid function has values that are extremely near to 0 or 1. Because of this, it is appropriate for use in **binary classification problems**.

#### Softmax

The Softmax function is a generalized form of the logistic function.

Softmax function has following characterstics and widely used in multi-class classification problems:
* It maps the feature space into probability functions
* It uses exponential
* It is differentiable

Softmax is used for multi-classification. The probabilities sum will be 1

References: 
* https://towardsdatascience.com/understanding-sigmoid-logistic-softmax-functions-and-cross-entropy-loss-log-loss-dbbbe0a17efb

### Loss Vs. Optimizer

Think of loss function has what to minimize and optimizer how to minimize the loss.

The *loss* is way of measuring difference between target label(s) and prediction label(s).
* Loss could be "mean squared error", "mean absolute error loss also known as L2 Loss", "Cross Entropy" ... and in order to reduce it, weights and biases are updated after each epoch. Optimizer is used to calculate and update them.

The optimization strategies aim at minimizing the cost function.

#### Loss Function

Loss function quantifies gap between prediction and ground truth.

For regression:
- Mean Squared Error (MSE)

For classification:
- Cross Entropy Loss

##### Loss Function Vs. Cost Function

Cost function and loss function are synonymous and used interchangeably, they are different.

A loss function is for a single training example. It is also sometimes called an error function. A cost function, on the other hand, is the average loss over the entire training dataset.

the log magnifies the mistake in the classification, so the misclassification will be penalized much more heavily compared to any linear loss functions. The closer the predicted value is to the opposite of the true value, the higher the loss will be, which will eventually become infinity. That’s exactly what we want a loss function to be.

#### Cross Entropy Loss

when designing a neural network multi-class classifier, you can you CrossEntropyLoss with no activation, or you can use NLLLoss with log-SoftMax activation.

#### Overfitting & Regularization

Help the network generalize to data it hasn’t seen.

Overfitting: The error decreases in the training set but increases in the test set
Overfitting example (a sine curve vs 9-degree polynomial)

Regularization

Early Stoppage - Stop training (or at least save a checkpoint) when performance on the validation set decreases

Dropout - Randomly remove some nodes in the network (along with incoming and outgoing edges)
- Usually p >= 0.5 (pis probability of keeping node)
- Input layers pshould be much higher (and use noise instead of dropout)
- Most deep learning frameworks come with a dropout layer


Regularization: Weight Penalty (aka Weight Decay)


Batch Normalization (BatchNorm, BN)
- Normalize hidden layer inputsto mini-batch mean & variance
- Reduces impact of earlier layers on later layers




# References

https://www.dropbox.com/s/c0g3sc1shi63x3q/deep_learning_basics.pdf?dl=0

