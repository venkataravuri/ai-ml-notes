# Neural Networks - Interview Notes

- [Describe several attention mechanisms, what are advantages and disadvantages?](#describe-several-attention-mechanisms-what-are-advantages-and-disadvantages)
- [Why do we need positional encoding in transformers?](#why-do-we-need-positional-encoding-in-transformers)
- [Describe convolution types and the motivation behind them.](#describe-convolution-types-and-the-motivation-behind-them)
- [How would you prevent a neural network from overfitting? How does Dropout prevent overfitting? Does it differ for train and test? How will you implement dropout during forward and backward passes?](#how-would-you-prevent-a-neural-network-from-overfitting-how-does-dropout-prevent-overfitting-does-it-differ-for-train-and-test-how-will-you-implement-dropout-during-forward-and-backward-passes)
- [What is Transfer Learning? Give an example.](#what-is-transfer-learning-give-an-example)
- [What is backpropagation?]()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()

---

PyTorch you define your Models as subclasses of torch.nn.Module.

The **__init__** function, initializes the layers that one want to use & specify the sizes of the network.

The **forward** method, specifies the connections between layers. 

## FAQ

#### Difference between torch.nn Vs. torch.nn.Functional?

- nn.Modules are defined as Python classes and have attributes, e.g. a ```nn.Conv2d``` module will have some internal attributes like ```self.weight```.
- nn.F.conv2d uses a functional (stateless) approach, just defines the operation and needs all arguments to be passed (including the weights and bias).
- Internally ```nn.Modules``` uses functional API in the forward method somewhere.
- A nn.Module is actually a OO wrapper around the functional interface, that contains a number of utility methods, like eval() and parameters(), and it automatically creates the parameters of the modules for you.
- For things that do not change between training/eval like sigmoid, relu, tanh, I think it makes sense to use functional.
- torch.nn module is more used for methods which have learnable parameters.
and functional for methods which do not have learnable parameters

[Source](https://stackoverflow.com/questions/42479902/what-does-view-do-in-pytorch)

```view()``` reshapes the tensor without copying memory. ```view()``` reshapes a tensor by 'stretching' or 'squeezing' its elements into the shape you specify:

```
import torch
a = torch.range(3, 4)
```

To reshape this tensor to make it a 2 x 6 tensor, use:
```
a = a.view(2, 6)
```

<img src="https://i.stack.imgur.com/ORqaP.png" width="60%" height="60%" />

How does view() work?

First let's look at what a tensor is under the hood:

<img src="https://i.stack.imgur.com/ee7Hj.png" width="60%" height="60%" />
<img src="https://i.stack.imgur.com/26Q9g.png" width="60%" height="60%" />

PyTorch makes a tensor by converting an underlying block of contiguous memory into a matrix-like object by adding a shape and stride attribute:
- **shape** states how long each dimension is
- **stride** states how many steps you need to take in memory til you reach the next element in each dimension

```view(dim1,dim2,...)``` returns a view of the same underlying information, but reshaped to a tensor of shape ```dim1 x dim2 x``` ... (by modifying the **shape** and **stride** attributes).

#### nn.Dropout

nn.Dropout is not a trainable layer
nn.Dropout changes behaviour when doing model.eval() and model.train()

#### PyTorch training with dropout and/or batch-normalization

All modules that make up a model may have two modes, training and test mode. These modules either have learnable parameters that need to be updated during training, like Batch Normalization or affect network topology in a sense like Dropout (by disabling some features during forward pass). some modules such as ReLU() only operate in one mode and thus do not have any change when modes change.

In ```.train()``` mode, when you feed an image, it passes trough layers until it faces a dropout and here, some features are disabled, thus their responses to the next layer is omitted, the output goes to other layers until it reaches the end of the network and you get a prediction.

the network may have correct or wrong predictions, which will accordingly update the weights. if the answer was right, the features/combinations of features that resulted in the correct answer will be positively affected and vice versa. So during training you do not need and should not disable dropout, as it affects the output and should be affecting it so that the model learns a better set of features.

In ```.eval()``` mode, before the forward pass, effectively disables all modules that has different phases for train/test mode such as Batch Normalization and Dropout (basically any module that has updateable/learnable parameters, or impacts network topology like dropout) will be disabled and you will not see them contributing to your network learning.

[Why Pytorch officially use mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] to normalize images?](https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2)

### Describe several attention mechanisms, what are advantages and disadvantages?

|Scaled-Dot Product Attention Mechanism|Multi-Head Attention Mechanism|Additive-Attention Mechanism|Self Attention|Entity-Aware Attention<br/>Location-Based Attention<br/>Global & Local Attentions|
|---|---|---|---|---|
|Scaled dot product is a specific formulation of multiplicative attention mechanism that calculates attention weights using dot-product of query and key vectors followed by proper scaling to prevent the dot-product from growing too large.|Instead of performing and obtaining a single Attention on large matrices Q, K and V, it is found to be more effecient to divide them into multiple matrices of smaller dimensions and perform Scaled-Dot Product on each of those smaller matrices.<br/>MultiHead attention uses multiple attention heads to attend to different parts of the input sequence which allows the model to learn different relationships between the different parts of the input sequence. For example, the model can learn to attend to the local context of a word, as well as the global context of the entire input sequence.|This type of mechanism is used in neural networks to learn long-range dependencies between different parts of a sequence. It works by computing a weighted sum of the hidden states of the encoder, where the weights are determined by how relevant each hidden state is to the current decoding step. First, the encoder reads the input sequence and produces a sequence of hidden states which represent the encoder's understanding of the input sequence.|It is a mechanism that allows a model to attend to different parts of the same input sequence. This is done by computing a weighted sum of the input sequence, where the weights are determined by how relevant each part of the sequence is to the current task.The basic idea behind self-attention is to compute attention weights for each word/token in a sequence with respect to all other words/tokens in the same sequence. These attention weights indicate the importance or relevance of each word/token to the others.||
|<img src="https://iq.opengenus.org/content/images/2023/05/Attention-Mechanism-2.png"/>|<img src="https://iq.opengenus.org/content/images/2023/05/MultiHead-Attention-1.png" />||||

### Why do we need positional encoding in transformers?

### How would you prevent a neural network from overfitting? How does Dropout prevent overfitting? Does it differ for train and test? How will you implement dropout during forward and backward passes?

Dropout is a technique to regularize in neural networks. When we drop certain nodes out, these units are not considered during a particular forward or backward pass in a network.

Dropout forces a neural network to learn more robust features that are useful in conjunction with many different random subsets of the other neurons. It will make the weights spread over the input features instead of focusing on just some features.

Dropout is used in the training phase to reduce the chance of overfitting. As you mention this layer deactivates certain neurons. The model will become more insensitive to weights of other nodes. Basically with the dropout layer the trained model will be the average of many thinned models.

The dropout will randomly mute some neurons in the neural network. At each training stage, individual nodes are either dropped out of the net 

dropout is disabled in test phase.

### Describe convolution types and the motivation behind them.

Refer to [Convolution Operations](https://github.com/venkataravuri/ai-ml/blob/master/docs/neural-networks-deeplearning.md#convolutional-neural-networks-cnn) notes.

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

---

### What is backpropagation?

- The goal of backpropagation is to update the weights for the neurons in order to minimize the loss function.
- Backpropagation takes the error from the previous forward propagation and feeds this error backward through the layers to update the weights. This process is iterated until the neural network model is converged.

### What is the loss function of a neural network model?

---

### When building a neural network model, how do you decide what should be the architecture of your model?

---

### What happens to the neural network gradients and weights when you initialize them with zeros?

Any constant initialization scheme will perform very poorly. Consider a neural network with two hidden units, and assume we initialize all the biases to 0 and the weights with some constant α. If we forward propagate an input (x1​, x2) in this network, the output of both hidden units will be relu(αx1​+αx2​). Thus, both hidden units will have identical influence on the cost, which will lead to identical gradients. Thus, both neurons will evolve symmetrically throughout training, effectively preventing different neurons from learning different things.

---

### Why do neural network weights need to be randomly initialized?

if you don’t initialize your weights randomly, you will end up with some problem called the symmetry problem where every neuron is going to learn kind of the same thing. To avoid that, you will make the neuron start at different places and let them evolve independently from each other as much as possible.

---

### Why sigmoid, Tanh is not used in hidden layers?

It results in the vanishing gradient problem and convergence will be slow towards global minima. A derivative of sigmoid lies between 0 to 0.25. A derivative of Tanh lies between 0 to 1. Because of this weight updates happen very slow.

---

### Why does Exploding Gradient problem happen?

This happens due to inappropriate weight initialization techniques. RELU works well with He initialization. Sigmoid and Tanh work well with Xavier Glorot initialization.

---

### When to use Sigmoid and Softmax activation?

When you are solving binary classification use Sigmoid and when you use Multiclass classification use Softmax.

---

### How do we decide the number of hidden layers and neurons that should be used?

---

### What is the difference between categorical_crossentropy and sparse_categorical_crossentropy?

- categorical_crossentropy (cce) produces a one-hot array containing the probable match for each category: [1,0,0] , [0,1,0], [0,0,1]
- sparse_categorical_crossentropy (scce) produces a category index of the most likely matching category: [1], [2], [3]

There are a number of situations to use scce, including:

- when your classes are mutually exclusive, i.e. you don’t care at all about other close-enough predictions,
- the number of categories is large to the prediction output becomes overwhelming.

### How can a neural network be used as a tool for dimensionality reduction? 

Yes, CNN does perform dimensionality reduction. A pooling layer is used for this. The main objective of Pooling is to reduce the spatial dimensions of a CNN.   

Deep auto-encoder

---

### Explain the significance of the RELU Activation function in Convolution Neural Network.

As a consequence, the usage of ReLU helps to prevent the exponential growth in the computation required to operate the neural network. If the CNN scales in size, the computational cost of adding extra ReLUs increases linearly   

### Why do we use a Pooling Layer in a CNN?

Pooling layers are used to reduce the spatial dimensions of feature maps generated by the convolutional layers. This process helps in reducing the computational complexity of the network and prevents overfitting.

The most common pooling operation is max-pooling, which takes the maximum value from a defined region (usually 2×2 or 3×3) of the feature map.

### What is the size of the feature map for a given input size image, Filter Size, Stride, and Padding amount?

We can compute the spatial size of the output volume as a function of the input volume size (W), the receptive field size of the Conv Layer neurons (F), the stride with which they are applied (S), and the amount of zero padding used (P) on the border. You can convince yourself that the correct formula for calculating how many neurons “fit” is given by (W−F+2P)/S+1. For example for a 7x7 input and a 3x3 filter with stride 1 and pad 0 we would get a 5x5 output. With stride 2 we would get a 3x3 output. L

### An input image has been converted into a matrix of size 12 X 12 along with a filter of size 3 X 3 with a Stride of 1. Determine the size of the convoluted matrix.

We can compute the spatial size of the output volume as a function of the input volume size (W), the receptive field size of the Conv Layer neurons (F), the stride with which they are applied (S), and the amount of zero padding used (P) on the border. You can convince yourself that the correct formula for calculating how many neurons “fit” is given by (W−F+2P)/S+1. For example for a 7x7 input and a 3x3 filter with stride 1 and pad 0 we would get a 5x5 output. With stride 2 we would get a 3x3 output.

### Explain the terms “Valid Padding” and “Same Padding” in CNN.

Padding refers to the process of adding a layer of zeros/ones to the sides of a matrix. We perform padding so that we can smoothly implement the convolution operation.  

To sum up, ‘valid’ padding means no padding. The output size of the convolutional layer shrinks depending on the input size & kernel size. On the contrary, ‘same’ padding means using padding.   

### What are the different types of Pooling? Explain their characteristics.


### Does the size of the feature map always reduce upon applying the filters? Explain why or why not.


### What is Stride? What is the effect of high Stride on the feature map?

Stride refers to the number of steps the filter matrix can shift after evaluating the convolution between input and filter. When the stride=1, the filter matrix shifts by one pixel; if it is 2, then the filter matrix must shift by two pixels.


### Explain the role of the flattening layer in CNN.

The flattening layer is usually towards the end of the CNN architecture, and it is used to transform all the two-dimensional matrices into a single lengthy vector. The output of this layer is passed to the fully-connected layer.


### List down the hyperparameters of a Pooling Layer.

The hyperparameters for a pooling layer are Filter size, Stride, Max, or average pooling.   

### What is the role of the Fully Connected (FC) Layer in CNN?

Fully connected layers serve as the final stage in a CNN architecture. They connect every neuron from the previous layer to every neuron in the subsequent layer, transforming the output from the convolutional and pooling layers into a single, continuous vector.

This vector is then passed through an activation function, such as a softmax function, to generate the final output probabilities for each clas

In an image classification task with 10 classes, the fully connected layer will output a 10-dimensional vector, with each element representing the probability of the input belonging to a specific class.

### Briefly explain the two major steps of CNN i.e, Feature Learning and Classification. 


### Let us consider a Convolutional Neural Network having three different convolutional layers in its architecture as –


### Explain the significance of “Parameter Sharing” and “Sparsity of connections” in CNN.


### What’s the difference between batch normalization and dropout layers in a CNN?

Batch Norm is a normalization technique done between the layers of a Neural Network instead of in the raw data. It is done along mini-batches instead of the full data set.
-  normalizing the inputs to take on a similar range of values can speed up learning.
-  Batch Norm reduces the internal covariate shift of the network.
-  it seems that Batch Norm has a regularization effect.  Because it is computed over mini-batches and not the entire data set, the model’s data distribution sees each time has some noise. This can act as a regularizer, 

Dropout is a _regularization technique_ that randomly drops out some fraction of the neurons in a layer during training. This helps to prevent overfitting by blocking information from certain neurons completely to make sure the _neurons do not co-adapt_. Dropout forces the network to learn more robust features that are useful in making predictions.

### What is the main purpose of using zero padding in a CNN?

Zero padding is a technique used in Convolutional Neural Networks (CNNs) to preserve the original input size.

It involves adding zeros to the borders of the input feature map when it is being processed by the kernel of a CNN.

The main purpose of using zero padding is to avoid losing information at the boundaries of the input feature map and to control the shrinkage of dimension after applying filters larger than 1x1.

### When would you prefer a 1D convolution over 2D convolutions?

1D convolutions are typically used when the input data is 1D, such as a time series or text. 2D convolutions are typically used when the input data is 2D, such as an image.

### What is deep residual learning?

Deep residual learning is a neural network architecture that allows for training very deep networks by alleviating the vanishing gradient problem. This is done by adding “shortcut” or “skip” connections between layers in the network, which allows for information to flow more freely between layers. This makes it easier for the network to learn complex functions, and results in better performance on tasks such as image classification.

<img src="https://i0.wp.com/developersbreach.com/wp-content/uploads/2020/08/cnn_banner.png?fit=1400%2C658&ssl=1" />