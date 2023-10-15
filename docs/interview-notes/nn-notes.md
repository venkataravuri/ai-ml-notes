# Neural Networks - Interview Notes

- [Describe several attention mechanisms, what are advantages and disadvantages?](#describe-several-attention-mechanisms-what-are-advantages-and-disadvantages)
- [Why do we need positional encoding in transformers?](#why-do-we-need-positional-encoding-in-transformers)
- [Describe convolution types and the motivation behind them.](#describe-convolution-types-and-the-motivation-behind-them)
- [How would you prevent a neural network from overfitting? How does Dropout prevent overfitting? Does it differ for train and test? How will you implement dropout during forward and backward passes?](#how-would-you-prevent-a-neural-network-from-overfitting-how-does-dropout-prevent-overfitting-does-it-differ-for-train-and-test-how-will-you-implement-dropout-during-forward-and-backward-passes)
- [What is Transfer Learning? Give an example.](#what-is-transfer-learning-give-an-example)
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()

---

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

