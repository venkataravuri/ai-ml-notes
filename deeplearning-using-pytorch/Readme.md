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
