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


