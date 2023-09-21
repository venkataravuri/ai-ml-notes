
## FAQ

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
