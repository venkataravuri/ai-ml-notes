import numpy as np
import torch
a = torch.ones((2, 2), requires_grad = True)

print(a)

b = a + 5

c = a.mean()

print(b, c)

c.backward()


print(a.grad)