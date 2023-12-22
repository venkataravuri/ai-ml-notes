import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,),)])

train_data = datasets.FashionMNIST('./fashion_mnist_data', download=True, train=True, transform=transform)
test_data = datasets.FashionMNIST('./fashion_mnist_data', download=True, train=False, transform=transform)

