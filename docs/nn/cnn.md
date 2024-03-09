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
