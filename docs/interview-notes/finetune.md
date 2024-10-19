

#### GPTQ vs. BitsandBytes (NF4)

NF4 is part of the bitsandbytes library,
- **Normalization**: Weights are normalized before quantization, allowing for efficient representation of common values.
- **4-bit Quantization**: Weights are quantized to 4 bits with evenly spaced levels based on normalized weights.
- **Dequantization During Computation**: Although weights are stored in 4-bit format, they are dequantized during computation to enhance performance.


GPTQ stands for Post-Training Quantization,
- **Dynamic Dequantization**: Weights are dynamically dequantized to float16 during inference, which improves performance while keeping memory requirements low.
- **Custom Kernels**: Utilizes specialized kernels for matrix-vector operations, resulting in faster processing speeds compared to other methods like bitsandbytes and GGML.
- **Performance Metrics**: In tests with models like Llama-7B, GPTQ showed lower perplexity (PPL) scores and higher token generation rates than both GGML and NF4


#### Skip Connections vs. Residual
there are two fundamental ways that one could use skip connections through different non-sequential layers:
a) **addition** as in residual architectures,
b) **concatenation** as in densely connected architectures.

<img src="https://theaisummer.com/static/8d19d048cd68d6dce362e025cf3b635a/1ac66/skip-connection.png"/>
<img src="https://theaisummer.com/static/b8156f7a258e0c46eb1e5e7b6bb591bf/ad12c/resnet-concatenation.png" />
