

### Describe several attention mechanisms, what are advantages and disadvantages?

|Scaled-Dot Product Attention Mechanism|Multi-Head Attention Mechanism|Additive-Attention Mechanism|Self Attention|Entity-Aware Attention<br/>Location-Based Attention<br/>Global & Local Attentions|
|---|---|---|---|---|
|Scaled dot product is a specific formulation of multiplicative attention mechanism that calculates attention weights using dot-product of query and key vectors followed by proper scaling to prevent the dot-product from growing too large.|Instead of performing and obtaining a single Attention on large matrices Q, K and V, it is found to be more effecient to divide them into multiple matrices of smaller dimensions and perform Scaled-Dot Product on each of those smaller matrices.<br/>MultiHead attention uses multiple attention heads to attend to different parts of the input sequence which allows the model to learn different relationships between the different parts of the input sequence. For example, the model can learn to attend to the local context of a word, as well as the global context of the entire input sequence.|This type of mechanism is used in neural networks to learn long-range dependencies between different parts of a sequence. It works by computing a weighted sum of the hidden states of the encoder, where the weights are determined by how relevant each hidden state is to the current decoding step. First, the encoder reads the input sequence and produces a sequence of hidden states which represent the encoder's understanding of the input sequence.|It is a mechanism that allows a model to attend to different parts of the same input sequence. This is done by computing a weighted sum of the input sequence, where the weights are determined by how relevant each part of the sequence is to the current task.The basic idea behind self-attention is to compute attention weights for each word/token in a sequence with respect to all other words/tokens in the same sequence. These attention weights indicate the importance or relevance of each word/token to the others.||
|<img src="https://iq.opengenus.org/content/images/2023/05/Attention-Mechanism-2.png"/>|<img src="https://iq.opengenus.org/content/images/2023/05/MultiHead-Attention-1.png" />||||

### Why do we need positional encoding in transformers?

### Describe convolution types and the motivation behind them.

Refer to [Convolution Operations](https://github.com/venkataravuri/ai-ml/blob/master/docs/neural-networks-deeplearning.md#convolutional-neural-networks-cnn) notes.

