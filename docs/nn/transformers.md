# Transformers Architecture


## Positional Encoding

Positional encoding describes the location or position of an entity in a sequence so that each position is assigned a unique representation.

There are many reasons why a single number, such as the index value, is not used to represent an item’s position in transformer models. For long sequences, the indices can grow large in magnitude. 

If you normalize the index value to lie between 0 and 1, it can create problems for variable length sequences as they would be normalized differently.

Transformers use a smart positional encoding scheme, where each position/index is mapped to a vector.

Hence, the **output of the positional encoding layer is a matrix**, where each row of the matrix represents an encoded object of the sequence summed with its positional information. 

An example of the matrix that encodes only the positional information is shown in the figure below.

<img src="https://machinelearningmastery.com/wp-content/uploads/2022/01/PE3.png" />

**Positional Encoding Layer in Transformers**

 Suppose you have an input sequence of length **L** and require the position of the **Kth** object within this sequence. The positional encoding is given by sine and cosine functions of varying frequencies:

$P(k, 2i) = sin(\frac{k}{n^{2i/d}})$

$P(k, 2i+1) = cos(\frac{k}{n^{2i/d}})$

Here:

k: Position of an object in the input sequence,

d: Dimension of the output embedding space

P(k,j): Position function for mapping a position in the input sequence to index of the positional matrix

n: User-defined scalar, set to 10,000 by the authors of Attention Is All You Need.

i: Used for mapping to column indices , with a single value of maps to both sine and cosine functions

For complete notes refer to, [A Gentle Introduction to Positional Encoding in Transformer Models, Part 1](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/)

