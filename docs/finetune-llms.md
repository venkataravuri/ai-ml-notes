# Finetune LLMs

## Concepts

### Prompt Tuning

prompt tuning (different from prompting) appends a tensor to the embedded inputs of a pretrained LLM.
The tensor is then tuned to optimize a loss function for the finetuning task and data while all other parameters in the LLM remain frozen.

#### Hard Prompting

Prompt tuning refers to techniques that vary the input prompt to achieve better modeling results. 

For example, suppose we are interested in translating an English sentence into German. We can ask the model in various different ways, as illustrated below.
An example of hard prompt tuning, that is, rearranging the input to get better outputs.

```
1) Translate the English sentence '{English Sentence}' into German language '{German Translation}'
2) English: '{English Sentence}' | German: '{German Translation}'
3) From English to German: '{English Sentence}' -> '{German Translation}'
```

Now, this concept illustrated above is referred to as **hard prompt tuning** since we directly change the discrete input tokens, which are not differentiable.

[Source](https://magazine.sebastianraschka.com/p/understanding-parameter-efficient) 

#### Soft Prompting

In contrast to hard prompt tuning, soft prompt tuning (Lester et al. 2021) concatenates the embeddings of
the input tokens with a trainable tensor that can be optimized via backpropagation to improve the modeling performance on a target task.

Soft prompt tuning is significantly more parameter-efficient than full-finetuning.


### Prefix Tuning

Prefix tuning is to add trainable tensors to each transformer block instead of only the input embeddings, as in soft prompt tuning. 

## Parameter-Efficient Finetuning Methods

The main idea behind prompt tuning, and parameter-efficient finetuning methods in general, is to add a small number of new parameters
to a pretrained LLM and only finetune the newly added parameters to make the LLM perform better on,
- (a) a target dataset (for example, a domain-specific dataset like medical or legal documents)
- and (b) a target task (for example, sentiment classification).

### Finetune Llama 2
