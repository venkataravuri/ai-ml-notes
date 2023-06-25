# Machine Learning

- [Evaulation Metrics](#evaluation-metrics)


## Feature Engineering

|Rating|Type|Topic
------------: | ------------- | -------------
|:star:|:newspaper:|[Dealing with missing values](https://www.kaggle.com/alexisbcook/missing-values)

## General FAQ
|Question|Notes|Reference|
|------------| ------------- | -------------|
|What are Logits in machine learning?|Logits interpreted to be the unnormalised (or not-yet normalised) predictions (or outputs) of a model. These can give results, but we don't normally stop with logits, because interpreting their raw values is not easy.|[Logits Explanation](https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean)|

### Evaluation Metrics

|Category|Task|Metric|Metric Summary|Reference|
|-----|-----|------|------|-----|
|-|Binary Classification|Confusion Matrix, Accuracy, Precision Recall and F1 Score|Shouldn’t use accuracy on imbalanced problems. Its easy to get a high accuracy score by simply classifying all observations as the majority class.|[Confusion Matrix, Accuracy, Precision, Recall, F1 Score](https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd)[Beyond Accuracy: Recall, Precision, F1-Score, ROC-AUC](https://medium.com/@priyankads/beyond-accuracy-recall-precision-f1-score-roc-auc-6ef2ce097966)|
|LLM / NLP|Text Summary & Translation|ROGUE|Used for evaluating test summarization and machine translation. Metric compares an automatically produced summary or translation against human-produced summary or translation. It measures how many of the n-grams in the references are in the predicted candidate.|[An intro to ROUGE, and how to use it to evaluate summaries](https://www.freecodecamp.org/news/what-is-rouge-and-how-it-works-for-evaluation-of-summaries-e059fb8ac840/)|
|LLM / NLP|Text Summary & Translation|Perplexity|Intuitively, perplexity means to be surprised. We measure how much the model is surprised by seeing new data. The lower the perplexity, the better the training is. Perplexity is calculated as exponent of the loss obtained from the model. Perplexity is usually used only to determine how well a model has learned the **training set**. Other metrics like BLEU, ROUGE etc., are used on the **test set** to measure test performance.|[Perplexity in Language Models](https://chiaracampagnola.io/2020/05/17/perplexity-in-language-models/)[Perplexity of Language Models](https://medium.com/@priyankads/perplexity-of-language-models-41160427ed72)|
|LLM / NLP|Text Summary & Translation|GLUE benchmark|GLUE benchmark that measures the general language understanding ability.|[Perplexity in Language Models](https://chiaracampagnola.io/2020/05/17/perplexity-in-language-models/)[Perplexity of Language Models](https://medium.com/@priyankads/perplexity-of-language-models-41160427ed72)|

### Info

> Why we want to wrap everything with a logarithm?

1. It’s related to the concept in information theory where you need to use log(x) bits to capture x amount of information.
2. Computers are capable of almost anything, except exact numeric representation.

Entropy
https://www.javatpoint.com/entropy-in-machine-learning

#### Cross Entropy Loss

**Entropy** measures the degree of randomness.

https://datajello.com/cross-entropy-and-negative-log-likelihood/

Cross refers to the fact that it needs to relate two distributions. It’s called the cross entropy of distribution q relative to a distribution p.
- p is the true distribution of X (this is the label of the y value in a ML problem)
- q is the estimated (observed) distribution of X (this is the predicted value of y-hat value in a ML problem)

##### Log Loss - Binary Cross-Entropy Loss



# Natural Language Processing (NLP)

* [Tokenization](https://github.com/venkataravuri/learning-diary/blob/master/data-science.md#tokenization)
* [Stemming](https://github.com/venkataravuri/learning-diary/blob/master/data-science.md#stemming)
* [Lemmatization](https://github.com/venkataravuri/learning-diary/blob/master/data-science.md#lemmatization)
* [Stop Words Removal](https://github.com/venkataravuri/learning-diary/blob/master/data-science.md#stop-words-removal)
* [Parts of Speech (POS) Tagging](https://github.com/venkataravuri/learning-diary/blob/master/data-science.md#parts-of-speech)
* [TF-IDF](https://github.com/venkataravuri/learning-diary/blob/master/data-science.md#tf-idf)
* [Word Embeddings](https://github.com/venkataravuri/learning-diary/blob/master/data-science.md#word-embeddings)

## Text Analysis / Text Mining

### Tokenization

Break the text into words.

### TF-IDF

|Rating|Type|Topic
------------: | ------------- | -------------
|:star::star::star:|:newspaper:|[?](https://www.analyticsvidhya.com/blog/2020/02/quick-introduction-bag-of-words-bow-tf-idf/)
||:newspaper:|[?](https://www.slideshare.net/DanSullivan10/a-first-look-at-tf-idfpdx-data-science-meetup)

### Normalizing

|Rating|Type|Topic
------------: | ------------- | -------------
|:star:|:newspaper:|[?](https://qbox.io/blog/introduction-to-elasticsearch-analyzers)
||:newspaper:|[?](https://engineering.linkedin.com/blog/2019/04/how-natural-language-processing-help-support)
||:newspaper:|[?](https://logz.io/blog/language-analyzers-tokenizers-not-built-elasticsearch-where-find-them/)
||:newspaper:|[?](https://www.elastic.co/blog/text-classification-made-easy-with-elasticsearch)

#### Stemming

Helps normalize the spelling of words, for instance translating the tokens 'sing', 'sings', and 'singing' all into the single stem 'sing'.
In ElasticSeaarch, you can use 'snowball' analyzer.

#### Lemmatization

Find the basic form of each word variation in the context. Examples are "account" from "accounts" and "break" from "broke."

### Stop Word Removal

Stop Word Filter: Filter out common words. In English, there are hundreds of stop words like "a," "my," and "on," to name a few, that have little bearing on relevance or meaning, and thus can safely be removed from the query in order to target the more valuable words.

### Part of Speech (PoS) Tagging

Part of Speech (PoS) Filter: Read through text and give each word a PoS based on the context. There are nine parts in English: adjective, adverb, conjunction, determiner, noun, number, preposition, pronoun, and verb. We only capture nouns, verbs, proper nouns, and adjectives, as they together represent the purpose of a text.

### Word Embeddings

|Rating|Type|Topic
------------: | ------------- | -------------
|:star:|:newspaper:|[?](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)
