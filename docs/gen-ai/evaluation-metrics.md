# Evaluation Metrics

**Text Summarization, Language Translation**
- [ROUGE](#rouge)
- [BLEU](#bleu)
- [Perplexity](#perplexity)

**Speech Recognition**

---

**Text Summarization**

**Extractive vs Abstractive**

There are two types of text summarization that a human, and nowadays a machine, can do.

- **Extractive**: Words and phrases are directly extracted from the text.
- **Abstractive**: Words and phrases are generated semantically consistent, ensuring the key information of the original text is maintained.

How do we measure the accuracy of a language-based sequence when dealing with language summarization or translation?

---

## ROUGE

ROUGE is used as an initial indicator of how much the machine-written summary overlaps with the human written summary, because it _does not take into account the semantic meaning and the factual accuracy_ of the summaries.

#### ROUGE-N

ROUGE-N measures the number of matching n-grams between the model-generated text and a human-produced reference.

Refer to this [article](https://medium.com/nlplanet/two-minutes-nlp-learn-the-rouge-metric-by-examples-f179cc285499) to calculate ROUGE 1 recall/precision/F1 Score and ROUGE 2 recall/precision/F1 Score.

<img src="https://2.bp.blogspot.com/-Epc-MOVyeck/WrY91dsmqtI/AAAAAAAAAAM/JCP9qck4RbAMVGz7ZqTAnO2ZtkpdK_D4gCLcBGAs/s1600/rougeN.jpg" width="50%" height="50%" />

#### ROUGE-L 

ROUGE-L is based on the longest common subsequence (LCS) between model-generated text and a human-produced reference.

- The longest sequence of words (not necessarily consecutive, but still in order) that is shared between both.
- A longer shared sequence should indicate more similarity between the two sequences.

Refer to this [article](https://medium.com/nlplanet/two-minutes-nlp-learn-the-rouge-metric-by-examples-f179cc285499) to compute ROUGE-L recall, precision, and F1-score.

<img src="https://image2.slideserve.com/4707631/rouge-l2-l.jpg" width="50%" height="50%" />

#### ROUGE-S

ROUGE-S allows us to add a degree of leniency to the n-gram matching performed with ROUGE-N and ROUGE-L. ROUGE-S is a skip-gram concurrence metric: this allows to search for consecutive words from the reference text that appear in the model output but are separated by one-or-more other words.

- **BLEU focuses on precision**: how many the words (and/or n-grams) in the machine-generated text appear in the human-produced reference.
- **ROUGE focuses on recall**: how many the words (and/or n-grams) in the human-produced references appear in the machine-generated model outputs.

---

## BLEU

BLEU, or the Bilingual Evaluation Understudy, is a metric for comparing a candidate translation to one or more reference translations.

Although developed for translation, it can be used to evaluate text generated for different natural language processing tasks, such as paraphrasing and text summarization.

The BLEU score is not perfect, but it’s quick and inexpensive to calculate, language-independent, and, above all, correlates highly with human evaluation.

https://medium.com/nlplanet/two-minutes-nlp-learn-the-bleu-metric-by-examples-df015ca73a86

<img src="https://3.bp.blogspot.com/-FQarElbZHfI/XLKkRHizgYI/AAAAAAAAQnI/iN2JD-K5tscj-8Jmar6tisOtr0f43s92wCLcBGAs/s1600/BLEU1.png" width="70%" height="70%" />

<img src="https://3.bp.blogspot.com/-HDoHlz3t9eo/WO1TxDJfSRI/AAAAAAAAISI/7B5FhctDglkxexKD_WSzTiR87h2B_OlXQCLcB/s1600/BLEU_94.jpg" width="70%" height="70%" />

## Perplexity

A language model is a probability distribution over sentences.
A language model is a probability matrix between a word and the next word that occurs in the corpus of the training set

Perplexity, known as PP, is “the inverse probability of the test set, normalised by the number of words”. In the Perplexity equation below, there are N words in a sentence, and each word is represented as w, where P is the probability of each w after the previous one. Also, we can expand the probability of W using the chain rule as followed.

$PP(W) = P(w_1 w_2 w_3 ... w_N)^{\frac{1}{N}}$

$= \sqrt[N]{\frac{1}{P(w_1 w_2 w_3 ... w_N)}}$

$= \sqrt[N]{\Pi_{i=1}^{N}\frac{1}{P(w_i\|w_{i-1})}}$

https://medium.com/nlplanet/two-minutes-nlp-perplexity-explained-with-simple-probabilities-6cdc46884584

---

## WER - Word Error Rate

Word error rate (WER) is a common metric of the performance of an automatic speech recognition (ASR) system.

The WER is derived from the Levenshtein distance, working at the word level.

$WER = \frac{(S + D + I)}{N}$

where,
S = Number of substitutions
D = Number of deletions
I = Number of insertions
N = Number of words in the reference.

