# Evaluation Metrics

- [ROUGE](#ROUGE)
- [BLEU](#BLEU)


**Text Summarization**

**Extractive vs Abstractive**

There are two types of text summarization that a human, and nowadays a machine, can do.

- **Extractive**: Words and phrases are directly extracted from the text.
- **Abstractive**: Words and phrases are generated semantically consistent, ensuring the key information of the original text is maintained.

How do we measure the accuracy of a language-based sequence when dealing with language summarization or translation?

## ROUGE

ROUGE is used as an initial indicator of how much the machine-written summary overlaps with the human written summary, because it _does not take into account the semantic meaning and the factual accuracy_ of the summaries.

#### ROUGE-N

ROUGE-N measures the number of matching n-grams between the model-generated text and a human-produced reference.

Refer to this [article](https://medium.com/nlplanet/two-minutes-nlp-learn-the-rouge-metric-by-examples-f179cc285499) to calculate ROUGE 1 recall/precision/F1 Score and ROUGE 2 recall/precision/F1 Score.

#### ROUGE-L 

ROUGE-L is based on the longest common subsequence (LCS) between model-generated text and a human-produced reference.

- The longest sequence of words (not necessarily consecutive, but still in order) that is shared between both.
- A longer shared sequence should indicate more similarity between the two sequences.

Refer to this [article](https://medium.com/nlplanet/two-minutes-nlp-learn-the-rouge-metric-by-examples-f179cc285499) to compute ROUGE-L recall, precision, and F1-score.

#### ROUGE-S

ROUGE-S allows us to add a degree of leniency to the n-gram matching performed with ROUGE-N and ROUGE-L. ROUGE-S is a skip-gram concurrence metric: this allows to search for consecutive words from the reference text that appear in the model output but are separated by one-or-more other words.

- **BLEU focuses on precision**: how many the words (and/or n-grams) in the machine-generated text appear in the human-produced reference.
- **ROUGE focuses on recall**: how many the words (and/or n-grams) in the human-produced references appear in the machine-generated model outputs.

## BLEU

BLEU, or the Bilingual Evaluation Understudy, is a metric for comparing a candidate translation to one or more reference translations.

Although developed for translation, it can be used to evaluate text generated for different natural language processing tasks, such as paraphrasing and text summarization.

The BLEU score is not perfect, but it’s quick and inexpensive to calculate, language-independent, and, above all, correlates highly with human evaluation.

https://medium.com/nlplanet/two-minutes-nlp-learn-the-bleu-metric-by-examples-df015ca73a86

