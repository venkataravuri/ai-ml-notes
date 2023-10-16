# Evaluation Metrics

- [ROUGE]()


## ROUGE

**Text Summarization**

**Extractive vs Abstractive**

There are two types of text summarization that a human, and nowadays a machine, can do.

- **Extractive**: Words and phrases are directly extracted from the text.
- **Abstractive**: Words and phrases are generated semantically consistent, ensuring the key information of the original text is maintained.

 How do we measure the accuracy of a language-based sequence when dealing with language summarization or translation?

ROUGE is best used only as an initial indicator of how much the machine-written summary overlaps with the human written summary, because it does not take into account the semantic meaning and the factual accuracy of the summaries..

ROUGE-N measures the number of matching ‘n-grams’ between our model-generated text and a ‘reference’.

ROUGE L - ROUGE longest common subsequence will give us the longest overlap.
