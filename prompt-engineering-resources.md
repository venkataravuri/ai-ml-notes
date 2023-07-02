# Prompt Engineering Resources

## Preread: How LLMs work?
Credits: [State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A) by Andrej Karpathy 

 Base Large Language Models are NOT AI Assistants
 - Base model does not answer questions
 - Just want to complete internet documents, they are "_document completers_".
 - Often **responds to questions with more questions**

> Based models are "_tricked into performing tasks_" with "_prompt engineering_".

> Based models can also be "_tricked into assistants_" with "_few shot prompting_".

https://karpathy.ai/stateofgpt.pdf

Large Language Models (LLM) undergo following "_training pipeline_" to become AI assistenats,
- Stage 1: **Un-supervised learning** using internnet archive, Wikipedia, books and more. Outcome is "_Base Model_"
- Stage 2: **Supervised Finetuning** with manually composed dataset "_prompt and ideal response_". Outcome "_SFT model_" can act as AI assistants.
- Stage 3: **Reward Modelling**, compare multiple completions of a prompt from SFT model and rank them. Perform binary classification reward best completion. Outcome  "Reward Model", cannot be used as assisstant"
- Stage 4: **Reinforcement Learning**, generate tokens to maximize reward. Outocome "Reinformcent Learning Model"

**Temperature** - In short, the lower the temperature the more deterministic the results in the sense that the highest probable next token is always picked. Increasing temperature could lead to more randomness encouraging more diverse or creative outputs

**Top_p** - 

## Prompt Engineering
Credits: [Source](https://github.com/shimon-d/prompt-eng-guide)

A prompt can contain information like the `instruction` or `question` you are passing to the model and including other details such as `inputs` or `examples`. Prompts can be used to perform all types of interesting and different tasks.

Instruct the model what you want to achieve such as "Write", "Classify", "Summarize", "Translate", "Order", etc.

- **Specificity**, more descriptive and detailed the prompt is, the better the results. This is particularly important when you have a desired outcome or style of generation you are seeking.

### Text Generation & Summarization
#### Talk is cheap, show me samples
[Best text summarization prompts](https://nerdschalk.com/chatgpt-prompts-to-summarize-text/)

**Samples**: 
List **pros and cons** for the <product> from this <text> ...
Summarize [text] into 2 paragraphs with **simple language to make it easier to understand**...
Summarize this **for me like I’m 8 years old** [text].

##### Hey, where can I use it at work?
- Creating marketing slogans or summary or bullet points based on product documentation or product guide.
- Classify customer email by understanding which product they are talking and what is this issue?

`Write a precise and concise summary of new release notes for Kubernetes version 1.28, including the main new features and improvements specified at below URL.
https://github.com/orgs/kubernetes/projects/140/views/1`

`Consider following while preparing release notes,`
1. Explain the new features  in simple, non-technical language, detailing how it is now more efficient and user-friendly.
2. The release notes should capture the reader’s attention and makes them want to learn more about the new release of Kubernetes version 1.28.
3. Categorize features specified in above URL into ‘New Features’, ‘Improvements’, ‘Bug Fixes’, and ‘Known Issues’.” 
4. Describe how to use the new features in Kubernetes version 1.28 in a way that would allow me to create a GIF or a short video clip to illustrate it.

### Reasoning

:star::star::star: https://www.allabtai.com/the-gpt-4-jug-problem-prompt/
    
### Question Answering

### Image Gnerators
https://www.allabtai.com/master-midjourney-in-1-minute/

https://labs.openai.com/e/HmoeRVI28qQX9pfeJvJr2ePB
https://dreamstudio.ai/generate

### Text Classification

### Conversation

### Code Generation

## Prompt Engineering Guide
- :star::star: [Learn Prompting](https://learnprompting.org/docs/intro)
- :star: [Prompt Engineering Guide](https://www.promptingguide.ai/)

## Prompt Engineering Marketplace

https://hero.page/samir

## System Roles

https://www.allabtai.com/chatgpt-gpt4-system-prompt-engineering-ultimate-guide/
https://www.allabtai.com/chatgpt-4-prompt-engineering-the-ultimate-problem-solver-prompt/

## Prompt Techniques

## Chain of Thought Prompting

Chain-of-thought prompting is an approach to improve the reasoning ability of large language models in arithmetic, commonsense, and symbolic reasoning tasks.
- It augments few-shot prompting with intermediate natural language reasoning steps.

https://paperswithcode.com/paper/chain-of-thought-prompting-elicits-reasoning

## Zero-shot CoT prompting

### ReAct - Reason and Acttion Prompting


## Plan and Execute Prompting



https://nordnet.blob.core.windows.net/bilde/20-Effective-ChatGPT-Prompts.pdf
