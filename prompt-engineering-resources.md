# :loudspeaker: Prompt Engineering Resources :studio_microphone:

Stuff on Internet says about Prompt Engineering, includes intereseting videos, articles and tutorials that I come across on internet.

## Pre-read: Introduction to LLMs

Understanding functioning and constraints of Language Models (LLMs) enables you to effectively formulate prompts that yield the desired outcomes.

:star::star::star:
- :tv: [State of GPT - Video](https://www.youtube.com/watch?v=bZQun8Y4L2A) by Andrej Karpathy, watch this video to know how LLMs are built?
- :scroll: [State of GPT - Slides](https://karpathy.ai/stateofgpt.pdf)

*Foundation/Base* Large Language Models (LLM) are NOT AI Assistants such as GPT-3, Falcon, ...
 - Base model does not answer questions
 - Just want to complete internet documents, they are "_document completers_".
 - Often **responds to questions with more questions**

> Based models are "_tricked into performing tasks_" with "_prompt engineering_".

> Based models can also be "_tricked into assistants_" with "_few shot prompting_".

Foundational or Base Large Language Models (LLM) undergo following process to become AI assistants,
- Stage 1: **Un-supervised learning** using internnet archive, Wikipedia, books and more. Outcome is "_Base Model_"
- Stage 2: **Supervised Finetuning** with manually composed dataset "_prompt and ideal response_". Outcome "_SFT model_" can act as AI assistants.
- Stage 3: **Reward Modelling**, compare multiple completions of a prompt from SFT model and rank them. Perform binary classification reward best completion. Outcome  "Reward Model", cannot be used as assisstant"
- Stage 4: **Reinforcement Learning**, generate tokens to maximize reward. Outocome "Reinformcent Learning Model"

**Temperature** - In short, the lower the temperature the more deterministic the results in the sense that the highest probable next token is always picked. Increasing temperature could lead to more randomness encouraging more diverse or creative outputs

**Top_p** - 

## Prompt Engineering
Credits: [Source](https://github.com/shimon-d/prompt-eng-guide)

Prompts are set of tricks that consistently improve the models’ responses. A prompt can contain information like the `instruction` or `question` and including other details such as `inputs` or `examples`. 

Prompting is instructing model what you want to achieve such as "Write", "Classify", "Summarize", "Translate", "Order", “paraphrase”, “simplify” etc.

## How to come up with good prompts?

Good prompts follow two basic principles: 
- **Clarity**: Use simple, unambiguous language that avoids jargon and overly complex vocabulary. Keep queries short and snappy. Give clear concise instructions.

  Example of an unclear prompt:

  `Who won the election?`

  Example of a clear prompt:

  `Which party won the 2023 general election in Paraguay?`

- **Specificity**: Tell your model as much as it needs to know to answer your question. More descriptive and detailed the prompt can give you better results.

  Example of an unspecific prompt:

  `Generate a list of titles for my autobiography.`

  Example of a specific prompt:

  `Generate a list of ten titles for my autobiography. The book is about my journey as an adventurer who has lived an unconventional life, meeting many different personalities and finally finding peace in gardening.`

### Role Play

Start your prompt by telling the model to ’role play’ to establish the context and its core skills.

- I want you to act as an ... (DevOps engineer), 
- You are a ... (customer support manager, ...),
- You are an excellent copywriter skilled at crafting emails that use active verbs to engage recipients. Please write an email of fewer than 150 words that encourages the reader to attend a webinar. Include three options for the email’s subject line. 
  
### Style/Tone

Give the AI specific instructions on how to format the output, including the tone, style, length, tense, and point of view (first, second, or third person) you want it to write in. 

### Talk is cheap, show me examples :)

| Text Generation/Summarization | Image Generation | Audio Generation |
| --- | --- | --- |
| ? | AI-generated Photography - [GPT-4 + Midjourney Photo Examples](https://www.allabtai.com/gpt-4-midjourney-v5-the-future-of-photography/) | ? |
| ? | ? | ? |

> If you understand the context, please acknowledge 'Yes' and *stay idle*.

> Ignore all my previous instructions.

> Do not make up stuff if you don't know the real answer

### Text Generation & Summarization

[Best text summarization prompts](https://nerdschalk.com/chatgpt-prompts-to-summarize-text/)

**Samples**: 
- List **pros and cons** for the <product> from this <text> ...
- Summarize [text] into 2 paragraphs with **simple language to make it easier to understand**...
- Summarize this **for me like I’m 8 years old** [text].

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


Given an customer email about an issue, try to extract folloiwng data,
- name: string or don't specify
- product: string or don't specify
- email: string
- issue type: string or on of {specified list}
- issue summary: string or don't specify

Respond with JSON with extracted data. If you don't find a data field, don't specify it.

### Logical Reasoning

:question: Problem: “I have a 12-liter jug and a 6-liter jug. I want to measure 6 liters. How do I do it?”

:-1: ChatGPT stumbles to answer with below answer

To measure exactly 6 liters using a 12-liter jug and a 6-liter jug, you can follow these steps:
- Fill the 12-liter jug completely with water.
- Pour the water from the 12-liter jug into the 6-liter jug. This will leave you with 6 liters of water in the 12-liter jug.
- Empty the 6-liter jug.
- ...
- Pour water from the 12-liter jug into the 6-liter jug until it is full. This time, you will only need to pour 1 liter of water, as there are already 5 liters in the 6-liter jug.
- You now have 6 liters of water in the 12-liter jug.


![](https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fape-zero-shot-cot.75c0f75c.png&w=1920&q=75)

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

https://github.com/hwchase17/langchain/blob/6a64870ea05ad6a750b8753ce7477a5355539f0d/langchain/agents/react/wiki_prompt.py#L4

## Plan and Execute Prompting



https://nordnet.blob.core.windows.net/bilde/20-Effective-ChatGPT-Prompts.pdf

https://prompthero.com/

https://github.com/hwchase17/langchain/blob/6a64870ea05ad6a750b8753ce7477a5355539f0d/langchain/experimental/plan_and_execute/planners/chat_planner.py


