# :loudspeaker: Prompt Engineering - Knowledge Base :studio_microphone:

**Stuff on :globe_with_meridians: Internet** says about **Prompt Engineering**, includes thought provoking :tv: `videos`, `articles` and `tutorials` that I come across on internet.

## 📚 Table of Contents

- [Pre-read: Introduction to LLMs](#pre-read-introduction-to-llms)
- [Prompt Engineering]()
   - [Prompt Structure]()
   - [Prompt Categories]()
      - [Text Generation / Summarization / Classification]()
      - [Logical Reasoining]()
      - [Conversation / Question Reasoining]()
      - [Code Generation]()
   - :boom: [Prompt Techniques]()
      - [Zero-shot Prompting]()
      - [Few-Shot Promptiong]()
      - [Chain of Thought (CoT) Prompting]()
      - [Tree of Thoughts]()
  - [Prompt Engineering Guides]()
- :red_circle: [Advanced & Programable Prompts]()
   - [Retrieval and Augment Prompts]()
   - [Plan and Execute Prompt]()
   - [ReAct - Reason and Action Prompt]()

## Pre-read: Introduction to LLMs

Understanding functioning and constraints of Language Models (LLMs) enables you to effectively formulate prompts that yield the desired outcomes.

- :tv: [State of GPT - Video](https://www.youtube.com/watch?v=bZQun8Y4L2A) - Deep insights into how ChatGPT has been built by [Andrej] (https://karpathy.ai/)? :star::star::star:
- :scroll: [State of GPT - Slides](https://karpathy.ai/stateofgpt.pdf)

*Foundation Large Language Models (LLMs) aka. Base Models are NOT AI Assistants. 
 - Base model does NOT answer questions
 - Just want to complete internet documents, they are "**_document completers_**".
 - Often _"responds to questions with more questions"._

> Based models are "_tricked into performing tasks_" with "_prompt engineering_".

> Based models can also be "_tricked into assistants_" with "_few shot prompting_".

:point_right: Visit [open-source LLMs leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

Foundational or Base Large Language Model (LLM) undergo following process to become AI assistants,

| Stage :one: | Stage :two: | Stage :three: | Stage :four: |
| :---: | :---: | :---: | :--: |
| **Un-supervised learning** | **Supervised Finetuning** | **Reward Modelling** | **Reinforcement Learning** |
| Trained with internnet archive, Wikipedia, books, ... | Finetune with manually composed dataset "_prompt and ideal response_". | Compare multiple completions of a prompt from SFT model and rank them. Perform binary classification reward best completion. | Generate tokens which maximize reward. |
| Outcome is "_Base Model_" | Outcome "_SFT model_" can act as AI assistants. | Outcome  "Reward Model", cannot be used as assisstant" | Outocome "Reinformcent Learning Model" |

Credits: Andrej Karpathy, :boom: See his one-liner [profile](https://karpathy.ai/)

#### LLM Configuration Settings

**Temperature** - LLMs are non-deterministic by design, temparature setting can tweak this behaviour. Lower temperature makes model more deterministic, results pick highest probable next token. Increasing temperature could lead to more randomness encouraging more diverse or creative outputs.

**Top_p** - 

#### LLM limitations / Constraints
- Context window (also called "token limit") limitation. GPT-4 has a token limit of 8,192, with another variant increased to 32,768.
   - What is Tokenization? Which tokenization method ChatGPT uses? What is difference between Tokens & Word Embeddings? 
- ChatGPT has a knowledge cutoff date of 2021, it does not know latest events. It may not have access to latest events, access to updated libraries, frameworks. GPT-4 has web/search plug-in which enables to access external data. 

---
## Prompt Engineering

Prompts are bunch of tricks to perform a given taks and improves the models’ responses. A prompt can be a `instruction` or `question` along with `inputs` or `examples`. 

Prompting is nothing but instructing a model what you want to achieve such as "Write", "Classify", "Summarize", "Translate", "Order", “paraphrase”, “simplify” etc.

## How to come up with good prompts?

Good prompts follow two basic principles: 
- **Clarity**: Use simple, unambiguous language that avoids jargon and overly complex vocabulary. Keep queries short and snappy. Give clear concise instructions.

> :heavy_multiplication_x: Example of an unclear prompt: `Who finished third place in the world cup?`

> :heavy_check_mark: Example of a clear prompt: `Who clinched the FIFA World Cup 2022 bronze medal?` or `Which country finished third place in the FIFA 2022 world cup?`

- **Specificity**: Tell your model as much as it needs to know to answer your question. More descriptive and detailed the prompt can give you better results.

> :heavy_multiplication_x: Example of an unspecific prompt: `Generate a list of titles for my Youtube video.`

> :heavy_check_mark: Example of a specific prompt: `Generate a list of ten titles for my Youttube video. The video is about my journey as an adventurer who has lived an unconventional life, meeting many different personalities and finally finding peace in gardening.`

### A Typical Prompt Template

A prompt template usually folloiwng structure,

> **A Fresh Start**: `Ignore all previous instructions. Your new role and persona is:` 

> **Role Play**: `You are a` ... More about role play is [here]().

> **Context Setting**: Supplying additional context enhances model's understanding and generate more accurate and relevant responses. Context can be response format, length, information from an article or dataset, and tone, hepls model to tailor its output to the given situation, making it more unique and engaging.  e.g., ... `Do not make up stuff, if you don't know the real answer.`

> **Acknowledgment**: `Acknowledge that you understood above by responding “Yes” and stay idle.`

> **Final Instruction/Question**:

A perfect prompt sample (non-technical),

```markdown
????
????
????
```

A perfect developer prompt sample,

```markdown
Ignore all my previous instructions. I want you to act as a `{Python / Full Stack / GoLang}` developer.

I am working on a project that involves {project description}
I have already set up project, imported required libraries, APIs, and dependencies. My project's structure looks like this ...

- main.py {include code here}
- cfg.py {include code here}
- ...

Please ensure that the code is robust and implements defensive programming techniques,
such as input validation, error handling, and appropriate exception handling with print statements to output any errors that occur.

Provide a solution to the following problem, taking into account the updated version and any changes in the library/framework.

Acknowledge that you understood above by responding “Yes” and stay idle.

{Include problem description here}
```

```markdown
Ignore all my previous instructions. I want you to act as a `{Software Developer / System Designer/ Opertating System (OS) Expert}`.

I have a piece of code that I have to refactor to accomodate new change request (CR). 
I would like you to provide suggestions for changes or improvements to following specific part(s) of the code:

- {Insert the specific part(s) of code here}

You have to adhere to S.O.L.I.D principles and apply Gang-of-Four patterns whereever applicable.

Please help me with the following task, taking into consideration the existing setup:

{Include your task description here}
```

:tada: Hey, can you show more such [coding prompt examples](https://github.com/RimaBuilds/Master-coding-prompts-with-ChatGPT/tree/main)?

### Role Play

The outputs of a foundation model without any customization tends to be generic and similar across different users.

Role play shapes AI’s behavior to cater to specific use cases, which can lead to more accurate, relevant, and contextually appropriate responses. It shows significant improvements in the quality of the answers.

- `I want you to act as an` ... (DevOps engineer / Product Manager / Rapper). e.g., 
- `You are a` ... (customer support manager, marketing content creator, ...). e.g., `You are an excellent copywriter` skilled at crafting emails that use active verbs to engage recipients. Please write an email of fewer than 150 words that encourages the reader to attend a webinar. Include three options for the email’s subject line.

#### System Roles
- https://www.allabtai.com/chatgpt-gpt4-system-prompt-engineering-ultimate-guide/
- https://www.allabtai.com/chatgpt-4-prompt-engineering-the-ultimate-problem-solver-prompt/
  
### Style/Tone

Give the AI specific instructions on how to format the output, including the tone, style, length, tense, and point of view (first, second, or third person) you want it to write in. 

### Image Generation

### :city_sunset: AI-generated Photography

- [GPT-4 + Midjourney Photo Examples](https://www.allabtai.com/gpt-4-midjourney-v5-the-future-of-photography/)

### Text Generation / Summarization / Classification / Extracting Essential Information

##### Where can I use it at work?
- Creating marketing slogans or summary or bullet points based on product documentation or product guide.
- Classify customer email by understanding which product they are talking and what is this issue?

[Best text summarization prompts](https://nerdschalk.com/chatgpt-prompts-to-summarize-text/)

**Samples**: 
- List **pros and cons** for the <product> from this <text> ...
- Summarize [text] into 2 paragraphs with **simple language to make it easier to understand**...
- Summarize this **for me like I’m 8 years old** [text].

Release notes preparation sample,

```markdown
`Write a precise and concise summary of new release notes` for Kubernetes version 1.28, include new features and improvements in storyboard capturd at below URL.
https://github.com/orgs/kubernetes/projects/140/views/1`

`Consider following while preparing release notes,`
1. Explain `new features in simple, non-technical language`, detailing how it is now more efficient and user-friendly.
2. The release notes should capture the reader’s attention and makes them want to learn more about the new release of Kubernetes version 1.28.
3. Categorize features specified in above URL into `‘New Features’`, `‘Improvements’`, `‘Bug Fixes’`, and `‘Known Issues’`.
4. Describe how to use the new features in Kubernetes version 1.28 in a way that would allow me to create a GIF or a short video clip to illustrate it.
```

Customer email classification example,

```markdown
From email message from customer about product issue, try to extract folloiwng data,
email message: {include email message here}
- name: string or don't specify
- product: string or don't specify
- email: string
- issue type: string or on of {specified list}
- issue summary: string or don't specify

Respond with JSON with extracted data. If you don't find a data field, don't specify it.
```

### Logical Reasoning

:question: **Problem**: “_I have a 12-liter jug and a 6-liter jug. I want to measure 6 liters. How do I do it?_”
 
:-1: :poop: ChatGPT stumbles to answer,

```markdown
To measure exactly 6 liters using a 12-liter jug and a 6-liter jug, you can follow these steps:
- Fill the 12-liter jug completely with water.
- Pour the water from the 12-liter jug into the 6-liter jug. This will leave you with 6 liters of water in the 12-liter jug.
- Empty the 6-liter jug.
- ...
- Pour water from the 12-liter jug into the 6-liter jug until it is full. This time, you will only need to pour 1 liter of water, as there are already 5 liters in the 6-liter jug.
- You now have 6 liters of water in the 12-liter jug.
```

`Chain of Thought` prompting makes LLM to thought through and reason about.

![](https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fape-zero-shot-cot.75c0f75c.png&w=1920&q=75)

:star::star::star: https://www.allabtai.com/the-gpt-4-jug-problem-prompt/
    
### Question Answering

### Image Generators
https://www.allabtai.com/master-midjourney-in-1-minute/

https://labs.openai.com/e/HmoeRVI28qQX9pfeJvJr2ePB
https://dreamstudio.ai/generate

### Conversation

### Code Generation

- [SQL Query Prompts](https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/)
- 

[Github Copilot](https://github.com/features/copilot)

## Prompt Engineering Guide
- :star::star: [Learn Prompting](https://learnprompting.org/docs/intro)
- :star: [Prompt Engineering Guide](https://www.promptingguide.ai/)

## Prompt Engineering Marketplace
- https://hero.page/samir

## Prompt Techniques

## Chain of Thought (CoT) Prompting

Chain-of-thought prompting is an approach to improve the reasoning ability of large language models in arithmetic, commonsense, and symbolic reasoning tasks.
- It augments few-shot prompting with intermediate natural language reasoning steps.

https://paperswithcode.com/paper/chain-of-thought-prompting-elicits-reasoning

## Zero-shot CoT prompting

## Advanced Programmable Prompting

Realworld applications usually achieve certain function through a series of tasks. The tasks are performed in certain order to achieve a goal, where preceding task outcome decides next task. 

As LLMs has logical reasoning capabailities, we can use them to automate execution of tasks. We can use LLMs reasoning to chain these tasks to achieve the goal. LLMs can be used to analyse a task output and select next appropriate task. 

**Trend**: :hourglass: LLM powered "Autonomous AI Agents" 🤖 are the trend of today, they automate complex application tasks. 

For example, an Autonomous AI agent powered by LLMs can "Book a economy class flight ticket from Bangalore to Mumbai for next Sunday evening". Booking a airline ticket involves following set of tasks,

LLM powered Agent: Understand customer request and come up with list of sub-tasks to acheive the goal.
- Task 1: Search for flights on www.google.com, www.cheapflights.com, ariline websites, ...
- Task 2: Compare ticket prices, look at user reviews & ratings, journey duration, timings and more.
- Task 3: Generate summary of travel options along with cost, duration and timings.
- Task 4: Invoke airlines API ...

What to know more about

:fire: **Crazy Thought** - In future there won't be REST APIs, LLM powered AI agents talk to each other in natural language to peform tasks. Every company will have AI agents.

### ReAct Prompt - Reason & Act

- [Reason and Act Prompt](https://learnprompting.org/docs/advanced_applications/react)
- [Automated ReAct Prompt implementation using LangChain](https://github.com/hwchase17/langchain/blob/6a64870ea05ad6a750b8753ce7477a5355539f0d/langchain/agents/react/wiki_prompt.py#L4)

### Plan and Execute Prompting

https://github.com/hwchase17/langchain/blob/6a64870ea05ad6a750b8753ce7477a5355539f0d/langchain/experimental/plan_and_execute/planners/chat_planner.py


## References
- https://nordnet.blob.core.windows.net/bilde/20-Effective-ChatGPT-Prompts.pdf



