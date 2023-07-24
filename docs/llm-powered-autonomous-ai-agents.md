# LLM-powered Autonomous AI Agents

> In the future, Large Language Models will be an integral part of virtually every software product, enhancing user experiences and turn them into more autonomous.

🚧 Doc is under construction 🚧

Large Language Models have potential to use beyond text generation, summarization, and code generation. LLMs can act as reasoning engines. They can be,
- Turned into a powerful general problem solver.
- Extend to automate complex workflows.
- Could power autonomous systems.

Autonomous LLM "agents" or "copilots" are new generation of AI assistants which can perform complex tasks when commanded to by a human, without needing close supervision. They can make logical decisions, and handle a number of tasks without consistent human intervention.

## What are Autonomous LLM Agents?

“Autonomous” LLM agents are goal-driven self-executing software that can generate, execute and priortize tasks to achieve a certain goal. They translate natural language prompts into actions and execute them.

Notable examples of browser based autonomous LLM agents are,
- AgentGPT
- BabyAGI
- Godmode.space
- AutoGPT (non browser based)

Autonomous agents achieve specified goals by breaking them into tasks, execute them independently without human intervention.

Agents use a LLM to determine which actions to take and in what order. The agent creates a chain-of-thought sequence on the fly by decomposing the user request.

## How do Autonomous LLM Agents Work?

Agents make use of LLMs & tools to perform actions in autonomous fashion. LLMs act as agent’s brain, where in tools enable an agent to take certain actions. Agents follow a chain-of-thought reasoning approach to decompose a problem into sequence of steps. 

<img src="assets/ai-agents-overiew.png" width="60%" height="60%" alt="AI Agents Overview"/>

Agents are autonomous with regards to,
1. Plan steps to solve a problem or achieve a goal using chain-of-thought reasoning.
2. For each step in steps:
    1. Decide which **tool** to accomplish the step
    2. Use tool to peform best course of **action** and record **observation**.
    3. Observation is then passed back into agent, and it decides what step to take next.
3. Agent repeats "Action" -> "Observation" -> "Thought" cycle, until it reaches satisfactory answer.

<img src="assets/agent-architecture.png" width="60%" height="60%" alt="AI Agents Components" />

## How to build Autonomous LLM Agents?

Critical functional components autonomous agent application are,

#### Task Planning

Agent decomposes complex tasks or goals into smaller steps. Task decomposing is done through LLM prompting. Different prompting techniques can be employeed such as,
1. ReAct - Reason and Act. See LangChain prompt template [here](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/react/wiki_prompt.py).
2. Modular Reasoning, Knowledge and Language (MRKL) prompt. See LangChain prompt template [here](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/mrkl/prompt.py)
3. ?

#### Tools

Tools are interfaces that an agent can use to interact with the world. These tools can be generic utilities (e.g. search), other chains, or even other agents.

Tools can be,
- Shell Tool
- Search Tools
- Requests
- Others ...

#### Memory

Chains and Agents are stateless, meaning that they treat each incoming query independently. Memory involves keeping a concept of state around throughout a user's interactions with an language model.

### Frameworks & Libraries

Popular frameworks to build LLM powered applications,

| <img src="https://python.langchain.com/img/parrot-chainlink-icon.png" width="12%" height="12%"/> LangChain | <img src="https://cdn-images-1.medium.com/v2/resize:fit:1200/1*_mrG8FG_LiD23x0-mEtUkw.jpeg" width="8%" height="8%"/> LlamaIndex | <img src="https://avatars.githubusercontent.com/u/128686189?s=48&v=4" width="8%" height="8%"/> Chainlit |
| --- | --- | --- |
|[LangChain](https://www.langchain.com/) is an open-source Python library that enables anyone who can write code to build LLM-powered applications. <br/>The package provides a generic interface to many foundation models, enables prompt management, and acts as a central interface to other components like prompt templates, other LLMs, external data, and other tools via agents.| [LlamaIndex](https://www.llamaindex.ai/) is an open-source project that provides a simple interface between LLMs and external data sources like APIs, PDFs, SQL etc.<br/>It provides indices over structured and unstructured data, helping to abstract away the differences across data sources. It can store context required for prompt engineering, deal with limitations when the context window is too big, and help make a trade-off between cost and performance during queries. | [Chainlit](https://docs.chainlit.io/overview) library is similar to Python’s Streamlit Library.<br/>This library is seamlessly integrated with LangFlow and LangChain(the library to build applications with Large Language Models), which we will do later in this guide.<br/>Chainlit even allows for visualizing multi-step reasoning.

LangChain framework components,

<img src="assets/langchain-components.png" width="60%" height="60%" alt="LangChain Components"/>

### Vector Datatabases

Vector databasse have advanced indexing and search algorithms that make them particularly efficient for similiarity searches. Vector databases can measure the distance between two vectors, which defines their relationship. Small distances suggest high relatedness, while larger distances suggest low relatedness.

#### What is a vector?
A vector is an array of numbers like [0, 1, 2, 3, 4, … ]. Vector can represent more complex objects such as words, sentences, images, and audio files in an embedding.

#### What is embedding? 
In the context of large language models, embeddings represent text as a dense vector of numbers to capture the meaning of words. They map the semantic meaning of words together or similar features into vectors. These embeddings can then be used for search engines, recommendation systems, and generative AIs such as ChatGPT. 

Vector databases enhance the memory of LLMs thorugh context injection. Prompt augmentation feeds LLMs with contextual data.

<img src="assets/vector-db-llms.png" width="50%" height="50%" alt="Vector DBs"/>

Most popular vector databases are,

| Pinecone | Chroma | Weaviate |
| --- | --- | --- |
| Pinecone is a cloud-based vector database designed to efficiently store, index and search extensive collections of high-dimensional vectors. | Chroma is an open source vector database that provides a fast and scalable way to store and retrieve embeddings. <br/>Chroma is designed to be lightweight and easy to use, with a simple API and support for multiple backends, including RocksDB and Faiss (Facebook AI Similarity Search) | Weaviate is an open source vector database designed to build and deploy AI-powered applications. Weaviate’s key features include support for semantic search and knowledge graphs and the ability to automatically extract entities and relationships from text data.|

## Example: Autonomous Travel Agent

An autonomous travel agent powered by LLMs such as ChatGPT to automate airline ticket booking process.

#### Demonstration

https://github.com/venkataravuri/ai-ml/tree/master/llm-powered-autonomous-agents

#### Code Walkthrough

https://github.com/venkataravuri/ai-ml/blob/master/llm-powered-autonomous-agents/pages/1_%E2%9B%93%EF%B8%8F_React.py

## Example: Tanzu Kubernetes Autonomous AI Agent

A sample autonoums agent that peforms Tanzu Kubernetes automation activities.

### Demo & Code Walkthrough

https://github.com/venkataravuri/ai-ml/blob/master/llm-powered-autonomous-agents/pages/2_%E2%9C%8D%EF%B8%8F_Plan_and_Execute.py

### Credits & References
- https://www.eweek.com/artificial-intelligence/autonomous-ai-agents/
