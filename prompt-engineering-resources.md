# Prompt Engineering

## LLM Insights
Credits: [State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A) by Andrej Karpathy 

 Base Large Language Models are NOT AI Assistants
 - Base model does not answer questions
 - Just want to complete internet documents, they are "_document completers_".
 - Often **responds to questions with more questions**

> Based models are "_tricked into performing tasks_" with "_prompt engineering_".

> Based models can be "_tricked into assistants_" with "_few shot prompting_".

Large Language Models (LLM) undergo following "_training pipeline_" to become AI assistenats,
- Stage 1: **Un-supervised learning** using internnet archive, Wikipedia, books and more. Outcome is "_Base Model_"
- Stage 2: **Supervised Finetuning** with manually composed dataset "_prompt and ideal response_". Outcome "_SFT model_" can act as AI assistants.
- Stage 3: **Reward Modelling**, compare multiple completions of a prompt from SFT model and rank them. Perform binary classification reward best completion. Outcome  "Reward Model", cannot be used as assisstant"
- Stage 4: **Reinforcement Learning**, generate tokens to maximize reward. Outocome "Reinformcent Learning Model"
