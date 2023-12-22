from langchain.llms import AzureOpenAI
import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01"
os.environ["OPENAI_API_BASE"] = "https://genieopenai.openai.azure.com/"

llm_azure_openai_gpt4 = AzureOpenAI(
                deployment_name='GPT-4-32k',
                model_name='gpt-4-32k',
                temperature=0,
                streaming=True
            )

llm_azure_openai_chatgpt = AzureOpenAI(
                deployment_name='ChatGPT',
                model_name='gpt-35-turbo',
                temperature=0,
                streaming=True
            )

llm_azure_openai_davinci = AzureOpenAI(
                deployment_name='venkat-text-davinci',
                model_name='text-davinci-003',
                temperature=0,
                streaming=True
            )