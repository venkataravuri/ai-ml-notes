from langchain.llms import AzureOpenAI
import os
from enum import Enum
import streamlit as st

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01"
os.environ["OPENAI_API_BASE"] = "https://genieopenai.openai.azure.com/"


class Models(Enum):
    AZURE_OPEN_AI = 0


MODEL_INFO: dict = {
    0: {
        "deployment_name": "venkat-text-davinci",
        "model_name": "text-davinci-003"
    },
    1: {
        "deployment_name": "ChatGPT",
        "model_name": "gpt-35-turbo"
    },
    2: {
        "deployment_name": "GPT-4-32k",
        "model_name": "gpt-4-32k"
    }
}


class LLMCatalog:
    def __init__(self):
        self.models = {
            Models.AZURE_OPEN_AI: AzureOpenAI(
                deployment_name=MODEL_INFO[st.session_state['model']]['deployment_name'],
                model_name=MODEL_INFO[st.session_state['model']]['model_name'],
                temperature=0,
                streaming=True
            )
        }

    def get_model(self, model: Models) -> AzureOpenAI:
        return self.models[Models.AZURE_OPEN_AI]
