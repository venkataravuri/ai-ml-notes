import streamlit as st
from layout import display_sidebar
from utils.llm_catalog import LLMCatalog, Models
from utils import get_tools
from langchain.agents import initialize_agent, AgentType
from layout.prompt_widget import prompt_section

st.header('💬 Reason & Action Prompt')
display_sidebar()


llm_catalog = LLMCatalog()
tools = get_tools(llm_catalog)
model = llm_catalog.get_model(Models.AZURE_OPEN_AI)
agent = initialize_agent(tools=tools, llm=model, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

prompt_section(agent)