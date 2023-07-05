import streamlit as st
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from layout import display_sidebar
from utils import LLMCatalog, Models, get_tools
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from layout import prompt_section

st.set_page_config(initial_sidebar_state="auto", layout='wide')
st.header('💬 Autonomous AI Agent :brain:')
display_sidebar()

llm_catalog = LLMCatalog()
tools = get_tools(llm_catalog)
model = llm_catalog.get_model(Models.AZURE_OPEN_AI)
agent = initialize_agent(tools=tools, llm=model, 
                          agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

prompt_section(agent, "REACT")