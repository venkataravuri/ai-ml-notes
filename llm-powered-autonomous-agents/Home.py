import streamlit as st
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from sidebar import sidebar
from langchain.agents import initialize_agent, AgentType, AgentExecutor

st.set_page_config(
    initial_sidebar_state="auto",
    layout='wide',
    page_icon='🤖')
st.header('🤖 LLM-powered Autonomous AI Agents :brain:')
sidebar()

st.text("A collection of LLM powered apps.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader(":chains: [ReAct Prompt - Reason & Act](/React)")
    st.markdown("""
[React prompt](https://www.promptingguide.ai/techniques/react) solve complex tasks using verbal reasoning.
ReAct is designed for tasks in which the LLM is allowed to perform actions for a task.

ReAct framework can allow LLMs to interact with external tools to retrieve additional information that leads to more reliable and factual responses.
        
[ReAct](https://learnprompting.org/docs/advanced_applications/react) is able to answer the question by,
- first reasoning about the question (**Thought 1**), 
- and then performing an action (**Act 1**) such as search internet. 
- it then receives an observation (**Obs 1**), 
- and continues with this thought, action, observation loop until it reaches a conclusion.
""")

with col2:
    st.subheader(":writing_hand: Plan & Execute Prompt")
    st.markdown("""
TODO.
""")

with col3:
    st.subheader(":earth_americas: Retrieval Augmented Generation")
    st.markdown("""
TODO.
""")
