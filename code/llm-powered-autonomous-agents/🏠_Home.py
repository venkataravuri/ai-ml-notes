import streamlit as st
from sidebar import sidebar

st.set_page_config(
    initial_sidebar_state="expanded",
    layout='wide',
    page_icon='🤖')
st.header('🤖 LLM-powered Apps (Autonomous AI Agents) :brain:')
sidebar()

st.markdown("---")
st.markdown("""_A collection of LLM powered apps built using LangChain, Streamlit and Chorma vector db._
            \n**👈 Select a demo from the sidebar** to see some examples""")

st.markdown("---")
st.markdown("""#### :chains: React Demo - App powered by ReAct Prompt""")
st.markdown("""
[React prompt](https://www.promptingguide.ai/techniques/react) solves complex tasks using verbal reasoning.
    
[ReAct](https://learnprompting.org/docs/advanced_applications/react) is able to answer the question by,
1. First reasoning about the question (**Thought 1**), 
2. and then performing an action (**Act 1**) such as search internet. 
3. It then receives an observation (**Obs 1**).
4. Cycle continues again with **thought, action, observation** in a loop until it reaches a conclusion.
""")

st.markdown("---")
st.subheader(":writing_hand: Plan & Execute Demo")
st.markdown("""
:anguished: Yet to code.
""")

st.markdown("---")
st.subheader(":earth_americas: Retrieval Augmented Generation Demo")
st.markdown("""
:flushed: Yet to code.
""")
