import streamlit as st
from langchain.agents import initialize_agent, AgentType
from globals import llm_azure_openai_chatgpt
from tools import tools
from sidebar import sidebar
from langchain.callbacks.streamlit import StreamlitCallbackHandler

st.header('⛓️ ReAct Prompt - Reason & Act')
sidebar()

agent = initialize_agent(tools=tools, llm=llm_azure_openai_chatgpt, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

def selected_sample_prompt(args):
    st.session_state.prompt = args

st.markdown("#### Try these sample prompts!")
col1, col2, col3 = st.columns(3)

with col1:
    st.button(label="Book a flight ticket", use_container_width=True,
            key='col1',
            on_click=selected_sample_prompt, args=('Book an economy class flight ticket from Bangalore to Mumbai for next Sunday evening.',))

with col2:
    st.button(label="Elon Musk's net worth", use_container_width=True,
            key='col2',
            on_click=selected_sample_prompt, args=('?',))

with col3:
    st.button(label="?", use_container_width=True,
            key='col3',
            on_click=selected_sample_prompt, args=("?",))


with st.form('AgentForm'):
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = ''

    text = st.text_area(label='Prompt', value=st.session_state['prompt'],
                        height=3, key='prompt')
    submitted = st.form_submit_button(label="Submit", type="primary")

    question_container = st.empty()
    results_container = st.empty()

    res = results_container.container()
    streamlit_callback = StreamlitCallbackHandler(parent_container=res,
        max_thought_containers=4,
        expand_new_thoughts=True,
        collapse_completed_thoughts=True)
        
    if submitted:
        question_container.write(f"**Question:** {text}")
        answer = agent.run(text, callbacks=[streamlit_callback])
        res.write(f"**Answer:** {answer}")