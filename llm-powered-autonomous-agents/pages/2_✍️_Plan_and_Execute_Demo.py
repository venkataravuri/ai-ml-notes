from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from sidebar import sidebar
from tools import tools
from globals import llm_azure_openai_chatgpt
import streamlit as st
from langchain.callbacks.streamlit import StreamlitCallbackHandler


st.header('✍️ Plan & Execute Prompt')
sidebar()

planner = load_chat_planner(llm_azure_openai_chatgpt)

executor = load_agent_executor(llm_azure_openai_chatgpt, tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

def selected_sample_prompt(args):
    st.session_state.prompt = args

st.markdown("#### Try these sample prompts!")
col1, col2, col3 = st.columns(3)

with col1:
    st.button(label="?", use_container_width=True,
            key='col1',
            on_click=selected_sample_prompt, 
            args=('?',))

with col2:
    st.button(label="?", use_container_width=True,
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