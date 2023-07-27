import streamlit as st
from langchain.agents import initialize_agent, AgentType
from globals import llm_azure_openai_chatgpt
from tools import tools
from sidebar import sidebar
from langchain.callbacks.streamlit import StreamlitCallbackHandler
import langchain

langchain.debug = True
langchain.verbose = True

st.header('⛓️ ReAct Prompt - Reason & Act')
sidebar()

agent = initialize_agent(tools=tools, llm=llm_azure_openai_chatgpt, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
                         handle_parsing_errors=True)

def selected_sample_prompt(args):
    st.session_state.prompt = args

st.markdown("#### Try these sample prompts!")
col1, col2, col3 = st.columns(3)

with col1:
    st.button(label="Book a flight ticket sample", use_container_width=True,
            key='col1',
            on_click=selected_sample_prompt, args=('Book an economy class flight ticket from Bangalore to Mumbai for next Sunday evening.',))

with col2:
    st.button(label="Elon Musk's net worth", use_container_width=True,
            key='col2',
            on_click=selected_sample_prompt, args=('Who is the richest person in the world now? What is net worth of the richest person?',))

with col3:
    st.button(label="AWS resources rampup sample.", use_container_width=True,
            key='col3',
            on_click=selected_sample_prompt, args=("I have a website hosed on AWS in US East Ohio region which uses 10 EC2 instances of type 'm5.2xlarge'. The website supports 100 requests per second. I have plans to run a marketing campaign next Monday, which will drive additional traffic to my website which will be 200 requests per second. To support additional web traffic, I need to add additional EC2 instances. I'm planning to use EC2 On-Demand instances. Can you suggest me how many EC2 On-Demand instances needed? What would be my additional cost?",))


with st.form('AgentForm'):
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = ''

    text = st.text_area(label='Prompt', 
                        height=3, key='prompt')
    submitted = st.form_submit_button(label="Submit", type="primary")

    question_container = st.empty()
    results_container = st.empty()

    res = results_container.container()
    streamlit_callback = StreamlitCallbackHandler(parent_container=res,
        max_thought_containers=4,
        expand_new_thoughts=True,
        collapse_completed_thoughts=False)
        
    if submitted:
        question_container.write(f"**Question:** {text}")
        answer = agent.run(text, callbacks=[streamlit_callback])
        res.write(f"**Answer:** {answer}")