import streamlit as st
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.agents import AgentExecutor


def handle_sample_selection(args):
    st.session_state.prompt = args

def prompt_section(agent: AgentExecutor, prompt_type):
    display_samples(prompt_type)

    with st.form('AutonomousAgentForm1'):

        if 'prompt' not in st.session_state:
            st.session_state['prompt'] = ''

        text = st.text_area(label='Prompt', value=st.session_state['prompt'],
                            height=4, key='prompt')
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


def display_samples(prompt_type: str = "REACT"):
    st.subheader("Try these sample prompts!")
    col1, col2, col3 = st.columns(3)

    if prompt_type == "REACT":
        with col1:
            st.button(label="Book economy class :blue[flight] ticket from :green[Bangalore] to :red[Mumbai] for next Sunday evening.",
                    type="secondary", use_container_width=True,
                    key='col1',
                    on_click=handle_sample_selection, args=('Book economy class flight ticket from Bangalore to Mumbai for next Sunday evening.',))

        with col2:
            st.button(label="Upgrade :blue[Kubernetes] (k8s) cluster to version 1.28.",
                    type="secondary", use_container_width=True,
                    key='col2',
                    on_click=handle_sample_selection, args=('Upgrade Kubernetes cluster to latest verion.',))

        with col3:
            st.button(label="What would be projected net worth of Adani by the end of Modi's prime minister term?",
                    type="secondary", use_container_width=True,
                    key='col3',
                    on_click=handle_sample_selection, args=("What would be projected net worth of Adani by the end of Modi's prime minister term?",))
    elif prompt_type == "PLAN_EXECUTE":
        with col1:
            label = "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
            st.button(label=label,
                    type="secondary", use_container_width=True,
                    key='col1',
                    on_click=handle_sample_selection, args=(label,))
