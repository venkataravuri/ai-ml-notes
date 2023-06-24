import streamlit as st
import os

def display_sidebar():
    with st.sidebar:
        st.text_input(label='Azure OpenAI API Key', type='password',
                    placeholder='Enter OpenAI API Key', value=os.environ["OPENAI_API_KEY"],
                    key='OPENAI_API_KEY')

        model_options = ["text-davinci-003 (GPT-3)", "gpt-35-turbo (ChatGPT)", "gpt-4-32k (GPT-4)"]
        model_option = st.radio(
            "Set Language Model",
            model_options,
            index=0,
            label_visibility="visible"
        )
        if 'model' not in st.session_state:
            st.session_state['model'] = model_options.index(model_option)
        st.markdown("[View the source code](https://github.com/venkataravuri/llm-app)")