import streamlit as st
import os

def sidebar():
    with st.sidebar:
        st.text_input(label='Azure OpenAI API Key', type='password',
                    placeholder='Enter OpenAI API Key', value=os.environ["OPENAI_API_KEY"],
                    key='OPENAI_API_KEY')

        st.markdown("[View the source code](https://github.com/venkataravuri/ai-ml/)")