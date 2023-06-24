from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain import LLMMathChain
from utils import get_tools
from utils.llm_catalog import LLMCatalog, Models
from layout.prompt_widget import prompt_section
from layout import display_sidebar
import streamlit as st

st.header('💬 Plan & Execute Prompt')
display_sidebar()

llm_catalog = LLMCatalog()
tools = get_tools(llm_catalog)
model = llm_catalog.get_model(Models.AZURE_OPEN_AI)

planner = load_chat_planner(model)

executor = load_agent_executor(model, tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

prompt_section(agent, "PLAN_EXECUTE")