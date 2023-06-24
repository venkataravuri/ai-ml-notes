from langchain.tools import Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents.tools import Tool
from langchain import LLMMathChain
from utils.llm_catalog import LLMCatalog, Models

search = DuckDuckGoSearchRun()

def get_tools(llm_catalog):
    llm = llm_catalog.get_model(Models.AZURE_OPEN_AI)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        )
    ]
    return tools
