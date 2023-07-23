from langchain.tools import Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents.tools import Tool
from langchain import LLMMathChain
from globals import llm_azure_openai_davinci

search = DuckDuckGoSearchRun()


llm_math_chain = LLMMathChain.from_llm(llm=llm_azure_openai_davinci, verbose=True)

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
