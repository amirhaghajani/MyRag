import asyncio
from urllib import response
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent, AgentStream, ToolCallResult
from llama_index.core.tools import FunctionTool
import wikipedia
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.core.tools.tool_spec.load_and_search import LoadAndSearchToolSpec
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding

# Set a descriptive User-Agent as per Wikimedia policy
wikipedia.set_user_agent("MyLlamaIndexApp/1.0 (a.aghajani@gmail.com)")

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)

Settings.llm = Ollama(
    model="qwen3:14b",
    temperature=0.8,
    request_timeout=90.0
)


wiki_search = WikipediaToolSpec().to_tool_list()[1]
tools = LoadAndSearchToolSpec.from_defaults(wiki_search).to_tool_list()

react_agent = ReActAgent(
    name="wiki_agent",
    description="An agent that searches Wikipedia",
    llm= Settings.llm,
    tools=tools,
    max_iterations=3,
)

agent = AgentWorkflow(agents=[react_agent], root_agent="wiki_agent")
async def main():
    handler = agent.run("What is the most famous place on every island in the Azores?")
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            print(f"\n--- Tool Call: {ev.tool_name} ---")
            print(f"Input: {ev.tool_kwargs}")
            print(f"Output: {ev.tool_output}\n")
        if isinstance(ev, AgentStream):
            print(ev.delta, end="", flush=True)
    response = await handler
    print("\n\n=== Final Response ===")
    print(response)
asyncio.run(main())
