import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

async def main():
    llm = ChatOllama(model="gemma4:e4b", base_url="http://localhost:11434", temperature=0.1)
    async for chunk in llm.astream([HumanMessage(content="Explain 1+1 in 1 sentence. Think before you answer.")]):
        print(repr(chunk))

asyncio.run(main())
