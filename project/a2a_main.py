
from a2a.agent import Agent
from a2a.message import Message
from a2a.schema import ToolCall, ToolResult
from transformers import pipeline
import asyncio

classifier = pipeline("zero-shot-classification")
TOOLS = {"iphone_sentiment": "iphone", "twitter_sentiment": "twitter"}

class Coordinator(Agent):
    def __init__(self):
        super().__init__("coordinator_agent")

    async def onInit(self):
        print("[Init] Tools registered:", TOOLS)

    async def onMessage(self, msg: Message):
        prompt = msg.content
        result = classifier(prompt, list(TOOLS.values()))
        label = result["labels"][0]
        tool = next(k for k, v in TOOLS.items() if v == label)
        tool_call = ToolCall(tool=tool, name="analyze_prompt", arguments={"text": prompt})
        output: ToolResult = await self.call_tool(tool_call)
        sentiment = output.output.get("sentiment")
        text = output.output.get("text")
        await self.send(msg.respond(f"[✓ {tool}] {sentiment}: {text}"))

if __name__ == "__main__":
    asyncio.run(Coordinator().run())
