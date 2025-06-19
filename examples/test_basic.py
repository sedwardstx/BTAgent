import asyncio
from agents import Agent, Runner
from agents import function_tool

@function_tool
async def say_hello(name: str = "there") -> str:
    """Say hello to someone."""
    return f"Hello {name}! Nice to meet you."

class BasicTestAgent(Agent):
    """A basic test agent."""
    
    def __init__(self):
        super().__init__(
            name="BasicTestAgent",
            instructions="You are a helpful assistant. Be friendly and concise.",
            tools=[say_hello]
        )

async def test_basic():
    agent = BasicTestAgent()
    
    print("Testing basic agent...")
    result = await Runner.run(agent, "Hi there!")
    print(f"User: Hi there!")
    print(f"Agent: {result.final_output}")
    print("✅ Basic agent test completed")

if __name__ == "__main__":
    asyncio.run(test_basic()) 