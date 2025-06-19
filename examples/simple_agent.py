import asyncio
from agents import Agent, Runner
from agents import function_tool

@function_tool
async def handle_small_talk(message: str) -> str:
    """Handle casual conversation and small talk."""
    return "I'm doing well! How about you?"

class SimpleAgent(Agent):
    """A simple agent without behavior trees for testing."""
    
    def __init__(self):
        super().__init__(
            name="SimpleAgent",
            instructions="""You are a friendly conversational agent. 
            Always be polite and helpful in your responses.""",
            tools=[handle_small_talk]
        )

async def main():
    # Create and run the agent
    agent = SimpleAgent()
    
    # Example conversation
    messages = [
        "Hi there!",
        "How are you doing today?",
        "What's the weather like?",
        "Can you tell me about yourself?",
        "Goodbye!"
    ]
    
    for message in messages:
        print(f"\nUser: {message}")
        result = await Runner.run(agent, message)
        print(f"Agent: {result.final_output}")
        print("Status: Message processed successfully")

if __name__ == "__main__":
    asyncio.run(main()) 