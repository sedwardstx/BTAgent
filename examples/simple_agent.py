import asyncio
from typing import Dict, Any
from bt_agent.core import BTAgent, BTAgentAction
from btengine.base import NodeStatus
from btengine.nodes import SequenceNode, SelectorNode
from agents.tools.base import BaseTool
from agents.function_tool import function_tool

class GreetAction(BTAgentAction):
    def tick(self) -> NodeStatus:
        """Handle greeting the user."""
        if not self.current_step:
            return NodeStatus.FAILURE
        
        # Check if this is the start of the conversation
        if not self.current_step.steps:
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

class UnderstandIntentAction(BTAgentAction):
    def tick(self) -> NodeStatus:
        """Understand the user's intent from their message."""
        if not self.current_step:
            return NodeStatus.FAILURE
        
        # Analyze the current message
        message = self.current_step.input_text.lower()
        
        # Check for farewell indicators
        if any(word in message for word in ["bye", "goodbye", "see you", "farewell"]):
            return NodeStatus.FAILURE  # Skip to farewell
        
        return NodeStatus.SUCCESS

@function_tool
async def handle_small_talk(message: str) -> str:
    """Handle casual conversation and small talk."""
    return "I'm doing well! How about you?"

class SmallTalkAction(BTAgentAction):
    def tick(self) -> NodeStatus:
        """Handle casual conversation."""
        if not self.current_step:
            return NodeStatus.FAILURE
        
        message = self.current_step.input_text.lower()
        small_talk_triggers = ["how are you", "what's up", "nice", "weather", "good"]
        
        if any(trigger in message for trigger in small_talk_triggers):
            # Use the small talk tool
            return NodeStatus.SUCCESS if self.tool_result else NodeStatus.RUNNING
        return NodeStatus.FAILURE

class SimpleConversationAgent(BTAgent):
    """A simple conversational agent that can greet, chat, and say goodbye."""
    
    def __init__(self):
        super().__init__(
            name="SimpleConversationAgent",
            instructions="""You are a friendly conversational agent that can:
            1. Greet users
            2. Engage in small talk
            3. Answer questions
            4. Say goodbye when appropriate
            
            Always be polite and helpful in your responses.""",
            tools=[handle_small_talk]
        )

    def setup_tree(self) -> BTNode:
        """Set up the behavior tree structure."""
        return SequenceNode("main", [
            GreetAction("greet", self),
            UnderstandIntentAction("understand_intent", self),
            SelectorNode("response", [
                SmallTalkAction("small_talk", self),
                # Other response actions...
            ])
        ])

async def main():
    # Create and run the agent
    agent = SimpleConversationAgent()
    
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

if __name__ == "__main__":
    asyncio.run(main()) 