import asyncio
from typing import Dict, Any
from bt_agent.core import BTAgent, BTAgentAction
from btengine.base import NodeStatus
from btengine.nodes import SequenceNode, SelectorNode

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

class SmallTalkAction(BTAgentAction):
    def tick(self) -> NodeStatus:
        """Handle casual conversation."""
        if not self.current_step:
            return NodeStatus.FAILURE
        
        message = self.current_step.input_text.lower()
        small_talk_triggers = ["how are you", "what's up", "nice", "weather", "good"]
        
        if any(trigger in message for trigger in small_talk_triggers):
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

class AnswerQuestionAction(BTAgentAction):
    def tick(self) -> NodeStatus:
        """Handle questions from the user."""
        if not self.current_step:
            return NodeStatus.FAILURE
        
        message = self.current_step.input_text.lower()
        question_indicators = ["what", "why", "how", "when", "where", "who", "?"]
        
        if any(indicator in message for indicator in question_indicators):
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

class FarewellAction(BTAgentAction):
    def tick(self) -> NodeStatus:
        """Handle saying goodbye."""
        if not self.current_step:
            return NodeStatus.FAILURE
        
        message = self.current_step.input_text.lower()
        if any(word in message for word in ["bye", "goodbye", "see you", "farewell"]):
            return NodeStatus.SUCCESS
        return NodeStatus.RUNNING

class SimpleConversationAgent(BTAgent):
    """A simple conversational agent that can greet, chat, and say goodbye."""
    
    def __init__(self):
        super().__init__(
            name="ConversationAgent",
            instructions="""You are a friendly conversational agent that can:
            1. Greet users
            2. Engage in small talk
            3. Answer questions
            4. Say goodbye when appropriate
            
            Use the available tools to structure the conversation naturally."""
        )

    def setup_tree(self):
        # Create nodes for different conversation phases
        greet_node = GreetAction("greet", self)
        
        conversation_sequence = SequenceNode("conversation", [
            UnderstandIntentAction("understand_intent", self),
            SelectorNode("response", [
                SmallTalkAction("small_talk", self),
                AnswerQuestionAction("answer_question", self)
            ])
        ])
        
        farewell_node = FarewellAction("farewell", self)
        
        # Create the main sequence
        return SequenceNode("main", [
            greet_node,
            conversation_sequence,
            farewell_node
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
        result = await agent.run(message)
        print(f"Agent: {result['output']}")

if __name__ == "__main__":
    asyncio.run(main()) 