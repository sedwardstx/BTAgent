import asyncio
import yaml
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
    
    def __init__(self, config_path: str = "examples/simple_agent.yaml"):
        # Load configuration from YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        super().__init__(
            name=config['name'],
            instructions=config['description']
        )
        self.config = config

    def setup_tree(self):
        # Create a mapping of action class names to their actual classes
        action_classes = {
            'GreetAction': GreetAction,
            'UnderstandIntentAction': UnderstandIntentAction,
            'SmallTalkAction': SmallTalkAction,
            'AnswerQuestionAction': AnswerQuestionAction,
            'FarewellAction': FarewellAction
        }
        
        def build_node(node_config):
            if node_config['type'] == 'action':
                action_class = action_classes[node_config['class']]
                return action_class(node_config['name'], self)
            elif node_config['type'] == 'sequence':
                children = [build_node(child) for child in node_config['children']]
                return SequenceNode(node_config['name'], children)
            elif node_config['type'] == 'selector':
                children = [build_node(child) for child in node_config['children']]
                return SelectorNode(node_config['name'], children)
            else:
                raise ValueError(f"Unknown node type: {node_config['type']}")
        
        # Build the tree from the YAML configuration
        return build_node(self.config['behavior_tree']['root'])

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