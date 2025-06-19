from typing import Optional, Dict, Any, List
from agents import Agent, Runner
from agents.exceptions import AgentError
from agents.items import AgentFinish, AgentStep, AgentMessage
from agents.tools.base import BaseTool
from agents.lifecycle import AgentHooks
from btengine.base import BTNode, NodeStatus
from btengine.nodes import ActionNodeBase, SequenceNode

class BTAgentAction(ActionNodeBase):
    """Base class for agent action nodes that integrate with the Agents SDK."""
    def __init__(self, name: str, agent: 'BTAgent'):
        super().__init__(name)
        self.agent = agent
        self.current_step: Optional[AgentStep] = None
        self.tool_result: Optional[Any] = None

    def update_step(self, step: AgentStep) -> None:
        """Update the current agent step."""
        self.current_step = step
        
    def update_tool_result(self, result: Any) -> None:
        """Update the result from a tool call."""
        self.tool_result = result

class BTAgentHooks(AgentHooks):
    """Hooks that bridge between Agent SDK events and behavior tree execution."""
    
    def __init__(self, agent: 'BTAgent'):
        self.agent = agent
        
    async def on_tool_start(self, tool_name: str, tool_input: Dict[str, Any]) -> None:
        """Called when a tool starts executing."""
        # Tool start could trigger node status update
        if self.agent.behavior_tree:
            self.agent.behavior_tree.tick()
            
    async def on_tool_end(self, tool_name: str, tool_output: Any) -> None:
        """Called when a tool finishes executing."""
        # Update the current action node with the tool result
        if isinstance(self.agent.behavior_tree, BTAgentAction):
            self.agent.behavior_tree.update_tool_result(tool_output)
        self.agent.behavior_tree.tick()
            
    async def on_tool_error(self, tool_name: str, error: Exception) -> None:
        """Called when a tool execution fails."""
        # Tool error maps to node failure
        if self.agent.behavior_tree:
            if isinstance(error, AgentError):
                self.agent.behavior_tree.status = NodeStatus.FAILURE
            self.agent.behavior_tree.tick()

class BTAgent(Agent):
    """A behavior tree-based AI agent that integrates with the OpenAI Agents SDK."""
    
    def __init__(
        self,
        name: str,
        instructions: str,
        tools: Optional[List[BaseTool]] = None,
        model: str = "gpt-4-turbo-preview"
    ):
        super().__init__(
            name=name,
            instructions=instructions,
            tools=tools or [],
            model=model,
            hooks=BTAgentHooks(self)
        )
        self.behavior_tree: Optional[BTNode] = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the behavior tree."""
        self.behavior_tree = self.setup_tree()
        if not self.behavior_tree:
            raise ValueError("Behavior tree not set up. Override setup_tree() method.")

    def setup_tree(self) -> BTNode:
        """Override this method to define the behavior tree structure."""
        raise NotImplementedError("Subclasses must implement setup_tree()")

    def tick(self) -> NodeStatus:
        """Execute one tick of the behavior tree."""
        if not self.behavior_tree:
            return NodeStatus.FAILURE
        return self.behavior_tree.tick()

    def reset(self) -> None:
        """Reset the agent's state."""
        if self.behavior_tree:
            self.behavior_tree.reset()
        super().reset()

    def map_tool_to_node_status(self, tool_result: Any) -> NodeStatus:
        """Map a tool result to a node status."""
        if isinstance(tool_result, AgentFinish):
            return NodeStatus.SUCCESS
        elif isinstance(tool_result, AgentError):
            return NodeStatus.FAILURE
        return NodeStatus.RUNNING

class BTAgentTool(BaseTool):
    """Base class for behavior tree agent tools."""
    
    def __init__(self, name: str, description: str, node: BTNode):
        super().__init__(name=name, description=description)
        self.node = node

    async def run(self, **kwargs) -> Any:
        """Execute the tool by ticking the associated node."""
        status = self.node.tick()
        if status == NodeStatus.FAILURE:
            raise AgentError(f"Node {self.node.name} failed")
        return {
            "status": status.value,
            "node": self.node.name
        } 