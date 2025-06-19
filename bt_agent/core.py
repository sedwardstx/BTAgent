from typing import Optional, Dict, Any, List
from agents import Agent, Runner
from agents.exceptions import AgentError
from agents.items import AgentFinish, AgentStep
from agents.tools import BaseTool
from btengine.base import BTNode, NodeStatus
from btengine.nodes import ActionNodeBase

class BTAgentAction(ActionNodeBase):
    """Base class for agent action nodes."""
    def __init__(self, name: str, agent: 'BTAgent'):
        super().__init__(name)
        self.agent = agent
        self.current_step: Optional[AgentStep] = None

    def update_step(self, step: AgentStep) -> None:
        """Update the current agent step."""
        self.current_step = step

class BTAgent:
    """Base class for behavior tree-based AI agents."""
    
    def __init__(
        self,
        name: str,
        instructions: str,
        tools: Optional[List[BaseTool]] = None,
        model: str = "gpt-4-turbo-preview"
    ):
        self.name = name
        self.agent = Agent(
            name=name,
            instructions=instructions,
            tools=tools or [],
            model=model
        )
        self.behavior_tree: Optional[BTNode] = None
        self.current_step: Optional[AgentStep] = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the behavior tree."""
        self.behavior_tree = self.setup_tree()
        if not self.behavior_tree:
            raise ValueError("Behavior tree not set up. Override setup_tree() method.")

    def setup_tree(self) -> BTNode:
        """Override this method to define the behavior tree structure."""
        raise NotImplementedError("Subclasses must implement setup_tree()")

    async def run(self, input_text: str) -> Dict[str, Any]:
        """Run the agent with the given input."""
        self.reset()
        
        async def step_callback(step: AgentStep) -> None:
            """Callback for each agent step."""
            self.current_step = step
            # Update all action nodes with the current step
            self._update_action_nodes(self.behavior_tree, step)

        try:
            result = await Runner.run(
                self.agent,
                input_text,
                callbacks={"on_step": step_callback}
            )
            return {
                "status": "success",
                "output": result.final_output,
                "steps": result.steps
            }
        except AgentError as e:
            return {
                "status": "error",
                "error": str(e),
                "steps": e.steps if hasattr(e, 'steps') else []
            }

    def _update_action_nodes(self, node: BTNode, step: AgentStep) -> None:
        """Recursively update all action nodes with the current step."""
        if isinstance(node, BTAgentAction):
            node.update_step(step)
        
        # Update children if any
        if hasattr(node, 'children'):
            for child in node.children:
                self._update_action_nodes(child, step)
        elif hasattr(node, 'child'):
            self._update_action_nodes(node.child, step)

    def tick(self) -> NodeStatus:
        """Execute one tick of the behavior tree."""
        if not self.behavior_tree:
            return NodeStatus.FAILURE
        return self.behavior_tree.tick()

    def reset(self) -> None:
        """Reset the agent's state."""
        if self.behavior_tree:
            self.behavior_tree.reset()
        self.current_step = None

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