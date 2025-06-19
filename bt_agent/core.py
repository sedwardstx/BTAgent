from typing import Optional, Dict, Any, List, Callable, Union, TypeVar, Generic
import asyncio
import logging
from dataclasses import dataclass, field
from agents import Agent, Runner
from agents.tools.base import BaseTool
from agents.lifecycle import AgentHooks
from agents.function_tool import function_tool
from agents.handoffs import Handoff
from btengine.base import BTNode, NodeStatus
from btengine.nodes import ActionNodeBase, SequenceNode, SelectorNode, ParallelNode, DecoratorNodeBase
import yaml

logger = logging.getLogger(__name__)

TContext = TypeVar('TContext')

@dataclass
class BTExecutionContext:
    """Context shared across all behavior tree nodes during execution."""
    current_input: Optional[str] = None
    tool_results: Dict[str, Any] = field(default_factory=dict)
    shared_memory: Dict[str, Any] = field(default_factory=dict)
    execution_count: int = 0
    max_executions: int = 100

class BTAgentAction(ActionNodeBase):
    """Base class for agent action nodes that integrate with the Agents SDK."""
    
    def __init__(self, name: str, agent: 'BTAgent', context_key: Optional[str] = None):
        super().__init__(name)
        self.agent = agent
        self.context_key = context_key or name
        self._last_status = NodeStatus.READY
        
    @property
    def execution_context(self) -> BTExecutionContext:
        """Get the current execution context."""
        return self.agent.execution_context
    
    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """Get data from shared memory."""
        return self.execution_context.shared_memory.get(key, default)
    
    def set_shared_data(self, key: str, value: Any) -> None:
        """Set data in shared memory."""
        self.execution_context.shared_memory[key] = value
    
    def get_tool_result(self, tool_name: str) -> Any:
        """Get the result of a specific tool call."""
        return self.execution_context.tool_results.get(tool_name)
    
    async def execute_async(self) -> NodeStatus:
        """Override this method for async execution logic."""
        return await self.execute()
    
    async def execute(self) -> NodeStatus:
        """Override this method for execution logic."""
        return NodeStatus.SUCCESS
    
    def tick(self) -> NodeStatus:
        """Execute the node synchronously."""
        try:
            # Check execution limits
            if self.execution_context.execution_count >= self.execution_context.max_executions:
                logger.warning(f"Max executions reached for {self.name}")
                return NodeStatus.FAILURE
            
            self.execution_context.execution_count += 1
            
            # Try async execution first
            if hasattr(self, 'execute_async'):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're already in an async context, create a task
                        task = asyncio.create_task(self.execute_async())
                        # This is a simplification - in practice you'd want proper async handling
                        return NodeStatus.RUNNING
                    else:
                        return loop.run_until_complete(self.execute_async())
                except Exception as e:
                    logger.error(f"Async execution failed for {self.name}: {e}")
                    return NodeStatus.FAILURE
            else:
                return self.execute()
                
        except Exception as e:
            logger.error(f"Node {self.name} execution failed: {e}")
            return NodeStatus.FAILURE

class BTToolAction(BTAgentAction):
    """Action node that executes a specific tool."""
    
    def __init__(self, name: str, agent: 'BTAgent', tool_name: str, tool_params: Optional[Dict[str, Any]] = None):
        super().__init__(name, agent)
        self.tool_name = tool_name
        self.tool_params = tool_params or {}
    
    async def execute_async(self) -> NodeStatus:
        """Execute the tool and return the appropriate status."""
        try:
            # Find the tool by name
            tool = None
            for t in self.agent.tools:
                if hasattr(t, 'name') and t.name == self.tool_name:
                    tool = t
                    break
            
            if not tool:
                logger.error(f"Tool {self.tool_name} not found")
                return NodeStatus.FAILURE
            
            # Execute the tool
            result = await tool.run(**self.tool_params)
            
            # Store the result
            self.execution_context.tool_results[self.tool_name] = result
            self.set_shared_data(f"{self.name}_result", result)
            
            return NodeStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"Tool {self.tool_name} execution failed: {e}")
            return NodeStatus.FAILURE

class BTHandoffAction(BTAgentAction):
    """Action node that performs handoff to another agent."""
    
    def __init__(self, name: str, agent: 'BTAgent', target_agent: Union[Agent, str], handoff_message: Optional[str] = None):
        super().__init__(name, agent)
        self.target_agent = target_agent
        self.handoff_message = handoff_message
    
    async def execute_async(self) -> NodeStatus:
        """Execute the handoff to another agent."""
        try:
            # Prepare handoff message
            message = self.handoff_message or f"Handing off from {self.agent.name} to handle the current task"
            
            # If target_agent is a string, find it in the agent's handoffs
            if isinstance(self.target_agent, str):
                target = None
                for handoff in self.agent.handoffs:
                    if hasattr(handoff, 'name') and handoff.name == self.target_agent:
                        target = handoff
                        break
                if not target:
                    logger.error(f"Target agent {self.target_agent} not found in handoffs")
                    return NodeStatus.FAILURE
                self.target_agent = target
            
            # Execute handoff (this would typically be handled by the Agent SDK)
            # For now, we'll store the handoff intention
            self.set_shared_data("handoff_target", self.target_agent)
            self.set_shared_data("handoff_message", message)
            
            return NodeStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"Handoff to {self.target_agent} failed: {e}")
            return NodeStatus.FAILURE

class BTConditionNode(BTAgentAction):
    """Condition node that evaluates a condition and returns SUCCESS/FAILURE."""
    
    def __init__(self, name: str, agent: 'BTAgent', condition_func: Callable[['BTExecutionContext'], bool]):
        super().__init__(name, agent)
        self.condition_func = condition_func
    
    def tick(self) -> NodeStatus:
        """Evaluate the condition."""
        try:
            result = self.condition_func(self.execution_context)
            return NodeStatus.SUCCESS if result else NodeStatus.FAILURE
        except Exception as e:
            logger.error(f"Condition {self.name} evaluation failed: {e}")
            return NodeStatus.FAILURE

class BTAgentHooks(AgentHooks):
    """Hooks that bridge between Agent SDK events and behavior tree execution."""
    
    def __init__(self, agent: 'BTAgent'):
        self.agent = agent
        
    async def on_agent_start(self, agent_name: str) -> None:
        """Called when agent starts execution."""
        logger.info(f"Agent {agent_name} starting execution")
        if self.agent.behavior_tree:
            self.agent.execution_context.execution_count = 0
            
    async def on_tool_start(self, tool_name: str, tool_input: Dict[str, Any]) -> None:
        """Called when a tool starts executing."""
        logger.debug(f"Tool {tool_name} starting with input: {tool_input}")
        
    async def on_tool_end(self, tool_name: str, tool_output: Any) -> None:
        """Called when a tool finishes executing."""
        logger.debug(f"Tool {tool_name} completed with output: {tool_output}")
        # Store tool result in execution context
        self.agent.execution_context.tool_results[tool_name] = tool_output
        
    async def on_error(self, error: Exception) -> None:
        """Called when an error occurs during agent execution."""
        logger.error(f"Agent execution error: {error}")
        # Mark current tree execution as failed
        if self.agent.behavior_tree:
            self.agent.behavior_tree.status = NodeStatus.FAILURE

class BTAgent(Agent, Generic[TContext]):
    """A behavior tree-based AI agent that integrates with the OpenAI Agents SDK."""
    
    def __init__(
        self,
        name: str,
        instructions: str,
        tools: Optional[List[BaseTool]] = None,
        handoffs: Optional[List[Union[Agent, Handoff]]] = None,
        mcp_servers: Optional[List[Any]] = None,
        model: str = "gpt-4o",
        context: Optional[TContext] = None,
        tree_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name=name,
            instructions=instructions,
            tools=tools or [],
            handoffs=handoffs or [],
            mcp_servers=mcp_servers or [],
            model=model,
            hooks=BTAgentHooks(self)
        )
        
        self.context = context
        self.behavior_tree: Optional[BTNode] = None
        self.execution_context = BTExecutionContext()
        self.tree_config = tree_config or {}
        self._node_registry: Dict[str, type] = {
            'action': BTAgentAction,
            'tool': BTToolAction,
            'handoff': BTHandoffAction,
            'condition': BTConditionNode,
            'sequence': SequenceNode,
            'selector': SelectorNode,
            'parallel': ParallelNode,
        }
        
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the behavior tree."""
        self.behavior_tree = self.setup_tree()
        if not self.behavior_tree:
            raise ValueError("Behavior tree not set up. Override setup_tree() method or provide tree_config.")

    def setup_tree(self) -> BTNode:
        """Override this method to define the behavior tree structure programmatically."""
        if self.tree_config:
            return self.build_tree_from_config(self.tree_config)
        raise NotImplementedError("Subclasses must implement setup_tree() or provide tree_config.")
    
    def build_tree_from_config(self, config: Dict[str, Any]) -> BTNode:
        """Build a behavior tree from configuration."""
        return self._build_node_recursive(config.get('root', config))
    
    def _build_node_recursive(self, node_config: Dict[str, Any]) -> BTNode:
        """Recursively build nodes from configuration."""
        node_type = node_config.get('type', 'action')
        node_name = node_config.get('name', f'node_{id(node_config)}')
        
        if node_type in ['sequence', 'selector', 'parallel']:
            # Composite nodes
            children = []
            for child_config in node_config.get('children', []):
                children.append(self._build_node_recursive(child_config))
            
            node_class = self._node_registry[node_type]
            return node_class(node_name, children)
        
        elif node_type == 'tool':
            # Tool action node
            tool_name = node_config.get('tool_name', node_name)
            tool_params = node_config.get('params', {})
            return BTToolAction(node_name, self, tool_name, tool_params)
        
        elif node_type == 'handoff':
            # Handoff action node
            target_agent = node_config.get('target_agent')
            handoff_message = node_config.get('message')
            return BTHandoffAction(node_name, self, target_agent, handoff_message)
        
        elif node_type == 'condition':
            # Condition node - would need to be defined in subclass
            condition_func = node_config.get('condition_func')
            if not condition_func:
                raise ValueError(f"Condition node {node_name} requires condition_func")
            return BTConditionNode(node_name, self, condition_func)
        
        else:
            # Default action node
            node_class = node_config.get('class', BTAgentAction)
            if isinstance(node_class, str):
                node_class = self._node_registry.get(node_class, BTAgentAction)
            return node_class(node_name, self)
    
    def register_node_type(self, name: str, node_class: type) -> None:
        """Register a custom node type."""
        self._node_registry[name] = node_class
    
    def load_tree_from_yaml(self, yaml_path: str) -> None:
        """Load behavior tree configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        self.tree_config = config
        self.behavior_tree = self.build_tree_from_config(config)
    
    async def execute_tree(self, input_data: Optional[Dict[str, Any]] = None) -> NodeStatus:
        """Execute the behavior tree with optional input data."""
        if not self.behavior_tree:
            return NodeStatus.FAILURE
        
        # Reset execution context
        self.execution_context = BTExecutionContext()
        if input_data:
            self.execution_context.shared_memory.update(input_data)
        
        # Execute the tree
        try:
            status = self.behavior_tree.tick()
            logger.info(f"Behavior tree execution completed with status: {status}")
            return status
        except Exception as e:
            logger.error(f"Behavior tree execution failed: {e}")
            return NodeStatus.FAILURE
    
    def tick(self) -> NodeStatus:
        """Execute one tick of the behavior tree."""
        if not self.behavior_tree:
            return NodeStatus.FAILURE
        return self.behavior_tree.tick()

    def reset(self) -> None:
        """Reset the agent's state."""
        if self.behavior_tree:
            self.behavior_tree.reset()
        self.execution_context = BTExecutionContext()
        super().reset()
    
    def get_tree_status(self) -> Dict[str, Any]:
        """Get the current status of the behavior tree."""
        if not self.behavior_tree:
            return {"status": "no_tree"}
        
        return {
            "tree_status": self.behavior_tree.status.name,
            "execution_count": self.execution_context.execution_count,
            "shared_memory_keys": list(self.execution_context.shared_memory.keys()),
            "tool_results_keys": list(self.execution_context.tool_results.keys())
        }

# Utility functions for creating common behavior tree patterns
def create_retry_decorator(child: BTNode, max_attempts: int = 3) -> BTNode:
    """Create a retry decorator that retries a child node up to max_attempts times."""
    class RetryDecorator(DecoratorNodeBase):
        def __init__(self, name: str, child: BTNode, max_attempts: int):
            super().__init__(name, child)
            self.max_attempts = max_attempts
            self.attempt_count = 0
        
        def tick(self) -> NodeStatus:
            if self.attempt_count >= self.max_attempts:
                return NodeStatus.FAILURE
            
            status = self.child.tick()
            if status == NodeStatus.FAILURE:
                self.attempt_count += 1
                if self.attempt_count < self.max_attempts:
                    self.child.reset()
                    return NodeStatus.RUNNING
            elif status == NodeStatus.SUCCESS:
                self.attempt_count = 0
            
            return status
        
        def reset(self) -> None:
            super().reset()
            self.attempt_count = 0
    
    return RetryDecorator(f"retry_{child.name}", child, max_attempts)

def create_timeout_decorator(child: BTNode, timeout_seconds: float) -> BTNode:
    """Create a timeout decorator that fails the child if it takes too long."""
    import time
    
    class TimeoutDecorator(DecoratorNodeBase):
        def __init__(self, name: str, child: BTNode, timeout: float):
            super().__init__(name, child)
            self.timeout = timeout
            self.start_time = None
        
        def tick(self) -> NodeStatus:
            if self.start_time is None:
                self.start_time = time.time()
            
            if time.time() - self.start_time > self.timeout:
                return NodeStatus.FAILURE
            
            status = self.child.tick()
            if status != NodeStatus.RUNNING:
                self.start_time = None
            
            return status
        
        def reset(self) -> None:
            super().reset()
            self.start_time = None
    
    return TimeoutDecorator(f"timeout_{child.name}", child, timeout_seconds) 