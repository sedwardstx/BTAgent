from typing import Optional, Dict, Any, List, Callable, Union, TypeVar, Generic
import asyncio
import logging
from dataclasses import dataclass, field
from agents import Agent, Runner
from agents import Tool as BaseTool
from agents import AgentHooks
from agents import function_tool
from agents.handoffs import Handoff
# Now using the full improved BTEngine with all new features!
from behavior_tree_engine.core import (
    NodeStatus, Node as BTNode, Action as ActionNodeBase, 
    Sequence as SequenceNode, Selector as SelectorNode, Parallel as ParallelNode,
    Decorator as DecoratorNodeBase, Inverter, Timeout, Repeater, RetryUntilSuccess,
    AsyncAction, BehaviorTree  # ✅ Now available!
)
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

class BTAgentAsyncAction(AsyncAction):
    """✅ Now using the real AsyncAction base class from BTEngine!"""
    
    def __init__(self, name: str, agent: 'BTAgent', context_key: Optional[str] = None):
        super().__init__(name)
        self.agent = agent
        self.context_key = context_key or name
        
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
    
    async def _async_tick(self) -> NodeStatus:
        """✅ Now using the proper BTEngine async method!"""
        try:
            # Check execution limits
            if self.execution_context.execution_count >= self.execution_context.max_executions:
                logger.warning(f"Max executions reached for {self.name}")
                return NodeStatus.FAILURE
            
            self.execution_context.execution_count += 1
            
            # Call the async execute method
            return await self.execute_async()
                
        except Exception as e:
            logger.error(f"Async node {self.name} execution failed: {e}")
            return NodeStatus.FAILURE
    
    async def execute_async(self) -> NodeStatus:
        """Override this method for async execution logic."""
        return NodeStatus.SUCCESS

class BTAgentAction(ActionNodeBase):
    """Base class for agent action nodes that integrate with the Agents SDK."""
    
    def __init__(self, name: str, agent: 'BTAgent', context_key: Optional[str] = None):
        super().__init__(name)
        self.agent = agent
        self.context_key = context_key or name
        self._last_status = NodeStatus.READY  # ✅ Now using the real READY status!
        
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
    
    def _tick(self) -> NodeStatus:
        """Execute the node synchronously."""
        try:
            # Check execution limits
            if self.execution_context.execution_count >= self.execution_context.max_executions:
                logger.warning(f"Max executions reached for {self.name}")
                return NodeStatus.FAILURE
            
            self.execution_context.execution_count += 1
            
            # For synchronous operations, just call execute
            return asyncio.run(self.execute())
                
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
        
    def on_agent_start(self, agent_name: str) -> None:
        """Called when agent starts execution."""
        logger.info(f"Agent {agent_name} starting execution")
        if self.agent.behavior_tree:
            self.agent.execution_context.execution_count = 0
            
    def on_tool_start(self, *args, **kwargs) -> None:
        """Called when a tool starts executing."""
        logger.debug(f"Tool starting with {len(args)} args and {len(kwargs)} kwargs")
        if len(args) >= 2 and hasattr(args[1], 'name'):
            logger.debug(f"Tool {args[1].name} starting")
        
    def on_tool_end(self, tool_name: str, tool_output: Any) -> None:
        """Called when a tool finishes executing."""
        logger.debug(f"Tool {tool_name} completed with output: {tool_output}")
        # Store tool result in execution context
        self.agent.execution_context.tool_results[tool_name] = tool_output
        
    def on_error(self, error: Exception) -> None:
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
            model=model
            # hooks=BTAgentHooks(self)  # Commented out for now
        )
        
        self.context = context
        self.behavior_tree: Optional[BehaviorTree] = None  # ✅ Now using the real BehaviorTree wrapper!
        self.execution_context = BTExecutionContext()
        self.tree_config = tree_config or {}
        self._node_registry: Dict[str, type] = {
            'action': BTAgentAction,
            'async_action': BTAgentAsyncAction,
            'tool': BTToolAction,
            'handoff': BTHandoffAction,
            'condition': BTConditionNode,
            'sequence': SequenceNode,
            'selector': SelectorNode,
            'parallel': ParallelNode,
            'inverter': Inverter,
            'timeout': Timeout,
            'repeater': Repeater,
            'retry': RetryUntilSuccess,
        }
        
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the behavior tree."""
        root_node = self.setup_tree()
        if not root_node:
            raise ValueError("Behavior tree not set up. Override setup_tree() method or provide tree_config.")
        
        # ✅ Now using the real BehaviorTree wrapper!
        self.behavior_tree = BehaviorTree(root_node, name=f"{self.name}_tree")

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
        
        elif node_type in ['inverter', 'timeout', 'repeater', 'retry']:
            # Decorator nodes - ✅ Now using improved constructors!
            child_config = node_config.get('child')
            if not child_config:
                raise ValueError(f"Decorator node {node_name} requires a child")
            
            child = self._build_node_recursive(child_config)
            node_class = self._node_registry[node_type]
            
            # ✅ Using the improved flexible constructors
            if node_type == 'timeout':
                timeout_sec = node_config.get('timeout_sec', 5.0)
                return node_class(child, name=node_name, timeout_sec=timeout_sec)
            elif node_type == 'retry':
                max_attempts = node_config.get('max_attempts', 3)
                return node_class(child, name=node_name, max_attempts=max_attempts)
            elif node_type == 'repeater':
                num_repeats = node_config.get('num_repeats', 1)
                return node_class(child, name=node_name, num_repeats=num_repeats)
            else:
                return node_class(child, name=node_name)
        
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
        
        elif node_type == 'async_action':
            # Async action node
            node_class = node_config.get('class', BTAgentAsyncAction)
            if isinstance(node_class, str):
                node_class = self._node_registry.get(node_class, BTAgentAsyncAction)
            return node_class(node_name, self)
        
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
        root_node = self.build_tree_from_config(config)
        self.behavior_tree = BehaviorTree(root_node, name=f"{self.name}_tree")
    
    async def execute_tree(self, input_data: Optional[Dict[str, Any]] = None) -> NodeStatus:
        """✅ Execute the behavior tree using the improved BehaviorTree wrapper!"""
        if not self.behavior_tree:
            return NodeStatus.FAILURE
        
        # Reset execution context
        self.execution_context = BTExecutionContext()
        if input_data:
            self.execution_context.shared_memory.update(input_data)
        
        # ✅ Use the new BehaviorTree async execution!
        try:
            self.behavior_tree.reset()
            status = await self.behavior_tree.run_until_complete_async()
            logger.info(f"Behavior tree execution completed with status: {status}")
            return status
        except Exception as e:
            logger.error(f"Behavior tree execution failed: {e}")
            return NodeStatus.FAILURE
    
    def execute_tree_sync(self, input_data: Optional[Dict[str, Any]] = None) -> NodeStatus:
        """✅ Execute the behavior tree synchronously using the improved wrapper!"""
        if not self.behavior_tree:
            return NodeStatus.FAILURE
        
        # Reset execution context
        self.execution_context = BTExecutionContext()
        if input_data:
            self.execution_context.shared_memory.update(input_data)
        
        # ✅ Use the new BehaviorTree sync execution!
        try:
            self.behavior_tree.reset()
            status = self.behavior_tree.run_until_complete()
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
    
    async def tick_async(self) -> NodeStatus:
        """✅ Execute one async tick using the improved BehaviorTree wrapper!"""
        if not self.behavior_tree:
            return NodeStatus.FAILURE
        return await self.behavior_tree.tick_async()

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
            "tree_status": self.behavior_tree.last_status.name,
            "tick_count": self.behavior_tree.tick_count,
            "execution_count": self.execution_context.execution_count,
            "shared_memory_keys": list(self.execution_context.shared_memory.keys()),
            "tool_results_keys": list(self.execution_context.tool_results.keys())
        }

# Utility functions for creating common behavior tree patterns
def create_retry_decorator(child: BTNode, max_attempts: int = 3) -> BTNode:
    """✅ Create a retry decorator using the improved constructor."""
    return RetryUntilSuccess(child, name=f"retry_{getattr(child, 'name', 'unknown')}", max_attempts=max_attempts)

def create_timeout_decorator(child: BTNode, timeout_seconds: float) -> BTNode:
    """✅ Create a timeout decorator using the improved constructor."""
    return Timeout(child, name=f"timeout_{getattr(child, 'name', 'unknown')}", timeout_sec=timeout_seconds)

def create_repeater_decorator(child: BTNode, repeat_count: int = 1) -> BTNode:
    """✅ Create a repeater decorator using the improved constructor."""
    return Repeater(child, name=f"repeat_{getattr(child, 'name', 'unknown')}", num_repeats=repeat_count)

def create_inverter_decorator(child: BTNode) -> BTNode:
    """✅ Create an inverter decorator using the improved constructor."""
    return Inverter(child, name=f"invert_{getattr(child, 'name', 'unknown')}") 