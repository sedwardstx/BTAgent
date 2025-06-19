# BTAgent Developer's Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Installation](#installation)
4. [Core Components](#core-components)
5. [Creating Custom Agents](#creating-custom-agents)
6. [Behavior Tree Concepts](#behavior-tree-concepts)
7. [Advanced Usage](#advanced-usage)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Introduction

BTAgent is a framework that combines OpenAI's Agent SDK with behavior tree patterns to create structured, predictable AI agents. It leverages the BTEngine library for behavior tree implementation and adds an integration layer for OpenAI's language models.

### Key Features
- Integration with OpenAI's Agent SDK
- Structured decision-making using behavior trees
- Event-driven architecture with step callbacks
- Extensible action node system
- Built-in utilities for tree visualization and debugging

## Architecture Overview

The framework consists of several key components:

```
BTAgent/
├── Core Agent Layer (BTAgent)
│   ├── OpenAI Agent Integration
│   └── Behavior Tree Management
├── Action System (BTAgentAction)
│   ├── Step Management
│   └── State Tracking
└── Tool System (BTAgentTool)
    ├── Node Execution
    └── Error Handling
```

### Component Interaction Flow
1. User input → BTAgent
2. BTAgent → OpenAI Agent SDK
3. Agent Steps → Behavior Tree Nodes
4. Node Results → Agent Response

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sedwardstx/BTAgent.git
cd BTAgent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
# Create a .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Core Components

### BTAgent Class
The base class for creating AI agents with behavior tree integration.

```python
from bt_agent import BTAgent

class MyAgent(BTAgent):
    def __init__(self):
        super().__init__(
            name="MyAgent",
            instructions="Agent instructions here",
            tools=[]  # Optional tools
        )

    def setup_tree(self):
        # Define your behavior tree structure
        return root_node
```

### BTAgentAction Class
Base class for creating action nodes that can access agent state.

```python
from bt_agent import BTAgentAction, NodeStatus

class MyAction(BTAgentAction):
    def tick(self) -> NodeStatus:
        if not self.current_step:
            return NodeStatus.FAILURE
            
        # Access current step information
        message = self.current_step.input_text
        
        # Implement action logic
        if some_condition:
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE
```

### BTAgentTool Class
Wrapper for executing behavior tree nodes as agent tools.

```python
from bt_agent import BTAgentTool

tool = BTAgentTool(
    name="my_tool",
    description="Tool description",
    node=my_action_node
)
```

## Creating Custom Agents

### 1. Define Action Nodes

```python
class GreetAction(BTAgentAction):
    def tick(self) -> NodeStatus:
        if not self.current_step:
            return NodeStatus.FAILURE
            
        if not self.current_step.steps:  # First interaction
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE
```

### 2. Create Agent Class

```python
from btengine.nodes import SequenceNode, SelectorNode

class MyCustomAgent(BTAgent):
    def __init__(self):
        super().__init__(
            name="CustomAgent",
            instructions="""
            Agent instructions here.
            Specify capabilities and behavior.
            """
        )

    def setup_tree(self):
        return SequenceNode("root", [
            GreetAction("greet", self),
            ProcessAction("process", self),
            RespondAction("respond", self)
        ])
```

### 3. Run the Agent

```python
async def main():
    agent = MyCustomAgent()
    result = await agent.run("User input here")
    print(result['output'])
```

## Behavior Tree Concepts

### Node Types

1. **Sequence Node**: Executes children in order until one fails
```python
sequence = SequenceNode("sequence", [
    ActionA("action_a", agent),
    ActionB("action_b", agent)
])
```

2. **Selector Node**: Tries children in order until one succeeds
```python
selector = SelectorNode("selector", [
    PrimaryAction("primary", agent),
    FallbackAction("fallback", agent)
])
```

3. **Parallel Node**: Executes children simultaneously
```python
parallel = ParallelNode("parallel", [
    MonitorAction("monitor", agent),
    ProcessAction("process", agent)
])
```

### Node Status Values
- `NodeStatus.SUCCESS`: Action completed successfully
- `NodeStatus.FAILURE`: Action failed
- `NodeStatus.RUNNING`: Action is still in progress
- `NodeStatus.READY`: Action hasn't started yet

## Advanced Usage

### Step Callbacks
```python
async def step_callback(step: AgentStep) -> None:
    print(f"Current step: {step.input_text}")
    # Process step information

agent = MyAgent()
result = await agent.run(
    "input",
    callbacks={"on_step": step_callback}
)
```

### Tree Visualization
```python
from bt_agent import print_tree

# Print the current state of the behavior tree
print_tree(agent.behavior_tree)
```

### State Management
```python
class StatefulAction(BTAgentAction):
    def __init__(self, name: str, agent: BTAgent):
        super().__init__(name, agent)
        self.state = {}

    def tick(self) -> NodeStatus:
        # Access and modify state
        self.state['last_input'] = self.current_step.input_text
        return NodeStatus.SUCCESS
```

## Best Practices

1. **Action Node Design**
   - Keep actions focused and single-purpose
   - Handle missing or invalid state gracefully
   - Use meaningful names for nodes and actions

2. **Tree Structure**
   - Organize nodes logically
   - Use selectors for fallback behavior
   - Keep trees shallow when possible

3. **Error Handling**
   - Always check for None/invalid states
   - Provide meaningful error messages
   - Use appropriate status returns

4. **Testing**
   - Test individual action nodes
   - Verify tree behavior with different inputs
   - Mock OpenAI responses for testing

## Troubleshooting

### Common Issues

1. **Node Not Executing**
   - Check parent node status
   - Verify node connections
   - Debug with print_tree()

2. **Agent Not Responding**
   - Check OpenAI API key
   - Verify tool registration
   - Check for error responses

3. **Unexpected Behavior**
   - Log node status changes
   - Verify step callback execution
   - Check action preconditions

### Debugging Tools

```python
# Tree status inspection
from bt_agent import get_tree_status
status = get_tree_status(agent.behavior_tree)

# Node finding
from bt_agent import find_node_by_name
node = find_node_by_name(agent.behavior_tree, "node_name")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write/update tests
5. Submit a pull request

For more information, see the [contribution guidelines](CONTRIBUTING.md). 