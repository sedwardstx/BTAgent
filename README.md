# BTAgent Framework

A framework for creating AI agents using OpenAI's Agent SDK integrated with Behavior Tree patterns.

## Overview

This framework combines the power of OpenAI's Agent SDK with behavior tree patterns to create structured, predictable AI agents. Each behavior tree node is treated as an agent step, allowing for complex decision-making processes while maintaining clear control flow.

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your OpenAI API key in a `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
BTAgent/
├── bt_agent/
│   ├── __init__.py
│   ├── core.py          # Core agent implementation
│   ├── nodes.py         # Behavior tree node implementations
│   └── utils.py         # Utility functions
├── examples/
│   └── simple_agent.py  # Example agent implementation
├── requirements.txt
└── README.md
```

## Usage

The framework provides a base `BTAgent` class that can be extended to create custom AI agents. Each agent is defined by:
1. A behavior tree structure
2. Node implementations that map to agent actions
3. OpenAI Agent SDK tools and instructions

Example:

```python
from bt_agent.core import BTAgent
from bt_agent.nodes import ActionNode, SequenceNode

class MyCustomAgent(BTAgent):
    def setup_tree(self):
        return SequenceNode("root", [
            ActionNode("greet", self.greet_action),
            ActionNode("process", self.process_action)
        ])

    def greet_action(self):
        # Implementation
        pass

    def process_action(self):
        # Implementation
        pass
```

## License

MIT License 