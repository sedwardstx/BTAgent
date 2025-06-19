# BTAgent - Behavior Tree AI Agent Framework

A powerful AI agent framework that combines OpenAI's Agents SDK with behavior tree execution for structured, reliable AI workflows.

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd BTAgent

# Install dependencies
pip install -r requirements.txt

# Install BTEngine (required)
pip install -e BTEngine/BehaviorTreeEngine
```

### Basic Usage
```python
from bt_agent.core import BTAgent, BTAgentAction
from behavior_tree_engine.core import NodeStatus

class SimpleAgent(BTAgent):
    def setup_tree(self):
        return BTAgentAction("greet", self)

agent = SimpleAgent(
    name="MyAgent",
    instructions="You are a helpful assistant."
)

# Execute the behavior tree
status = await agent.execute_tree({"message": "Hello!"})
```

## 🔧 Troubleshooting

### ❌ Error: "cannot import name 'Timeout' from 'behavior_tree_engine.core'"

This indicates you have an **old version** of BTEngine. Here are quick fixes:

#### Option 1: Automated Reset (Recommended)
```bash
# Check your current version
python check_btengine_version.py

# Reset to the correct version
python reset_btengine.py
```

#### Option 2: Windows Batch Script
```bash
# Double-click or run from command prompt
reset_btengine.bat
```

#### Option 3: Manual Reset
```bash
# Uninstall old versions
pip uninstall behavior-tree-engine behavior_tree_engine btengine -y
pip cache purge

# Install correct version
pip install -e BTEngine/BehaviorTreeEngine

# Verify installation
python -c "from behavior_tree_engine.core import Timeout, AsyncAction; print('✅ Success!')"
```

### 🌐 For Remote/Separate Computers

If you're deploying to a different computer:

1. **Clone the full repository** (including BTEngine subdirectory)
2. **Run the reset script**: `python reset_btengine.py`
3. **Verify installation**: `python check_btengine_version.py`

## 📊 Examples

### Simple Conversational Agent
```bash
python examples/simple_agent.py
```

### Complex Task Execution
```bash
python examples/complex_task_agent.py
```

### Compatible Version (Works with Old BTEngine)
```bash
python examples/complex_task_agent_compatible.py
```

## ✅ Verification

To verify everything is working correctly:

```bash
# Check BTEngine version and features
python check_btengine_version.py

# Run the examples
python examples/simple_agent.py
python examples/complex_task_agent.py
```

You should see:
- ✅ All BTEngine features available
- ✅ Examples run without import errors
- ✅ "Task execution completed with status: SUCCESS"

## 🏗️ Architecture

BTAgent combines:
- **OpenAI Agents SDK** for AI capabilities and tool calling
- **BTEngine** for behavior tree execution and flow control
- **Async/await support** for high-performance operations
- **Shared memory system** for inter-node communication

## 📚 Documentation

For detailed setup and troubleshooting, see:
- [`BTENGINE_SETUP.md`](BTENGINE_SETUP.md) - Complete setup guide
- [`examples/`](examples/) - Working examples
- [`bt_agent/core.py`](bt_agent/core.py) - Core framework code

## 🆘 Getting Help

If you're still having issues:

1. **Run diagnostics**: `python check_btengine_version.py`
2. **Check the setup guide**: [`BTENGINE_SETUP.md`](BTENGINE_SETUP.md)
3. **Try the reset script**: `python reset_btengine.py`
4. **Verify with examples**: `python examples/simple_agent.py`

## 🎯 Features

- ✅ **Async execution** with proper await support
- ✅ **Retry and timeout decorators** for robust operations
- ✅ **Shared memory** for complex data flows
- ✅ **Tool integration** with OpenAI Agents SDK
- ✅ **YAML configuration** support
- ✅ **Production ready** with comprehensive error handling 