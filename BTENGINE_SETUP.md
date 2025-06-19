# BTEngine Setup & Troubleshooting Guide

## 🚨 **Common Issue: "cannot import name 'Timeout' from 'behavior_tree_engine.core'"**

This error indicates you have an **old version** of BTEngine installed. The new version includes `Timeout`, `AsyncAction`, `BehaviorTree`, and `NodeStatus.READY`.

## 🔧 **Quick Fix (Automated)**

### Option 1: Use the Reset Script
```bash
# Run the automated reset script
python reset_btengine.py
```

### Option 2: Check Version First
```bash
# Check what version you have
python check_btengine_version.py

# If old version detected, run reset
python reset_btengine.py
```

## 🛠️ **Manual Fix**

### Step 1: Clean Uninstall
```bash
# Uninstall all possible old versions
pip uninstall behavior-tree-engine -y
pip uninstall behavior_tree_engine -y  
pip uninstall btengine -y
pip uninstall BehaviorTreeEngine -y

# Clear pip cache
pip cache purge
```

### Step 2: Install Correct Version
```bash
# From BTAgent directory, install the correct BTEngine
pip install -e BTEngine/BehaviorTreeEngine
```

### Step 3: Verify Installation
```bash
python -c "from behavior_tree_engine.core import Timeout, AsyncAction, BehaviorTree, NodeStatus; print('✅ All features available!'); print('NodeStatus.READY:', NodeStatus.READY)"
```

## 🌐 **For Remote/Separate Computers**

### Method 1: Git Clone + Install
```bash
# Clone the repository
git clone <your-repo-url>
cd BTAgent

# Run the reset script
python reset_btengine.py
```

### Method 2: Package the BTEngine
```bash
# On your working computer, create a wheel
cd BTEngine/BehaviorTreeEngine
python setup.py bdist_wheel

# Copy the wheel file to the other computer and install
pip install dist/behavior_tree_engine-0.1.0-py3-none-any.whl --force-reinstall
```

### Method 3: Direct Requirements Update
Update your `requirements.txt` to point directly to the GitHub repo:
```txt
# Replace this line in requirements.txt:
# -e ./BTEngine

# With this:
git+https://github.com/yourusername/BTEngine.git#subdirectory=BehaviorTreeEngine
```

## 🧪 **Testing Your Installation**

### Quick Test
```python
from behavior_tree_engine.core import NodeStatus, AsyncAction, BehaviorTree, Timeout
print("✅ All imports successful!")
print(f"NodeStatus.READY = {NodeStatus.READY}")
```

### Full Test
```bash
# Run the complex example
python examples/complex_task_agent.py

# Should see: "Task execution completed with status: SUCCESS"
```

## 🔍 **Troubleshooting**

### Problem: Still getting import errors after reset
**Solution**: Check if you have multiple Python environments
```bash
# Check which Python you're using
which python
pip show behavior_tree_engine

# Make sure you're in the right virtual environment
```

### Problem: Permission denied during installation
**Solution**: Use user installation or virtual environment
```bash
pip install -e BTEngine/BehaviorTreeEngine --user
# OR
python -m venv btenv
source btenv/bin/activate  # Linux/Mac
btenv\Scripts\activate     # Windows
pip install -e BTEngine/BehaviorTreeEngine
```

### Problem: Cannot find BTEngine/BehaviorTreeEngine
**Solution**: Make sure you're in the right directory
```bash
# You should see these directories:
ls -la
# Should show: BTEngine/, bt_agent/, examples/

# If not, clone the full repository first
```

## 📋 **Environment Requirements**

- Python 3.8+
- pip (latest version recommended)
- Git (for cloning repositories)

## 🆘 **Still Having Issues?**

1. **Run the diagnostic**: `python check_btengine_version.py`
2. **Check the output** for specific error messages
3. **Try the automated reset**: `python reset_btengine.py`
4. **Verify with a simple test**: `python -c "from behavior_tree_engine.core import Timeout; print('Success!')"`

## ✅ **Success Indicators**

When everything is working correctly, you should see:
- ✅ No import errors for `Timeout`, `AsyncAction`, `BehaviorTree`
- ✅ `NodeStatus.READY` is available
- ✅ `examples/complex_task_agent.py` runs successfully
- ✅ Output shows "Task execution completed with status: SUCCESS" 