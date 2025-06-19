@echo off
echo 🚀 BTEngine Environment Reset (Windows)
echo ==================================================

echo 🔄 Step 1: Uninstalling old BTEngine versions...
pip uninstall behavior-tree-engine -y >nul 2>&1
pip uninstall behavior_tree_engine -y >nul 2>&1
pip uninstall btengine -y >nul 2>&1
pip uninstall BehaviorTreeEngine -y >nul 2>&1

echo 🔄 Step 2: Clearing pip cache...
pip cache purge >nul 2>&1

echo 🔄 Step 3: Installing new BTEngine...
if exist "BTEngine\BehaviorTreeEngine\setup.py" (
    pip install -e BTEngine\BehaviorTreeEngine
    if %ERRORLEVEL% EQU 0 (
        echo ✅ BTEngine installation completed successfully!
    ) else (
        echo ❌ BTEngine installation failed!
        exit /b 1
    )
) else (
    echo ❌ Could not find BTEngine\BehaviorTreeEngine directory!
    echo 📁 Please ensure you're running this from the BTAgent directory
    exit /b 1
)

echo 🧪 Step 4: Verifying installation...
python -c "from behavior_tree_engine.core import Timeout, AsyncAction, BehaviorTree, NodeStatus; NodeStatus.READY; print('✅ All features verified successfully!')" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo 🎉 BTEngine environment reset completed successfully!
    echo 🚀 You can now run your BTAgent examples.
) else (
    echo ❌ Verification failed - some features may be missing
    echo 🔧 Try running: python reset_btengine.py
    exit /b 1
)

pause 