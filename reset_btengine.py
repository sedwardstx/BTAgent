#!/usr/bin/env python3
"""
BTEngine Environment Reset Script
Automatically uninstalls old versions and installs the latest BTEngine.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors gracefully."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  {description} had issues (this may be normal):")
        if e.stdout.strip():
            print(f"   Stdout: {e.stdout.strip()}")
        if e.stderr.strip():
            print(f"   Stderr: {e.stderr.strip()}")
        return False

def find_btengine_path():
    """Find the BTEngine installation path."""
    possible_paths = [
        "BTEngine/BehaviorTreeEngine",
        "../BTEngine/BehaviorTreeEngine", 
        "../../BTEngine/BehaviorTreeEngine",
        "./BTEngine/BehaviorTreeEngine"
    ]
    
    for path in possible_paths:
        if Path(path).exists() and Path(path, "setup.py").exists():
            return str(Path(path).resolve())
    
    return None

def reset_btengine_environment():
    """Reset the BTEngine environment completely."""
    print("🚀 BTEngine Environment Reset")
    print("=" * 50)
    
    # Step 1: Uninstall all possible old versions
    old_packages = [
        "behavior-tree-engine",
        "behavior_tree_engine", 
        "btengine",
        "BehaviorTreeEngine"
    ]
    
    for package in old_packages:
        run_command(f"pip uninstall {package} -y", f"Uninstalling {package}")
    
    # Step 2: Clear pip cache
    run_command("pip cache purge", "Clearing pip cache")
    
    # Step 3: Find BTEngine path
    btengine_path = find_btengine_path()
    if not btengine_path:
        print("❌ Could not find BTEngine/BehaviorTreeEngine directory!")
        print("📁 Please ensure you're running this from the BTAgent directory")
        print("📁 Or that BTEngine/BehaviorTreeEngine exists relative to this script")
        return False
    
    print(f"📍 Found BTEngine at: {btengine_path}")
    
    # Step 4: Install the new version
    install_cmd = f"pip install -e \"{btengine_path}\""
    if not run_command(install_cmd, f"Installing BTEngine from {btengine_path}"):
        print("❌ Failed to install BTEngine!")
        return False
    
    # Step 5: Verify installation
    print("\n🧪 Verifying installation...")
    try:
        from behavior_tree_engine.core import NodeStatus, AsyncAction, BehaviorTree, Timeout
        NodeStatus.READY  # This will fail on old versions
        print("✅ All new features verified successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except AttributeError as e:
        print(f"❌ Missing features (old version still installed): {e}")
        return False

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--check-only":
        # Just run the checker
        from check_btengine_version import check_btengine_installation
        return check_btengine_installation()
    
    success = reset_btengine_environment()
    
    if success:
        print("\n🎉 BTEngine environment reset completed successfully!")
        print("🚀 You can now run your BTAgent examples.")
    else:
        print("\n❌ BTEngine environment reset failed!")
        print("🔧 Please check the error messages above and try manual installation.")
    
    return success

if __name__ == "__main__":
    main() 