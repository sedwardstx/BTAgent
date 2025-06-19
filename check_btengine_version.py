#!/usr/bin/env python3
"""
BTEngine Version Checker
Run this script to diagnose BTEngine installation issues.
"""

def check_btengine_installation():
    print("🔍 BTEngine Installation Diagnostic")
    print("=" * 50)
    
    # Check if package is installed
    try:
        import behavior_tree_engine
        print("✅ behavior_tree_engine package found")
        print(f"📍 Package location: {behavior_tree_engine.__file__}")
    except ImportError as e:
        print(f"❌ behavior_tree_engine not found: {e}")
        return False
    
    # Check package version
    try:
        version = getattr(behavior_tree_engine, '__version__', 'Unknown')
        print(f"📦 Package version: {version}")
    except:
        print("⚠️  Version information not available")
    
    # Check core module
    try:
        from behavior_tree_engine import core
        print("✅ behavior_tree_engine.core module found")
        print(f"📍 Core module location: {core.__file__}")
    except ImportError as e:
        print(f"❌ behavior_tree_engine.core not found: {e}")
        return False
    
    # Check for new features
    print("\n🧪 Feature Availability Check:")
    
    features = {
        'NodeStatus.READY': ('NodeStatus', 'READY'),
        'AsyncAction': ('AsyncAction', None),
        'BehaviorTree': ('BehaviorTree', None),
        'Timeout': ('Timeout', None),
        'Repeater': ('Repeater', None),
        'RetryUntilSuccess': ('RetryUntilSuccess', None)
    }
    
    for feature_name, (class_name, attr_name) in features.items():
        try:
            obj = getattr(core, class_name)
            if attr_name:
                getattr(obj, attr_name)
            print(f"✅ {feature_name}")
        except AttributeError:
            print(f"❌ {feature_name} - MISSING (Old version)")
        except Exception as e:
            print(f"⚠️  {feature_name} - Error: {e}")
    
    # Check if this is the old or new version
    try:
        from behavior_tree_engine.core import Timeout, AsyncAction, BehaviorTree
        from behavior_tree_engine.core import NodeStatus
        NodeStatus.READY
        print("\n🎉 NEW VERSION DETECTED - All features available!")
        return True
    except (ImportError, AttributeError):
        print("\n⚠️  OLD VERSION DETECTED - Missing new features!")
        print("\n🔧 SOLUTION NEEDED:")
        print("1. Uninstall old version: pip uninstall behavior-tree-engine -y")
        print("2. Install from BTEngine/BehaviorTreeEngine: pip install -e BTEngine/BehaviorTreeEngine")
        return False

if __name__ == "__main__":
    check_btengine_installation() 