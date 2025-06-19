from typing import Dict, Any, Optional
from btengine.base import BTNode, NodeStatus

def print_tree(node: BTNode, level: int = 0) -> None:
    """Print a visual representation of the behavior tree."""
    indent = "  " * level
    print(f"{indent}└─ {node.name} [{node.status.value}]")
    if hasattr(node, "children"):
        for child in node.children:
            print_tree(child, level + 1)
    elif hasattr(node, "child"):
        print_tree(node.child, level + 1)

def find_node_by_name(root: BTNode, name: str) -> Optional[BTNode]:
    """Find a node in the tree by its name."""
    if root.name == name:
        return root
    
    if hasattr(root, "children"):
        for child in root.children:
            result = find_node_by_name(child, name)
            if result:
                return result
    elif hasattr(root, "child"):
        return find_node_by_name(root.child, name)
    
    return None

def get_tree_status(root: BTNode) -> Dict[str, Any]:
    """Get the status of all nodes in the tree."""
    status = {
        "name": root.name,
        "status": root.status.value
    }
    
    if hasattr(root, "children"):
        status["children"] = [get_tree_status(child) for child in root.children]
    elif hasattr(root, "child"):
        status["child"] = get_tree_status(root.child)
    
    return status

def reset_subtree(node: BTNode) -> None:
    """Reset a subtree starting from the given node."""
    node.reset()
    if hasattr(node, "children"):
        for child in node.children:
            reset_subtree(child)
    elif hasattr(node, "child"):
        reset_subtree(node.child) 