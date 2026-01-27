def dls(source, target, start_node, target_node, depth):
    """
    Depth-Limited Search (DLS)
    5
    Args:
    source (list): List of source nodes for edges.
    target (list): List of target nodes for edges.
    start_node (int): The starting node for DFS.
    target_node (int): The node to be searched for.
    depth (int): Depth limit
    Returns:
    tuple: (node_list, result)
    node_list is a list of nodes in the order they are visited (expanded) by
    the algorithm, not necessarily a single root-to-goal path.
    result is 1 if target is found or 0 when not found.
    """
    # Initialize visited set, stack, and node_list
    node_list = []
    visited = set()
    stack = []
    
    # Add starting node to stack: (node, current_depth)
    stack.append((start_node, 0))
    
    while stack:
        # Pop the last node (LIFO)
        current_node, current_depth = stack.pop()
        
        # Skip if already visited
        if current_node in visited:
            continue
            
        # Mark as visited and add to node_list
        visited.add(current_node)
        node_list.append(current_node)
        
        # If target node is found, return success
        if current_node == target_node:
            return (node_list, 1)
        
        # Only expand if depth limit not reached
        if current_depth < depth:
            # Get all children of the current node
            children = [target[i] for i in range(len(source)) if source[i] == current_node]
            
            # Add unvisited children to the stack
            for child in children:
                if child not in visited:
                    stack.append((child, current_depth + 1))
    
    # Target node not found within depth limit
    return (node_list, 0)


def ids(source, target, start_node, target_node, max_depth=100):
    # Iteratively increase depth limit
    for depth in range(max_depth):
        node_list, result = dls(source, target, start_node, target_node, depth)
        
        # If target found, return the node list
        if result == 1:
            return node_list
    
    # Target not found within max_depth
    print("Target not found!")
    return -1


source = [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 5, 15, 15, 15]
target = [3, 5, 4, 2, 6, 10, 7, 9, 8, 14, 11, 12, 13, 15, 17, 16, 18]

# Testing DLS at depth = 2
node_list2, result = dls(source, target, 1, 15, 2)
print("Node List 2 (DLS at depth=2):", node_list2)
print("Result:", result)

# Testing IDS
node_list3 = ids(source, target, 1, 15)
print("Node List 3 (IDS):", node_list3)