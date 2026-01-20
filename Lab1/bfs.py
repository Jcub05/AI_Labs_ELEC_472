from collections import deque


def bfs(source, target, start_node, target_node):
    """
    Breadth First Search (BFS).
    
    Args:
        source (list): List of source nodes for edges.
        target (list): List of target nodes for edges.
        start_node (int): The starting node for BFS.
        target_node (int): The node to be searched for.
    
    Returns:
        list: A list representing the path of visited nodes, or -1 if target is not found.
    """
    # Initialize visited set, queue, and node_list
    node_list = []
    visited = set()
    queue = deque()
    
    # Add starting node to visited set and queue
    visited.add(start_node)
    queue.append(start_node)
    
    while queue:
        # Dequeue the first node
        current_node = queue.popleft()
        
        # Add current node to the visited list
        node_list.append(current_node)
        
        # If target node is found, return the node list
        if current_node == target_node:
            return node_list
        
        # Get all children of the current node
        children = [target[i] for i in range(len(source)) if source[i] == current_node]
        
        # Add unvisited children to the queue
        for child in children:
            if child not in visited:
                visited.add(child)
                queue.append(child)
    
    # If target node is not found, return -1
    print("Target not found!")
    return -1