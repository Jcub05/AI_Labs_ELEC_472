from collections import deque
import matplotlib.pyplot as plt


def graph(source, target, weights=[], names={}):
    """
    Creates and display a graph.
    
    Parameters:
        source (list): List of source nodes.
        target (list): List of target nodes.
        weights (list): List of edge weights corresponding to source-target pairs.
        names (dict): Dictionary mapping node identifiers to names for labeling.
    
    Returns:
        None
    """
    # Get unique nodes
    nodes = []
    for s, t in zip(source, target):
        if s not in nodes:
            nodes.append(s)
        if t not in nodes:
            nodes.append(t)
    
    # If weights are not provided add None
    if len(weights) == 0:
        weights = [None for _ in range(max(len(source), len(target)))]
    
    # Create a mapping for node levels
    levels = {}
    for s, t in zip(source, target):
        if s not in levels:
            levels[s] = 0
        levels[t] = levels[s] + 1
    
    positions = {}
    max_width = max(levels.values()) + 1
    level_counts = [0] * max_width
    
    # for node in sorted(nodes, key=lambda n: levels.get(n, 0)):
    for node in nodes:
        level = levels.get(node, 0)
        x = (level_counts[level] + 0.5) / (sum(level_counts) + 1 if sum(level_counts) > 0 else 1)
        y = -level
        positions[node] = (x, y)
        level_counts[level] += 1
    
    print(positions)
    
    # Adjust root node to be at the center
    for node, level in levels.items():
        if level == 0:
            positions[node] = (0.5, 0)
    
    # Draw nodes
    plt.figure(figsize=(8, 8))
    for node, (x, y) in positions.items():
        plt.scatter(x, y, s=700, color='skyblue', zorder=2)
        plt.text(x, y, names.get(node, node), fontsize=12, ha='center', va='center', zorder=3)
    
    # Draw edges with weights
    for s, t, w in zip(source, target, weights):
        x1, y1 = positions[s]
        x2, y2 = positions[t]
        plt.plot([x1, x2], [y1, y2], color='black', zorder=1)
        if w is not None:
            plt.text((x1 + x2) / 2, (y1 + y2) / 2, str(w), fontsize=10, color='red', zorder=4)
    
    # Display the graph
    plt.axis('off')
    plt.show()
    return


source = [1, 1, 1, 2, 2, 2, 2, 8] # Source Nodes
target = [3, 4, 2, 6, 5, 7, 8, 9] # Target Nodes
weights = [200, 300, 900, 400, 0, 739, 100, 50] # Edge Weights
names = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'Jacob', 8: 'H', 9: 'I'} #Node Names
graph(source, target, weights, names)