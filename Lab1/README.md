# Lab 1: Graph Search Algorithms

## Overview
This lab implements fundamental graph search algorithms used in artificial intelligence and pathfinding.

## Files
- **`bfs.py`** - Breadth-First Search (BFS) implementation
- **`dfs.py`** - Depth-First Search (DFS) implementation

## Algorithms

### Breadth-First Search (BFS)
Explores nodes level by level, guaranteeing the shortest path in unweighted graphs. Uses a queue (FIFO) data structure.

### Depth-First Search (DFS)
Explores as far as possible along each branch before backtracking. Uses a stack (LIFO) data structure.

## Usage
Both algorithms take:
- `source` - List of source nodes for edges
- `target` - List of target nodes for edges
- `start_node` - Starting node for search
- `target_node` - Node to find

Returns a list of visited nodes representing the path, or -1 if target is not found.
