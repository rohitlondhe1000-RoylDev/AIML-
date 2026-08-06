
alphabet_graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F","G"],
    "D": ["H"],
    "F": ["I","J"],
}


def dfs_iterative(graph, start_node):
    """Performs Depth-First Search (DFS) iteratively using a Stack."""
    visited = set()
    stack = [start_node]  

    print("Following is the Depth-First Search traversal (Iterative):")

    while stack:
        current_node = stack.pop()  

        if current_node not in visited:
            print(current_node, end=" ")
            visited.add(current_node)

            
            for neighbor in reversed(graph.get(current_node, [])):
                if neighbor not in visited:
                    stack.append(neighbor)
    print()  


if __name__ == "__main__":
    dfs_iterative(alphabet_graph, "A")

    #Following is the Depth-First Search traversal (Iterative):
    #A B D H E C F I J G 