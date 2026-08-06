from collections import deque

graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F","G"],
    "D": ["H"],
    "F": ["I","J"],
}


def bfs(graph, start_node):
    """Performs Breadth-First Search (BFS) starting from start_node."""
    visited = set()  
    queue = deque([start_node])  

    visited.add(start_node)

    print("Following is the Breadth-First Search:")

    while queue:
        current_node = queue.popleft()
        print(current_node, end=" ")

        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    print()  


if __name__ == "__main__":
    bfs(graph, "A")


    #Following is the Breadth-First Search:
    #A B C D E F G H I J 