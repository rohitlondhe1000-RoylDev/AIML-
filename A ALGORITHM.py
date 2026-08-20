import heapq

def a_star(graph, heuristic, start, goal):

    open_list = [(0 + heuristic[start], start)]

    g_cost = {node: float("inf") for node in graph}
    g_cost[start] = 0

    parent = {node: None for node in graph}

    closed_set = set()

    while open_list:
        
        current_f, current = heapq.heappop(open_list)
 
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path, g_cost[goal]

        if current in closed_set:
            continue
        closed_set.add(current)

        for neighbor, edge_cost in graph[current]:
            if neighbor in closed_set:
                continue

            tentative_g = g_cost[current] + edge_cost

            if tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                parent[neighbor] = current
                f_cost = tentative_g + heuristic[neighbor]
                heapq.heappush(open_list, (f_cost, neighbor))

    return None, float("inf")

graph = {
    "A": [("B", 1), ("C", 4)],
    "B": [("A", 1), ("C", 2), ("D", 5)],
    "C": [("A", 4), ("B", 2), ("D", 1)],
    "D": [("B", 5), ("C", 1), ("E", 3)],
    "E": [("D", 3)],
}
heuristic = {"A": 7, "B": 6, "C": 2, "D": 1, "E": 0}

if __name__ == "__main__":
    start_node = "A"
    goal_node = "E"

    path, cost = a_star(graph, heuristic, start_node, goal_node)

    if path:
        print("Optimal Path:", " -> ".join(path))
        print("Total Path Cost:", cost)
    else:
        print("No path found.")

#Optimal Path: A -> C -> D -> E
#Total Path Cost: 8