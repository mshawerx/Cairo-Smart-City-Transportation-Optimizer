from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

PathResult = Dict[str, Any]


def _safe_node(graph: nx.Graph, node: Any) -> str:
    node = str(node)
    if node not in graph:
        raise ValueError(f"Node {node!r} does not exist in the graph.")
    return node


def _reconstruct_path(parent: Dict[str, Optional[str]], target: str) -> List[str]:
    path: List[str] = []
    current: Optional[str] = target
    while current is not None:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    return path


def _path_cost(graph: nx.Graph, path: List[str], weight: str) -> float:
    if not path or len(path) == 1:
        return 0.0
    return round(sum(float(graph[u][v].get(weight, math.inf)) for u, v in zip(path[:-1], path[1:])), 3)


def _road_name(graph: nx.Graph, u: str, v: str) -> str:
    """Human-readable road label used in tables and map hovers."""
    from_name = graph.nodes[str(u)].get("name", str(u))
    to_name = graph.nodes[str(v)].get("name", str(v))
    return f"{from_name} → {to_name}"


def _straight_line_km(graph: nx.Graph, u: str, v: str) -> float:
    """Approximate straight-line distance between two longitude/latitude points in km."""
    x1, y1 = graph.nodes[u].get("pos", (0.0, 0.0))
    x2, y2 = graph.nodes[v].get("pos", (0.0, 0.0))
    avg_lat = math.radians((y1 + y2) / 2)
    km_per_lon = 111.32 * max(math.cos(avg_lat), 0.1)
    dx = (x1 - x2) * km_per_lon
    dy = (y1 - y2) * 110.57
    return math.sqrt(dx * dx + dy * dy)


def dijkstra_shortest_path(graph: nx.Graph, source: Any, target: Any, weight: str = "distance") -> PathResult:
    """Classic Dijkstra: guaranteed optimal path for non-negative edge weights."""
    source = _safe_node(graph, source)
    target = _safe_node(graph, target)
    if source == target:
        return {"algorithm": "Dijkstra", "path": [source], "cost": 0.0, "visited_nodes": 1, "weight": weight}

    distances = {node: math.inf for node in graph.nodes}
    parent: Dict[str, Optional[str]] = {source: None}
    distances[source] = 0.0
    visited = set()
    heap: List[Tuple[float, str]] = [(0.0, source)]

    while heap:
        current_cost, current = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)
        if current == target:
            break
        for neighbor in graph.neighbors(current):
            edge_weight = float(graph[current][neighbor].get(weight, math.inf))
            if edge_weight < 0:
                raise ValueError("Dijkstra cannot be used with negative edge weights.")
            new_cost = current_cost + edge_weight
            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                parent[neighbor] = current
                heapq.heappush(heap, (new_cost, neighbor))

    if distances[target] == math.inf:
        return {"algorithm": "Dijkstra", "path": None, "cost": math.inf, "visited_nodes": len(visited), "weight": weight}

    path = _reconstruct_path(parent, target)
    return {"algorithm": "Dijkstra", "path": path, "cost": round(distances[target], 3), "visited_nodes": len(visited), "weight": weight}


def astar_shortest_path(graph: nx.Graph, source: Any, target: Any, weight: str = "distance") -> PathResult:
    """A* search: uses a geographic heuristic to guide the search toward the target."""
    source = _safe_node(graph, source)
    target = _safe_node(graph, target)

    def heuristic(node: str) -> float:
        # travel_time is always >= distance in this project, so straight-line distance remains safe.
        if weight in {"distance", "travel_time"} or weight.endswith("_travel_time"):
            return _straight_line_km(graph, node, target)
        return 0.0

    open_heap: List[Tuple[float, float, str]] = [(heuristic(source), 0.0, source)]
    g_score = {node: math.inf for node in graph.nodes}
    g_score[source] = 0.0
    parent: Dict[str, Optional[str]] = {source: None}
    visited = set()

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == target:
            path = _reconstruct_path(parent, target)
            return {"algorithm": "A*", "path": path, "cost": round(current_g, 3), "visited_nodes": len(visited), "weight": weight}
        for neighbor in graph.neighbors(current):
            edge_weight = float(graph[current][neighbor].get(weight, math.inf))
            tentative_g = current_g + edge_weight
            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                parent[neighbor] = current
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    return {"algorithm": "A*", "path": None, "cost": math.inf, "visited_nodes": len(visited), "weight": weight}


def greedy_best_first_path(graph: nx.Graph, source: Any, target: Any, weight: str = "distance") -> PathResult:
    """Greedy best-first search.

    It expands the most promising frontier node using a local priority instead of the
    full accumulated cost. This makes it different from Dijkstra/A*: it usually finds
    a valid route quickly, but it is not guaranteed to be the cheapest route.
    """
    source = _safe_node(graph, source)
    target = _safe_node(graph, target)
    parent: Dict[str, Optional[str]] = {source: None}
    visited = set()
    # priority = straight-line closeness + immediate edge cost; no accumulated path cost.
    heap: List[Tuple[float, str]] = [(_straight_line_km(graph, source, target), source)]

    while heap:
        _, current = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)
        if current == target:
            path = _reconstruct_path(parent, target)
            return {"algorithm": "Greedy", "path": path, "cost": _path_cost(graph, path, weight), "visited_nodes": len(visited), "weight": weight, "optimal": False}
        for neighbor in graph.neighbors(current):
            if neighbor in visited:
                continue
            if neighbor not in parent:
                parent[neighbor] = current
            edge_weight = float(graph[current][neighbor].get(weight, math.inf))
            priority = _straight_line_km(graph, neighbor, target) + 0.05 * edge_weight
            heapq.heappush(heap, (priority, neighbor))

    return {"algorithm": "Greedy", "path": None, "cost": math.inf, "visited_nodes": len(visited), "weight": weight, "optimal": False}

def time_dependent_dijkstra(graph: nx.Graph, source: Any, target: Any, hour: int) -> PathResult:
    """Dijkstra where the edge weight changes according to the selected time of day."""
    hour = int(hour) % 24
    if 6 <= hour <= 11:
        period = "morning"
    elif 12 <= hour <= 15:
        period = "afternoon"
    elif 16 <= hour <= 21:
        period = "evening"
    else:
        period = "night"
    weight = f"{period}_travel_time"
    result = dijkstra_shortest_path(graph, source, target, weight=weight)
    result["algorithm"] = "Time-dependent Dijkstra"
    result["period"] = period
    result["hour"] = hour
    return result


def kruskal_mst(graph: nx.Graph, weight: str = "construction_cost") -> Dict[str, Any]:
    """Minimum Spanning Tree using Kruskal's union-find logic."""
    parent = {node: node for node in graph.nodes}
    rank = {node: 0 for node in graph.nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> bool:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return False
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1
        return True

    sorted_edges = sorted(graph.edges(data=True), key=lambda item: float(item[2].get(weight, item[2].get("distance", 0))))
    mst_edges = []
    total_cost = 0.0
    mst_graph = nx.Graph()
    mst_graph.add_nodes_from(graph.nodes(data=True))

    for u, v, data in sorted_edges:
        edge_weight = float(data.get(weight, data.get("distance", 0)))
        if union(u, v):
            attrs = dict(data)
            attrs["mst_weight"] = edge_weight
            mst_graph.add_edge(u, v, **attrs)
            mst_edges.append({
                "road_id": f"{u}-{v}",
                "road_name": data.get("road_name", _road_name(graph, u, v)),
                "from": u,
                "to": v,
                "weight": round(edge_weight, 3),
                "distance": data.get("distance"),
                "road_type": data.get("road_type", "road"),
            })
            total_cost += edge_weight
            if len(mst_edges) == graph.number_of_nodes() - 1:
                break

    return {
        "algorithm": "Kruskal MST",
        "graph": mst_graph,
        "edges": mst_edges,
        "total_weight": round(total_cost, 3),
        "edge_count": len(mst_edges),
        "node_count": graph.number_of_nodes(),
        "connected": nx.is_connected(mst_graph) if mst_graph.number_of_nodes() > 0 else False,
        "weight": weight,
    }


def optimize_road_maintenance(graph: nx.Graph, budget: int) -> Dict[str, Any]:
    """0/1 Knapsack: select roads to repair under a fixed budget."""
    budget = max(0, int(budget))
    roads = []
    for u, v, data in graph.edges(data=True):
        if data.get("road_type") != "existing":
            continue
        condition = float(data.get("condition", 10))
        distance = float(data.get("distance", 1))
        avg_traffic = float(data.get("avg_traffic", 0))
        repair_cost = max(50, int(round((10 - condition) * 120 + distance * 12)))
        urgency = max(1, int(round((10 - condition) * 15 + avg_traffic / 120)))
        roads.append({
            "road_id": f"{u}-{v}",
            "road_name": data.get("road_name", _road_name(graph, u, v)),
            "from": u,
            "to": v,
            "repair_cost": repair_cost,
            "urgency": urgency,
            "condition": condition,
        })

    n = len(roads)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]
    keep = [[False] * (budget + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        cost = roads[i - 1]["repair_cost"]
        urgency = roads[i - 1]["urgency"]
        for w in range(budget + 1):
            skip = dp[i - 1][w]
            take = -1
            if cost <= w:
                take = dp[i - 1][w - cost] + urgency
            if take > skip:
                dp[i][w] = take
                keep[i][w] = True
            else:
                dp[i][w] = skip

    selected = []
    w = budget
    for i in range(n, 0, -1):
        if keep[i][w]:
            road = roads[i - 1]
            selected.append(road)
            w -= road["repair_cost"]
    selected.reverse()
    return {"algorithm": "Dynamic Programming / 0-1 Knapsack", "budget": budget, "selected_roads": selected, "total_urgency": dp[n][budget], "used_budget": sum(r["repair_cost"] for r in selected)}


def optimize_traffic_signal(car_counts: Dict[str, int], cycle_seconds: int = 100, min_green: int = 5) -> Dict[str, float]:
    """Allocate green-light seconds proportionally to traffic demand."""
    if not car_counts:
        return {}
    cycle_seconds = max(cycle_seconds, min_green * len(car_counts))
    total = sum(max(0, int(count)) for count in car_counts.values())
    if total == 0:
        equal = round(cycle_seconds / len(car_counts), 2)
        return {direction: equal for direction in car_counts}

    raw = {direction: max(min_green, (max(0, int(count)) / total) * cycle_seconds) for direction, count in car_counts.items()}
    raw_total = sum(raw.values())
    return {direction: round(seconds * cycle_seconds / raw_total, 2) for direction, seconds in raw.items()}


def optimize_emergency_signal(car_counts: Dict[str, int], emergency_direction: str, cycle_seconds: int = 100) -> Dict[str, float]:
    """Give emergency direction a priority green time and distribute the rest fairly."""
    if emergency_direction not in car_counts:
        raise ValueError("Emergency direction must exist in car_counts.")
    base = optimize_traffic_signal(car_counts, cycle_seconds=cycle_seconds)
    priority = min(cycle_seconds * 0.65, max(base[emergency_direction] * 1.7, cycle_seconds * 0.45))
    remaining = cycle_seconds - priority
    other_counts = {d: c for d, c in car_counts.items() if d != emergency_direction}
    other_allocation = optimize_traffic_signal(other_counts, cycle_seconds=int(round(remaining))) if other_counts else {}
    result = {emergency_direction: round(priority, 2)}
    result.update(other_allocation)
    return result


def optimize_public_transit(bus_routes: List[Dict[str, Any]], max_buses: int) -> Dict[str, Any]:
    """0/1 Knapsack for public transit: maximize passengers using limited buses."""
    max_buses = max(0, int(max_buses))
    routes = []
    for route in bus_routes:
        buses = int(route.get("buses_assigned", 0))
        passengers = int(route.get("daily_passengers", 0))
        if buses > 0:
            routes.append({
                "route_id": route.get("route_id", "route"),
                "buses_required": buses,
                "daily_passengers": passengers,
                "stops": [str(stop) for stop in route.get("stops", [])],
            })

    n = len(routes)
    dp = [[0] * (max_buses + 1) for _ in range(n + 1)]
    keep = [[False] * (max_buses + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        buses = routes[i - 1]["buses_required"]
        passengers = routes[i - 1]["daily_passengers"]
        for b in range(max_buses + 1):
            skip = dp[i - 1][b]
            take = -1
            if buses <= b:
                take = dp[i - 1][b - buses] + passengers
            if take > skip:
                dp[i][b] = take
                keep[i][b] = True
            else:
                dp[i][b] = skip

    selected = []
    b = max_buses
    for i in range(n, 0, -1):
        if keep[i][b]:
            route = routes[i - 1]
            selected.append(route)
            b -= route["buses_required"]
    selected.reverse()

    return {
        "algorithm": "Dynamic Programming / Transit Knapsack",
        "max_buses": max_buses,
        "selected_routes": selected,
        "used_buses": sum(route["buses_required"] for route in selected),
        "total_passengers": dp[n][max_buses],
    }
