import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (
    CairoTransportationData,
    astar_shortest_path,
    dijkstra_shortest_path,
    greedy_best_first_path,
    kruskal_mst,
    optimize_public_transit,
    optimize_road_maintenance,
    optimize_traffic_signal,
    time_dependent_dijkstra,
    train_congestion_forecaster,
)


def test_project_algorithms_run():
    data = CairoTransportationData("data").load()
    graph = data.graph
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0

    dijkstra = dijkstra_shortest_path(graph, "1", "13", weight="distance")
    astar = astar_shortest_path(graph, "1", "13", weight="travel_time")
    greedy = greedy_best_first_path(graph, "1", "13", weight="distance")
    time_dep = time_dependent_dijkstra(graph, "1", "13", hour=8)

    assert dijkstra["path"]
    assert astar["path"]
    assert greedy["path"]
    assert time_dep["path"]
    assert dijkstra["cost"] > 0
    assert time_dep["period"] == "morning"

    mst = kruskal_mst(graph)
    assert mst["edge_count"] == graph.number_of_nodes() - 1
    assert mst["connected"] is True

    maintenance = optimize_road_maintenance(graph, 1500)
    assert maintenance["used_budget"] <= 1500
    assert maintenance["selected_roads"]
    assert "road_name" in maintenance["selected_roads"][0]
    assert graph.nodes["10"]["pos"] != graph.nodes["F3"]["pos"]  # Dokki no longer overlaps Cairo University on the map.

    transit = optimize_public_transit(data.bus_routes, 75)
    assert transit["used_buses"] <= 75

    signal = optimize_traffic_signal({"north": 50, "south": 30, "east": 70, "west": 20})
    assert round(sum(signal.values()), 1) == 100.0

    model, metrics, df = train_congestion_forecaster(graph, data.traffic_patterns)
    assert metrics["training_rows"] == len(df)
