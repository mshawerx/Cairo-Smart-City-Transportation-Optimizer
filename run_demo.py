from core import (
    CairoTransportationData,
    dijkstra_shortest_path,
    astar_shortest_path,
    greedy_best_first_path,
    time_dependent_dijkstra,
    kruskal_mst,
    optimize_road_maintenance,
    optimize_traffic_signal,
    optimize_emergency_signal,
    optimize_public_transit,
    train_congestion_forecaster,
    predict_congestion,
)


def show_path(data, result):
    names = " -> ".join(data.path_names(result["path"])) if result.get("path") else "No path"
    print(f"{result['algorithm']} [{result.get('weight')}]: cost={result['cost']} visited={result.get('visited_nodes')}\n  {names}")


def main():
    data = CairoTransportationData("data").load()
    g = data.graph
    print(f"Loaded graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} roads")

    source, target = "2", "7"
    show_path(data, dijkstra_shortest_path(g, source, target, weight="distance"))
    show_path(data, astar_shortest_path(g, source, target, weight="travel_time"))
    show_path(data, greedy_best_first_path(g, source, target, weight="distance"))
    show_path(data, time_dependent_dijkstra(g, source, target, hour=8))

    mst = kruskal_mst(g)
    print(f"Kruskal MST: {mst['edge_count']} edges, total_weight={mst['total_weight']}, connected={mst['connected']}")

    maintenance = optimize_road_maintenance(g, budget=1500)
    print(f"Maintenance DP: selected={len(maintenance['selected_roads'])}, used_budget={maintenance['used_budget']}, urgency={maintenance['total_urgency']}")

    signal = optimize_traffic_signal({"north": 50, "south": 30, "east": 70, "west": 20})
    emergency = optimize_emergency_signal({"north": 50, "south": 30, "east": 70, "west": 20}, "east")
    print(f"Signal normal: {signal}")
    print(f"Signal emergency: {emergency}")

    transit = optimize_public_transit(data.bus_routes, max_buses=75)
    print(f"Transit DP: routes={[r['route_id'] for r in transit['selected_routes']]}, buses={transit['used_buses']}, passengers={transit['total_passengers']}")

    model, metrics, _ = train_congestion_forecaster(g, data.traffic_patterns)
    prediction = predict_congestion(model, g, "1-3", 8, data.traffic_patterns)
    print(f"Random Forest: metrics={metrics}, predicted traffic on road 1-3 at 08:00 = {prediction}")


if __name__ == "__main__":
    main()
