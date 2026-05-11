"""Core package for Cairo Smart City Transportation Optimizer."""

from .data_loader import CairoTransportationData
from .algorithms import (
    dijkstra_shortest_path,
    astar_shortest_path,
    greedy_best_first_path,
    time_dependent_dijkstra,
    kruskal_mst,
    optimize_road_maintenance,
    optimize_traffic_signal,
    optimize_emergency_signal,
    optimize_public_transit,
)
from .traffic_ai import train_congestion_forecaster, predict_congestion
