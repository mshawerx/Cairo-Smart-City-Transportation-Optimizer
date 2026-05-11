from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx


class CairoTransportationData:
    """Load the project JSON files and build a weighted Cairo road graph.

    Nodes represent neighborhoods/facilities. Edges represent existing and proposed roads.
    Every edge receives multiple weights so each algorithm can optimize a different objective:
    distance, average travel time, time-dependent travel time, construction cost, and congestion.
    """

    PERIODS = ("morning", "afternoon", "evening", "night")

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.neighborhoods: List[Dict[str, Any]] = []
        self.facilities: List[Dict[str, Any]] = []
        self.existing_roads: List[Dict[str, Any]] = []
        self.new_roads: List[Dict[str, Any]] = []
        self.traffic_patterns: List[Dict[str, Any]] = []
        self.metro_lines: List[Dict[str, Any]] = []
        self.bus_routes: List[Dict[str, Any]] = []
        self.public_transport_demand: List[Dict[str, Any]] = []
        self.graph = nx.Graph()

    def load(self) -> "CairoTransportationData":
        self.neighborhoods = self._read_json("neighborhoods.json")
        self.facilities = self._read_json("facilities.json")
        roads = self._read_json("roads.json")
        transport = self._read_json("transport.json")

        self.existing_roads = roads.get("existing_roads", [])
        self.new_roads = roads.get("new_roads", [])
        self.traffic_patterns = transport.get("traffic_patterns", [])
        self.metro_lines = transport.get("metro_lines", [])
        self.bus_routes = transport.get("bus_routes", [])
        self.public_transport_demand = transport.get("public_transport_demand", [])

        self._build_graph()
        return self

    def _read_json(self, filename: str) -> Any:
        path = self.data_dir / filename
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _build_graph(self) -> None:
        self.graph.clear()
        self._add_nodes()
        self._add_existing_roads()
        self._add_proposed_roads()
        self._connect_isolated_facilities()

    def _add_nodes(self) -> None:
        for area in self.neighborhoods:
            node_id = str(area["ID"])
            self.graph.add_node(
                node_id,
                id=node_id,
                name=area["Name"],
                category=area["Type"],
                node_type="neighborhood",
                population=int(area["Population"]),
                pos=(float(area["X"]), float(area["Y"])),
                importance=math.log(max(int(area["Population"]), 1)),
            )

        for facility in self.facilities:
            node_id = str(facility["id"])
            self.graph.add_node(
                node_id,
                id=node_id,
                name=facility["name"],
                category=facility["type"],
                node_type="facility",
                population=0,
                pos=(float(facility["longitude"]), float(facility["latitude"])),
                importance=3.0,
            )

    def _add_existing_roads(self) -> None:
        for road in self.existing_roads:
            u = str(road["from_id"])
            v = str(road["to_id"])
            distance = float(road["distance_km"])
            capacity = max(float(road["capacity"]), 1.0)
            condition = float(road.get("condition", 7))
            traffic = self.find_traffic(u, v)
            self._add_road_edge(
                u=u,
                v=v,
                distance=distance,
                capacity=capacity,
                condition=condition,
                traffic=traffic,
                road_type="existing",
                construction_cost=distance * 25,
            )

    def _add_proposed_roads(self) -> None:
        for road in self.new_roads:
            u = str(road["from"])
            v = str(road["to"])
            distance = float(road["distance"])
            capacity = max(float(road["capacity"]), 1.0)
            traffic = self.find_traffic(u, v)
            self._add_road_edge(
                u=u,
                v=v,
                distance=distance,
                capacity=capacity,
                condition=10.0,
                traffic=traffic,
                road_type="proposed",
                construction_cost=float(road.get("cost", distance * 40)),
            )

    def _add_road_edge(
        self,
        u: str,
        v: str,
        distance: float,
        capacity: float,
        condition: float,
        traffic: Dict[str, float],
        road_type: str,
        construction_cost: float,
    ) -> None:
        avg_traffic = sum(float(traffic[p]) for p in self.PERIODS) / len(self.PERIODS)
        congestion_ratio = avg_traffic / capacity
        # The assignment data does not include speeds, so travel_time is a normalized time score.
        # It increases when traffic/capacity increases and when road condition decreases.
        condition_penalty = 1 + (10 - condition) / 20
        travel_time = distance * (1 + congestion_ratio) * condition_penalty

        road_name = f"{self.graph.nodes[u].get('name', u)} → {self.graph.nodes[v].get('name', v)}"
        attrs: Dict[str, Any] = {
            "road_id": f"{u}-{v}",
            "road_name": road_name,
            "distance": round(distance, 3),
            "capacity": capacity,
            "condition": condition,
            "avg_traffic": round(avg_traffic, 3),
            "congestion": round(congestion_ratio, 4),
            "travel_time": round(travel_time, 3),
            "construction_cost": round(construction_cost, 3),
            "road_type": road_type,
        }
        for period in self.PERIODS:
            period_traffic = float(traffic[period])
            period_ratio = period_traffic / capacity
            attrs[f"{period}_traffic"] = period_traffic
            attrs[f"{period}_travel_time"] = round(distance * (1 + period_ratio) * condition_penalty, 3)

        self.graph.add_edge(u, v, **attrs)


    def _approx_distance_km(self, u: str, v: str) -> float:
        x1, y1 = self.graph.nodes[u].get("pos", (0.0, 0.0))
        x2, y2 = self.graph.nodes[v].get("pos", (0.0, 0.0))
        avg_lat = math.radians((y1 + y2) / 2)
        km_per_lon = 111.32 * max(math.cos(avg_lat), 0.1)
        dx = (x1 - x2) * km_per_lon
        dy = (y1 - y2) * 110.57
        return max(0.1, math.sqrt(dx * dx + dy * dy))

    def _connect_isolated_facilities(self) -> None:
        """Connect facilities that have no roads to the nearest neighborhood.

        The source dataset contains several facilities without road entries.
        A full MST/pathfinding requirement needs a connected graph, so these
        lightweight access roads keep every facility reachable while preserving
        the original roads as `existing` or `proposed`.
        """
        neighborhoods = [node for node, attrs in self.graph.nodes(data=True) if attrs.get("node_type") == "neighborhood"]
        facilities = [node for node, attrs in self.graph.nodes(data=True) if attrs.get("node_type") == "facility"]
        for facility in facilities:
            if self.graph.degree(facility) > 0 or not neighborhoods:
                continue
            nearest = min(neighborhoods, key=lambda node: self._approx_distance_km(facility, node))
            distance = self._approx_distance_km(facility, nearest)
            self._add_road_edge(
                u=facility,
                v=nearest,
                distance=distance,
                capacity=1800.0,
                condition=8.0,
                traffic={"morning": 900.0, "afternoon": 700.0, "evening": 850.0, "night": 350.0},
                road_type="access",
                construction_cost=distance * 30,
            )

    def find_traffic(self, u: str, v: str) -> Dict[str, float]:
        key1 = f"{u}-{v}"
        key2 = f"{v}-{u}"
        for pattern in self.traffic_patterns:
            if pattern.get("road") in {key1, key2}:
                return {period: float(pattern.get(period, 0)) for period in self.PERIODS}
        return {"morning": 1000.0, "afternoon": 800.0, "evening": 900.0, "night": 500.0}

    def node_label(self, node_id: str) -> str:
        node = self.graph.nodes[str(node_id)]
        return f"{node.get('name', node_id)} ({node_id})"

    def path_names(self, path: Iterable[str] | None) -> List[str]:
        if not path:
            return []
        return [self.node_label(str(node)) for node in path]

    @staticmethod
    def period_from_hour(hour: int) -> str:
        hour = int(hour) % 24
        if 6 <= hour <= 11:
            return "morning"
        if 12 <= hour <= 15:
            return "afternoon"
        if 16 <= hour <= 21:
            return "evening"
        return "night"

    def locations_for_ui(self) -> List[Tuple[str, str]]:
        locations = [(node, self.node_label(node)) for node in self.graph.nodes]
        return sorted(locations, key=lambda item: item[1])
