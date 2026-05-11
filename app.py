from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import (
    CairoTransportationData,
    astar_shortest_path,
    dijkstra_shortest_path,
    greedy_best_first_path,
    kruskal_mst,
    optimize_emergency_signal,
    optimize_public_transit,
    optimize_road_maintenance,
    optimize_traffic_signal,
    predict_congestion,
    time_dependent_dijkstra,
    train_congestion_forecaster,
)

st.set_page_config(page_title="Cairo Smart City Transportation Optimizer", layout="wide")


@st.cache_resource
def load_project_data():
    return CairoTransportationData("data").load()


@st.cache_resource
def train_ai(_graph, traffic_patterns):
    return train_congestion_forecaster(_graph, traffic_patterns)


def label_options(data):
    return {label: node for node, label in data.locations_for_ui()}


def path_to_text(data, path):
    return " → ".join(data.path_names(path)) if path else "No path found"


def road_label(data, u, v):
    return f"{data.node_label(str(u))} → {data.node_label(str(v))}"


def prepare_road_dataframe(rows, data):
    """Make road result tables readable by adding road names beside road IDs."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "road_name" not in df.columns and {"from", "to"}.issubset(df.columns):
        df.insert(1, "road_name", [road_label(data, row["from"], row["to"]) for _, row in df.iterrows()])
    preferred = ["road_id", "road_name", "from", "to", "repair_cost", "urgency", "condition", "weight", "distance", "road_type"]
    ordered = [col for col in preferred if col in df.columns] + [col for col in df.columns if col not in preferred]
    return df[ordered]


def transport_routes_dataframe(transit, data):
    df = pd.DataFrame(transit["selected_routes"])
    if df.empty:
        return df
    if "stops" in df.columns:
        df["stop_names"] = df["stops"].apply(lambda stops: " → ".join(data.path_names(stops)))
    preferred = ["route_id", "buses_required", "daily_passengers", "stops", "stop_names"]
    ordered = [col for col in preferred if col in df.columns] + [col for col in df.columns if col not in preferred]
    return df[ordered]


def _node_lon_lat(graph, node):
    lon, lat = graph.nodes[str(node)]["pos"]
    return float(lon), float(lat)


def _path_coordinates(graph, path):
    lons, lats = [], []
    for node in path:
        lon, lat = _node_lon_lat(graph, node)
        lons.append(lon)
        lats.append(lat)
    return lons, lats


def _node_hover(attrs, node_id):
    return (
        f"<b>{attrs.get('name', node_id)} ({node_id})</b>"
        f"<br>Type: {attrs.get('node_type')}"
        f"<br>Category: {attrs.get('category')}"
        f"<br>Lat/Lon: {attrs.get('pos')[1]:.5f}, {attrs.get('pos')[0]:.5f}"
    )


def draw_network(data, highlighted_path=None, mst_graph=None):
    """Draw the Cairo network on a real OpenStreetMap/GPS-style basemap.

    The previous version used a plain coordinate chart. This version keeps the
    same algorithms and results, but visualizes them on real street-map tiles
    and adds public-transport layers: bus routes, bus stops, and metro lines.
    """
    graph = data.graph
    fig = go.Figure()

    # Real road network edges from the project data.
    road_styles = {
        "existing": {"name": "Existing roads", "color": "rgba(180, 180, 180, 0.72)", "width": 2.2},
        "proposed": {"name": "Proposed roads", "color": "rgba(255, 193, 7, 0.78)", "width": 2.0},
        "access": {"name": "Facility access roads", "color": "rgba(156, 204, 101, 0.72)", "width": 1.8},
    }
    shown_road_types = set()
    for u, v, edge_data in graph.edges(data=True):
        road_type = edge_data.get("road_type", "existing")
        style = road_styles.get(road_type, road_styles["existing"])
        lon0, lat0 = _node_lon_lat(graph, u)
        lon1, lat1 = _node_lon_lat(graph, v)
        road_name = edge_data.get("road_name", road_label(data, u, v))
        hover = (
            f"<b>{road_name}</b>"
            f"<br>Road ID: {u}-{v}"
            f"<br>Type: {road_type}"
            f"<br>Distance: {edge_data.get('distance')} km"
            f"<br>Travel time score: {edge_data.get('travel_time')}"
            f"<br>Capacity: {edge_data.get('capacity')}"
            f"<br>Congestion: {edge_data.get('congestion')}"
        )
        fig.add_trace(go.Scattermapbox(
            lon=[lon0, lon1],
            lat=[lat0, lat1],
            mode="lines",
            line=dict(width=style["width"], color=style["color"]),
            hoverinfo="text",
            text=hover,
            name=style["name"],
            legendgroup=road_type,
            showlegend=road_type not in shown_road_types,
        ))
        shown_road_types.add(road_type)

    # Metro lines from the assignment transport data.
    metro_colors = ["#E53935", "#43A047", "#1E88E5", "#8E24AA"]
    for idx, line in enumerate(data.metro_lines):
        stations = [str(stop) for stop in line.get("stations", []) if str(stop) in graph.nodes]
        if len(stations) < 2:
            continue
        lons, lats = _path_coordinates(graph, stations)
        fig.add_trace(go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode="lines+markers",
            line=dict(width=4.2, color=metro_colors[idx % len(metro_colors)]),
            marker=dict(size=8, color=metro_colors[idx % len(metro_colors)]),
            name=f"Metro {line.get('line_id')}",
            legendgroup="metro",
            hoverinfo="text",
            text=f"<b>{line.get('name')}</b><br>Daily passengers: {line.get('daily_passengers'):,}",
        ))

    # Bus routes and bus-stop markers.
    stop_to_routes = {}
    for route in data.bus_routes:
        route_id = route.get("route_id")
        stops = [str(stop) for stop in route.get("stops", []) if str(stop) in graph.nodes]
        for stop in stops:
            stop_to_routes.setdefault(stop, []).append(route_id)
        if len(stops) < 2:
            continue
        lons, lats = _path_coordinates(graph, stops)
        fig.add_trace(go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode="lines",
            line=dict(width=2.4, color="rgba(30, 136, 229, 0.62)"),
            name="Bus routes",
            legendgroup="bus_routes",
            showlegend=route_id == data.bus_routes[0].get("route_id"),
            hoverinfo="text",
            text=(
                f"<b>Bus route {route_id}</b>"
                f"<br>Buses assigned: {route.get('buses_assigned')}"
                f"<br>Daily passengers: {route.get('daily_passengers'):,}"
                f"<br>Stops: {' → '.join(data.path_names(stops))}"
            ),
        ))

    if stop_to_routes:
        bus_lons, bus_lats, bus_text, bus_labels = [], [], [], []
        for stop, routes in sorted(stop_to_routes.items(), key=lambda item: graph.nodes[item[0]].get("name")):
            lon, lat = _node_lon_lat(graph, stop)
            bus_lons.append(lon)
            bus_lats.append(lat)
            bus_labels.append("🚌")
            bus_text.append(
                f"<b>Bus stop: {graph.nodes[stop].get('name')} ({stop})</b>"
                f"<br>Routes: {', '.join(routes)}"
            )
        fig.add_trace(go.Scattermapbox(
            lon=bus_lons,
            lat=bus_lats,
            mode="markers+text",
            text=bus_labels,
            textposition="bottom center",
            marker=dict(size=12, color="#FDD835"),
            name="Bus stops",
            hoverinfo="text",
            hovertext=bus_text,
        ))

    # Highlight MST network if selected.
    if mst_graph is not None:
        first = True
        for u, v in mst_graph.edges():
            lon0, lat0 = _node_lon_lat(graph, u)
            lon1, lat1 = _node_lon_lat(graph, v)
            fig.add_trace(go.Scattermapbox(
                lon=[lon0, lon1],
                lat=[lat0, lat1],
                mode="lines",
                line=dict(width=5.2, color="#00B0FF"),
                name="Kruskal MST",
                legendgroup="mst",
                showlegend=first,
                hoverinfo="text",
                text=f"MST edge: {data.node_label(u)} ↔ {data.node_label(v)}",
            ))
            first = False

    # Highlight selected route from the chosen algorithm.
    if highlighted_path and len(highlighted_path) > 1:
        lons, lats = _path_coordinates(graph, highlighted_path)
        path_labels = data.path_names(highlighted_path)
        fig.add_trace(go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode="lines+markers+text",
            line=dict(width=6.5, color="#FF1744"),
            marker=dict(size=10, color="#FF1744"),
            name="Selected algorithm route",
            hoverinfo="text",
            text=path_labels,
            textposition="bottom right",
            hovertext=[f"<b>Selected route node</b><br>{label}<br><br>Full route:<br>{path_to_text(data, highlighted_path)}" for label in path_labels],
        ))

    # Location markers.
    neighborhood_lons, neighborhood_lats, neighborhood_names, neighborhood_hover = [], [], [], []
    facility_lons, facility_lats, facility_names, facility_hover = [], [], [], []
    for node, attrs in graph.nodes(data=True):
        lon, lat = _node_lon_lat(graph, node)
        label = f"{attrs.get('name')} ({node})"
        hover = _node_hover(attrs, node)
        if attrs.get("node_type") == "neighborhood":
            neighborhood_lons.append(lon)
            neighborhood_lats.append(lat)
            neighborhood_names.append(label)
            neighborhood_hover.append(hover)
        else:
            facility_lons.append(lon)
            facility_lats.append(lat)
            facility_names.append(label)
            facility_hover.append(hover)

    fig.add_trace(go.Scattermapbox(
        lon=neighborhood_lons,
        lat=neighborhood_lats,
        mode="markers+text",
        text=neighborhood_names,
        textposition="top center",
        marker=dict(size=14, color="#2E7D32"),
        name="Neighborhoods",
        hoverinfo="text",
        hovertext=neighborhood_hover,
    ))
    fig.add_trace(go.Scattermapbox(
        lon=facility_lons,
        lat=facility_lats,
        mode="markers+text",
        text=facility_names,
        textposition="top center",
        marker=dict(size=14, color="#EF6C00"),
        name="Facilities",
        hoverinfo="text",
        hovertext=facility_hover,
    ))

    lons = [attrs["pos"][0] for _, attrs in graph.nodes(data=True)]
    lats = [attrs["pos"][1] for _, attrs in graph.nodes(data=True)]
    center = {"lon": sum(lons) / len(lons), "lat": sum(lats) / len(lats)}

    fig.update_layout(
        title="Realistic Cairo Transportation Map - OpenStreetMap GPS View",
        height=720,
        margin=dict(l=5, r=5, t=42, b=5),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        mapbox=dict(
            style="open-street-map",
            center=center,
            zoom=8.55,
        ),
    )
    return fig


data = load_project_data()
graph = data.graph

st.title("🚗 Cairo Smart City Transportation Optimizer")
st.caption("Dijkstra, A*, Greedy, Time-dependent Dijkstra, Kruskal MST, Dynamic Programming, Traffic Signals, Public Transit, and Random Forest forecasting.")

left, right = st.columns([0.32, 0.68], gap="large")

with left:
    st.subheader("Inputs")
    options = label_options(data)
    labels = list(options.keys())
    default_start = next((i for i, label in enumerate(labels) if "Nasr City" in label), 0)
    default_end = next((i for i, label in enumerate(labels) if "6th October City" in label), min(8, len(labels) - 1))
    start_label = st.selectbox("Start", labels, index=default_start)
    end_label = st.selectbox("Target", labels, index=default_end)
    algorithm = st.selectbox("Algorithm", ["Dijkstra - shortest distance", "A* - fastest travel time", "Greedy - local shortest distance", "Time-dependent Dijkstra", "Kruskal MST"])
    hour = st.slider("Hour for time-dependent traffic", 0, 23, 8)

source, target = options[start_label], options[end_label]
result = None
mst_result = None
highlighted_path = None

if algorithm == "Dijkstra - shortest distance":
    result = dijkstra_shortest_path(graph, source, target, weight="distance")
    highlighted_path = result.get("path")
elif algorithm == "A* - fastest travel time":
    result = astar_shortest_path(graph, source, target, weight="travel_time")
    highlighted_path = result.get("path")
elif algorithm == "Greedy - local shortest distance":
    result = greedy_best_first_path(graph, source, target, weight="distance")
    highlighted_path = result.get("path")
elif algorithm == "Time-dependent Dijkstra":
    result = time_dependent_dijkstra(graph, source, target, hour=hour)
    highlighted_path = result.get("path")
else:
    mst_result = kruskal_mst(graph, weight="construction_cost")

with right:
    st.plotly_chart(draw_network(data, highlighted_path=highlighted_path, mst_graph=mst_result["graph"] if mst_result else None), use_container_width=True)

if result:
    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", result["algorithm"])
    c2.metric("Cost", result["cost"] if result["cost"] != float("inf") else "∞")
    c3.metric("Visited nodes", result.get("visited_nodes", "-"))
    st.info(path_to_text(data, result.get("path")))

if mst_result:
    c1, c2, c3 = st.columns(3)
    c1.metric("MST edges", mst_result["edge_count"])
    c2.metric("Total construction weight", mst_result["total_weight"])
    c3.metric("Connected", "Yes" if mst_result["connected"] else "No")
    st.dataframe(prepare_road_dataframe(mst_result["edges"], data), use_container_width=True)

st.divider()
st.subheader("Algorithm Comparison")
comparison = [
    dijkstra_shortest_path(graph, source, target, weight="distance"),
    astar_shortest_path(graph, source, target, weight="travel_time"),
    greedy_best_first_path(graph, source, target, weight="distance"),
    time_dependent_dijkstra(graph, source, target, hour=hour),
]
comparison_rows = []
for item in comparison:
    comparison_rows.append({
        "Algorithm": item["algorithm"],
        "Weight/Period": item.get("weight") if item["algorithm"] != "Time-dependent Dijkstra" else item.get("period"),
        "Cost": item.get("cost"),
        "Visited Nodes": item.get("visited_nodes"),
        "Path": path_to_text(data, item.get("path")),
    })
st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)
st.caption("Different algorithms may sometimes return the same route, but they optimize different objectives and use different search strategies. Compare path, cost, visited nodes, and selected weight/period.")

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("Road Maintenance - DP Knapsack")
    budget = st.slider("Maintenance budget", 100, 5000, 1500, step=100)
    maintenance = optimize_road_maintenance(graph, budget=budget)
    st.metric("Used budget", maintenance["used_budget"])
    st.metric("Total urgency", maintenance["total_urgency"])
    st.dataframe(prepare_road_dataframe(maintenance["selected_roads"], data), use_container_width=True)

with c2:
    st.subheader("Public Transit - DP Knapsack")
    max_buses = st.slider("Available buses", 10, 180, 75, step=5)
    transit = optimize_public_transit(data.bus_routes, max_buses=max_buses)
    st.metric("Used buses", transit["used_buses"])
    st.metric("Passengers served", transit["total_passengers"])
    st.dataframe(transport_routes_dataframe(transit, data), use_container_width=True)

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("Traffic Signal Optimization")
    north = st.number_input("North cars", 0, 10000, 50)
    south = st.number_input("South cars", 0, 10000, 30)
    east = st.number_input("East cars", 0, 10000, 70)
    west = st.number_input("West cars", 0, 10000, 20)
    counts = {"north": north, "south": south, "east": east, "west": west}
    normal_signal = optimize_traffic_signal(counts)
    emergency_direction = st.selectbox("Emergency direction", list(counts.keys()), index=2)
    emergency_signal = optimize_emergency_signal(counts, emergency_direction)
    st.write("Normal green time allocation")
    st.json(normal_signal)
    st.write("Emergency allocation")
    st.json(emergency_signal)

with c2:
    st.subheader("Random Forest Congestion Forecast")
    model, metrics, frame = train_ai(graph, data.traffic_patterns)
    roads = sorted(frame["road"].unique().tolist())
    road_display = {}
    for rid in roads:
        try:
            u, v = rid.split("-")
            road_display[f"{rid} | {road_label(data, u, v)}"] = rid
        except ValueError:
            road_display[rid] = rid
    selected_road_label = st.selectbox("Road", list(road_display.keys()))
    road_id = road_display[selected_road_label]
    pred_hour = st.slider("Prediction hour", 0, 23, 8, key="pred_hour")
    prediction = predict_congestion(model, graph, road_id, pred_hour, data.traffic_patterns)
    st.metric("Predicted traffic count", prediction)
    st.write("Model metrics")
    st.json(metrics)
