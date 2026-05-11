# Cairo Smart City Transportation Optimizer

A complete smart-city transportation optimization project for Cairo. The project models neighborhoods, facilities, roads, traffic, and public transport as a graph, then applies classic algorithms and AI forecasting.

## Implemented Requirements

| Requirement | Implementation | Main File |
|---|---|---|
| Shortest path | Dijkstra using `distance` or `travel_time` weights | `core/algorithms.py` |
| Guided shortest path | A* using geographic heuristic | `core/algorithms.py` |
| Greedy path | Local cheapest-neighbor route for comparison | `core/algorithms.py` |
| Time-based routing | Time-dependent Dijkstra using morning/afternoon/evening/night traffic | `core/algorithms.py` |
| Infrastructure design | Kruskal Minimum Spanning Tree | `core/algorithms.py` |
| Maintenance planning | Dynamic Programming / 0-1 Knapsack under budget | `core/algorithms.py` |
| Traffic signal optimization | Proportional green-light allocation + emergency priority | `core/algorithms.py` |
| Public transit scheduling | Dynamic Programming / Knapsack using available buses | `core/algorithms.py` |
| Traffic AI | Random Forest regression for traffic/congestion forecasting | `core/traffic_ai.py` |
| GUI | Streamlit dashboard with a real OpenStreetMap GPS-style Cairo map, bus stops, metro lines, routes, and result tables | `app.py` |

## Realistic Map Visualization

The dashboard map uses OpenStreetMap street tiles to show Cairo in a GPS-style view instead of a plain coordinate chart. It overlays:

- Existing roads, proposed roads, and facility access roads
- Highlighted algorithm route for Dijkstra, A*, Greedy, and Time-dependent Dijkstra
- Kruskal MST network when MST is selected
- Bus routes and bus-stop markers
- Metro lines and metro stations
- Neighborhood and facility markers with hover details

The map tiles load from the internet when the app is running, so the machine or deployed web app should have internet access for the full GPS-style background.

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Run Demo from Terminal

```bash
python run_demo.py
```

This prints sample results for all algorithms, so the project can be checked without opening the web dashboard.

## Run Tests

```bash
pytest -q
```

## Docker Deployment

```bash
docker build -t cairo-smart-transport .
docker run -p 8501:8501 cairo-smart-transport
```

## Important Algorithm Notes

- Dijkstra and A* are both optimal when they optimize the same non-negative weight, so they can return the same path. The dashboard compares them using different objectives: Dijkstra for shortest distance and A* for fastest travel-time score.
- Greedy is intentionally not guaranteed optimal; it is included to show how local decisions can differ from optimal algorithms.
- Time-dependent Dijkstra changes edge weights according to the selected hour: morning, afternoon, evening, or night.
- MST is not a route between two points. It designs a low-cost network connecting all nodes.
- Maintenance and public transit modules are both Dynamic Programming examples, but they solve different constraints: road budget vs available buses.

## Project Structure

```text
CairoProject_Final/
├── app.py
├── run_demo.py
├── requirements.txt
├── Dockerfile
├── README.md
├── core/
│   ├── __init__.py
│   ├── algorithms.py
│   ├── data_loader.py
│   └── traffic_ai.py
├── data/
│   ├── neighborhoods.json
│   ├── facilities.json
│   ├── roads.json
│   └── transport.json
└── tests/
    └── test_algorithms.py
```


## Final Accuracy Fixes

- Readable road-name columns were added beside road IDs in road-based result tables, for example: `2-3 | Nasr City → Downtown Cairo`.
- Dokki and Cairo University map coordinates were separated so selecting Dokki ends at the Dokki marker, not at Cairo University.
- Selected algorithm routes now show visible node labels on the GPS-style map for clearer demonstration.
